# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, pandas as pd,\
       matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns,\
       arviz as az, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)
mpl.rc("figure", **{"figsize": (8, 6), "autolayout": True})

#%% data
dfhogg = pd.DataFrame(dict(
	x = [201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146],
	y = [592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344],
	sigma_x = [9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5.],
	sigma_y = [61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22],
	rho_xy = [-.84, .31, .64, -.27, -.33, .67, -.02, -.05, -.84, -.69, .3, -.46, -.03, .5, .73, -.52, .9, .4, -.78, -.56]
), dtype = "float32")
dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")

X_np = dfhogg['x'].values
xrange = np.array([X_np.min() - 1, X_np.max() + 1])
sigma_y_np = dfhogg['sigma_y'].values
Y_np = dfhogg['y'].values

#%% OLS model
# not batch friendly (to run 1 chain only, not multiple chains)
_mdl_ols = tfd.JointDistributionNamed(dict( # ATTENTION: a dict
	b0 = tfd.Normal(loc = 0., scale = 1.),
	b1 = tfd.Normal(loc = 0., scale = 1.),
	likelihood = lambda b0, b1: tfd.Independent( # to ensure the log_prob is not incorrectly broadcasted
		tfd.Normal(loc = b0 + b1*X_np, scale = sigma_y_np),
		reinterpreted_batch_ndims = 1
	)
))
param_sample = _mdl_ols.sample() # ATTENTION: a dict
_mdl_ols.log_prob(param_sample)

# manual batched version
_mdl_ols_batch = tfd.JointDistributionNamed(dict( # ATTENTION: a dict
	b0 = tfd.Normal(loc = 0., scale = 1.),
	b1 = tfd.Normal(loc = 0., scale = 1.),
	likelihood = lambda b0, b1: tfd.Independent(
		tfd.Normal( # add more dims by broadcasting
			loc = b0[..., tf.newaxis] + b1[..., tf.newaxis]*X_np[tf.newaxis, ...],
			scale = sigma_y_np[tf.newaxis, ...]
		),
		reinterpreted_batch_ndims = 1
	)
))

# auto batched version
mdl_ols_batch = tfd.JointDistributionNamedAutoBatched(dict( # ATTENTION: a dict
	likelihood = lambda b0, b1: tfd.Normal(loc = b0 + b1*X_np, scale = sigma_y_np), # implicit operations in background
	b0 = tfd.Normal(loc = 0., scale = 1.), # doesn't need to be in order
	b1 = tfd.Normal(loc = 0., scale = 1.),
))

# other way
@tfd.JointDistributionCoroutineAutoBatched
def model():
	b0 = yield tfd.Normal(loc = 0., scale = 1., name = "b0")
	b1 = yield tfd.Normal(loc = 0., scale = 1., name = "b1")
	yhat = b0 + b1*X_np
	likelihood = yield tfd.Normal(loc = yhat, scale = sigma_y_np, name = "yhat")

model.sample()

#%% MLE
b0_est_MLE, b1_est_MLE = tfp.optimizer.lbfgs_minimize(
	lambda x: tfp.math.value_and_gradient( # negative log likelihood
		lambda x: -tf.squeeze(model.log_prob(yhat=Y_np, *x)),
		x
	),
	initial_position = tf.zeros(2, dtype = tf.float32),
	tolerance = 1e-20,
	x_tolerance = 1e-8
).position.numpy()

dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")
plt.plot(xrange, b0_est_MLE + b1_est_MLE * xrange, c = "r", label = "OLS")
plt.legend(loc = "lower right")

#%% McMC
var_name = ['b0', 'b1']

# a helper function in McMC chain
def trace_fn(current_state, kernel_results):
	mdr = kernel_results.inner_results.inner_results
	return mdr.target_log_prob, mdr.leapfrogs_taken, mdr.has_divergence, mdr.energy, mdr.log_accept_ratio

@tf.function(autograph = False, experimental_compile = True) # speed up a lot the McMC sampling
def run_mcmc( # pass numeric arguments as Tensors whenever possible
	init_state, unconstraining_bijectors,
	num_steps = tf.constant(50000), burnin = tf.constant(10000),
	num_leapfrog_steps = tf.constant(3), step_size = tf.constant(.5)
):
	kernel0 = tfp.mcmc.NoUTurnSampler(
		target_log_prob_fn = lambda *args: model.log_prob(yhat=Y_np, *args),
		step_size = step_size
	)
	kernel1 = tfp.mcmc.TransformedTransitionKernel(
		inner_kernel= kernel0,
		bijector = unconstraining_bijectors
	)
	kernel2 = tfp.mcmc.DualAveragingStepSizeAdaptation( # pkr = previous kernel results
		inner_kernel = kernel1,
		num_adaptation_steps = tf.cast(tf.constant(0.8)*tf.cast(burnin, dtype= tf.float32), dtype = tf.int32),
		step_size_setter_fn = lambda pkr, new_step_size: pkr._replace(inner_results = pkr.inner_results._replace(step_size=new_step_size)),
		step_size_getter_fn = lambda pkr: pkr.inner_results.step_size,
		log_accept_prob_getter_fn = lambda pkr: pkr.inner_results.log_accept_ratio
	)
	# tf.get_logger().setLevel("ERROR") # multiple chains
	return tfp.mcmc.sample_chain( # ATTENTION: 2 values to unpack
		num_results = num_steps,
		num_burnin_steps = burnin,
		current_state = init_state,
		kernel = kernel2,
		trace_fn = trace_fn
	)

nchain = 5
init_state = [model.sample(nchain)._asdict()[_] for _ in var_name]
unconstraining_bijectors = [tfb.Identity()]*len(var_name) # map contrained parameters to real (2 params)
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)

#%% using the pymc3 naming convention, with log_likelihood instead of lp so that ArviZ can compute loo and waic
sample_stats_name = ['log_likelihood', 'tree_size', 'diverging', 'energy', 'mean_tree_accept']

sample_stats = {k: v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}
# sample_stats['tree_size'] = np.diff(sample_stats['tree_size'], axis=1) # multilple chains
posterior = {k: v.numpy() for k, v in zip(var_name, samples)}
# posterior = {k:np.swapaxes(v.numpy(), 1, 0) for k, v in zip(var_name, samples)} # multilple chains
az_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)

sns.displot(posterior["b0"], bins = 100, kde = True)
tf.reduce_mean(posterior["b0"])

az.summary(az_trace)
az.plot_trace(az_trace)

# Student T likelihood: fatter tails: allows outliers to have a smaller influence in the likelihood estimation
@tfd.JointDistributionCoroutineAutoBatched
def mdl_studentt():
	b0 = yield tfd.Normal(loc = 0., scale = 1., name="b0")
	b1 = yield tfd.Normal(loc = 0., scale = 1., name="b1")
	df = yield tfd.Uniform(low = 1, high = 100, name="df")
	likelihood = yield tfd.StudentT(loc = b0 + b1*X_np, scale = sigma_y_np, df = df, name="yhat")

b0_est_StT, b1_est_StT, df_est_StT = tfp.optimizer.lbfgs_minimize(
	lambda x: tfp.math.value_and_gradient( # negative log likelihood
		lambda x: -tf.squeeze(mdl_studentt.log_prob(yhat=Y_np, *x)),
		x
	),
	initial_position = [2., 1., 20.],
	tolerance = 1e-20,
	x_tolerance = 1e-20
).position.numpy()

dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")
plt.plot(xrange, b0_est_MLE + b1_est_MLE * xrange, c = "r", label = "OLS")
plt.plot(xrange, b0_est_StT + b1_est_StT * xrange, c = "g", label = "Student-T")
plt.legend(loc = "lower right")

# bijector to map contrained parameters to real
a, b = tf.constant(1.), tf.constant(100.)
c, d = tf.constant(b - a), tf.constant(tf.math.log(b - a))
unconstraining_bijectors = [
	tfb.Identity(), tfb.Identity(),
	tfb.Inline(
		inverse_fn = lambda x: tf.math.log(x - a) - tf.math.log(b - x),
		forward_fn = lambda x: c * tf.sigmoid(x) + a,
		forward_log_det_jacobian_fn = lambda x: d - 2 * tf.nn.softplus(-x) - x,
		forward_min_event_ndims = 0,
	),
]
