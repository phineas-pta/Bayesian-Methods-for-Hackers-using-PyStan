# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, pandas as pd, arviz as az, seaborn as sns, tensorflow_probability as tfp
from matplotlib import pyplot as plt, dates as mdates
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

raw_data = pd.read_csv("data/challenger_data.csv")
raw_data["Date"] = pd.to_datetime(raw_data["Date"], infer_datetime_format=True)

# pop last row -> drop missing -> change dtype
challenger = raw_data.drop(raw_data.tail(1).index).dropna().astype({"Damage Incident": 'int32'})
challenger.plot.scatter(x = "Temperature", y = "Damage Incident")

challenger.plot.scatter(x = "Date", y = "Temperature", c = "Damage Incident", cmap = plt.cm.get_cmap('RdBu', 2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

dam = tf.cast(challenger["Damage Incident"], "float32")
temp = tf.cast(challenger["Temperature"], "float32")
var_name = ["alpha", "beta"]

@tfd.JointDistributionCoroutineAutoBatched
def mdl_batch():
	alpha = yield tfd.Normal(loc = 0., scale = 1000., name="alpha")
	beta = yield tfd.Normal(loc = 0., scale = 1000., name="beta")
	prob = 1./(1. + tf.exp(beta * temp + alpha))
	dam = yield tfd.Bernoulli(probs = prob, name="damage")

# a helper function in McMC chain
def trace_fn(current_state, kernel_results):
	mdr = kernel_results.inner_results.inner_results
	return mdr.target_log_prob, mdr.leapfrogs_taken, mdr.has_divergence, mdr.energy, mdr.log_accept_ratio

@tf.function(autograph = False, experimental_compile = True) # speed up a lot the McMC sampling
def run_mcmc( # pass numeric arguments as Tensors whenever possible
	init_state, unconstraining_bijectors,
	num_steps = 50000, burnin = 10000,
	num_leapfrog_steps = 3, step_size = .5
):
	kernel0 = tfp.mcmc.NoUTurnSampler(
		target_log_prob_fn = lambda *args: mdl_batch.log_prob(damage=dam, *args),
		step_size = step_size
	)
	kernel1 = tfp.mcmc.TransformedTransitionKernel(
		inner_kernel= kernel0,
		bijector = unconstraining_bijectors
	)
	kernel2 = tfp.mcmc.DualAveragingStepSizeAdaptation( # pkr = previous kernel results
		inner_kernel = kernel1,
		num_adaptation_steps = int(0.8*burnin),
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

nchain = 4
init_state = [mdl_batch.sample(nchain)._asdict()[_] for _ in var_name]
# Since HMC operates over unconstrained space, we need to transform the samples so they live in real-space.
# α ≈ 100×ß, so apply bijector to 100× the unconstrained α to get back to the original problem space
unconstraining_bijectors = [tfb.Scale(100.), tfb.Identity()]
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)

#%% using the pymc3 naming convention, with log_likelihood instead of lp so that ArviZ can compute loo and waic
sample_stats_name = ['log_likelihood', 'tree_size', 'diverging', 'energy', 'mean_tree_accept']

sample_stats = {k: v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}
posterior = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in zip(var_name, samples)}
az_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)

az.summary(az_trace)
az.plot_trace(az_trace)
