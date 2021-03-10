# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, pandas as pd,\
       matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns,\
       arviz as az, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

baseball = pd.read_csv("data/EfronMorrisBB.txt", sep = "\t")
X = tf.cast(baseball["At-Bats"].values, "float32")
Y = tf.cast(baseball["Hits"].values, "float32")
var_name = ['phi', 'kappa_log', 'thetas']

@tfd.JointDistributionCoroutineAutoBatched
def mdl_baseball():
	phi = yield tfd.Uniform(low = 0., high = 1., name="phi")
	kappa_log = yield tfd.Exponential(rate = 1.5, name="kappa_log")
	alpha = tf.exp(kappa_log) * phi
	beta = tf.exp(kappa_log) * (1. - phi)
	theta = tfd.Beta(concentration0 = beta, concentration1 = alpha) # weird
	thetas = yield tfd.Sample(theta, sample_shape = len(baseball), name="thetas")
	likelihood = yield tfd.Binomial(total_count = X, probs = thetas, name="y")

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
		target_log_prob_fn = lambda *x: mdl_baseball.log_prob(y=Y, *x),
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
init_state = [mdl_baseball.sample(nchain)._asdict()[_] for _ in var_name]
unconstraining_bijectors = [tfb.Sigmoid(), tfb.Exp(), tfb.Sigmoid(),]
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)

#%% using the pymc3 naming convention, with log_likelihood instead of lp so that ArviZ can compute loo and waic
sample_stats_name = ['log_likelihood', 'tree_size', 'diverging', 'energy', 'mean_tree_accept']

sample_stats = {k: v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}
posterior = {k:np.swapaxes(v.numpy(), 1, 0) for k, v in zip(var_name, samples)}
az_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)
