# -*- coding: utf-8 -*-

import numpy as np, arviz as az, tensorflow as tf, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

pA, N = .05, 1500
occurrences = tfd.Bernoulli(probs = pA).sample(sample_shape = N, seed = 123)
occur_float = tf.cast(tf.reduce_sum(occurrences), dtype = "float32") # ATTENTION: binomial

@tfd.JointDistributionCoroutineAutoBatched
def mdl_batch():
	pA = yield tfd.Uniform(low=0., high=1., name="pA")
	occur = yield tfd.Binomial(total_count=N, probs=pA, name="occur")

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
		target_log_prob_fn = lambda *args: mdl_batch.log_prob(occur=occur_float, *args),
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
	return tfp.mcmc.sample_chain( # ATTENTION: 2 values to unpack
		num_results = num_steps,
		num_burnin_steps = burnin,
		current_state = init_state,
		kernel = kernel2,
		trace_fn = trace_fn
	)

nchain = 4
init_state = [mdl_batch.sample(nchain)._asdict()["pA"]]
unconstraining_bijectors = [tfb.Identity()]
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)

#%% using the pymc3 naming convention, with log_likelihood instead of lp so that ArviZ can compute loo and waic
sample_stats_name = ['log_likelihood', 'tree_size', 'diverging', 'energy', 'mean_tree_accept']

sample_stats = {k: v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}
posterior = {k: np.swapaxes(v.numpy(), 1, 0) for k, v in zip(var_name, samples)}
az_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)

az.summary(az_trace)
az.plot_trace(az_trace)
