# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

count_data = tf.constant([
    13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
    19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
], dtype = tf.float32)

n = tf.size(count_data, out_type = tf.float32)

var_name = ["lambda1", "lambda2", "tau"]

@tfd.JointDistributionCoroutineAutoBatched
def mdl_batch():
	lamb1 = yield tfd.Gamma(concentration=8., rate=0.3, name="lambda1")
	lamb2 = yield tfd.Gamma(concentration=8., rate=0.3, name="lambda2")
	tau = yield tfd.Uniform(low=0., high=1., name="tau")
	rate = tf.gather([lamb1, lamb2], indices = tf.cast(tau*n <= tf.range(n), dtype=tf.int32)) # vectorize ifelse
	Y = yield tfd.Poisson(rate=rate,name="Y")

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
		target_log_prob_fn = lambda *args: mdl_batch.log_prob(Y=count_data, *args),
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
init_state = [mdl_batch.sample(nchain)._asdict()[_] for _ in var_name]
unconstraining_bijectors = [tfb.Exp()]*2 + [tfb.Sigmoid()]
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)
