# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors
tf.config.optimizer.set_jit(True)

N, p = 1000, .1 # 1000 subjects, 100 cheaters
true_answers = tfd.Bernoulli(probs=p).sample(sample_shape=N) # unknown
n1_coin_flip = tfd.Bernoulli(probs=0.5).sample(sample_shape=N) # unknown
n2_coin_flip = tfd.Bernoulli(probs=0.5).sample(sample_shape=N) # unknown

# subject responds YES if (n1 toss = heads & cheater) or (n1 toss = tails & n2 = heads)
observed_yes = n1_coin_flip * true_answers + (1 - n1_coin_flip) * n2_coin_flip
np.mean(observed_yes) # ≈ .5*p + .5²

# EXPLAINATION why prob_yes = .5*prob_cheat + .5² (0.5 = prob flip coin)
# ┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
# |             |                  └ 2nd flip = heads » answer = YES
# |             └ 1st flip = heads                    » answer = no
# └ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
#               |                  └ 2nd flip = heads » answer = YES
#               └ 1st flip = heads                    » answer = YES

n_yes = tf.cast(tf.reduce_sum(observed_yes), tf.float32)

def prob_flip_heads():
	return tfd.Binomial(total_count = N, probs = 0.5).sample() / N
def prob_yes(prob_cheat):
	prob_heads_1st = prob_flip_heads() # unknown so we’ll have to simulate
	prob_heads_2nd = prob_flip_heads() # unknown so we’ll have to simulate
	return prob_heads_1st * prob_cheat + (1. - prob_heads_1st) * prob_heads_2nd

@tfd.JointDistributionCoroutineAutoBatched
def mdl_batch():
	prob_cheat = yield tfd.Uniform(low = 0., high = 1., name="prob_cheat")
	n_yes = yield tfd.Binomial(total_count = N, probs = prob_yes(prob_cheat), name="n_yes")

number_of_steps, burnin = 40000, 10000
initial_chain_state = [.4]
target_log_prob_fn = lambda *args: mdl_batch.log_prob(n_yes=n_yes, *args)

metropolis = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn = target_log_prob_fn)

[posterior_prob] = tfp.mcmc.sample_chain(
	num_results = number_of_steps,
	num_burnin_steps = burnin,
	current_state = initial_chain_state,
	kernel = metropolis,
	trace_fn = None
)

sns.displot(posterior_prob[burnin:], bins = "sqrt", kde = True)
