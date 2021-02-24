# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, seaborn as sns
from matplotlib import pyplot as plt, axes
from tensorflow_probability import distributions as tfd, bijectors as tfb, mcmc

###############################################################################
# coin flipping

rv_coin_flip_prior = tfd.Bernoulli(probs = 0.5, dtype = tf.int32)
num_trials = tf.constant([0, 1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000])

# prepend a 0 onto tally of heads and tails, for zeroth flip
coin_flip_data = tf.pad(
	tensor = rv_coin_flip_prior.sample(num_trials[-1]), # flip 2000 times
	paddings = tf.constant([[1, 0,]]),
	mode = "CONSTANT"
)

# compute cumulative headcounts from 0 to 2000 flips, and then grab them at each of num_trials intervals
cumulative_headcounts = tf.gather(params = tf.cumsum(coin_flip_data), indices = num_trials)

rv_observed_heads = tfd.Beta(
	concentration1 = tf.cast(1 + cumulative_headcounts, dtype = tf.float32),
	concentration0 = tf.cast(1 + num_trials - cumulative_headcounts, dtype = tf.float32)
)

probs_of_heads = tf.linspace(start = 0., stop = 1., num = 100, name = "linspace")
observed_probs_heads = tf.transpose(rv_observed_heads.prob(probs_of_heads[:, tf.newaxis]))

# For the already prepared, I'm using Binomial's conj. prior.
plt.figure(figsize = (8, 6))
for i in range(len(num_trials)):
	sx = plt.subplot(len(num_trials)/2, 2, i+1)
	if i == len(num_trials)-1:
		plt.xlabel("$p$, probability of heads")
	plt.setp(sx.get_yticklabels(), visible = False)
	plt.plot(
		probs_of_heads, observed_probs_heads[i],
		label = "observe %d tosses,\n %d heads" % (num_trials[i], cumulative_headcounts[i])
	)
	plt.fill_between(
		probs_of_heads, 0, observed_probs_heads[i],
		color = '#5DA5DA', alpha = 0.4
	)
	plt.vlines(0.5, 0, 4, color = "k", linestyles = "--", lw = 1)
	leg = plt.legend()
	leg.get_frame().set_alpha(0.4)
	plt.autoscale(tight = True)

###############################################################################
# text messaging

# Defining our Data and assumptions
count_data = tf.constant([
    13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
    19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
], dtype = tf.float32)
n_count_data = tf.shape(count_data)
days = tf.range(n_count_data[0], dtype = tf.int32)

# Visualizing the Results
plt.figure(figsize = (12.5, 4))
plt.bar(days.numpy(), count_data, color = "#5DA5DA")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data[0].numpy())
