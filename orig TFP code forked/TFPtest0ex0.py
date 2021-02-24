# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, tensorflow_probability as tfp
tfd, tfb = tfp.distributions, tfp.bijectors

N = 100
X = np.random.randn(N, 1)
beta, c, eps = 0.4, 1.5, 0.3
y = X*beta + c + np.random.randn(N, 1)*eps
plt.plot(X, y, '+')

def forward_random(n_replica):
	# prior sample for coefficient
	beta_samples = tfd.Normal(0., 10.).sample(n_replica)
	# prior sample for intercept
	c_samples = tfd.Normal(0., 10.).sample(n_replica)
	# prior sample for noise
	sigma_samples = tfd.Gamma(2., 5.).sample(n_replica)
	# linear function
	y_hat = beta_samples[tf.newaxis, ...] * X + c_samples[tf.newaxis, ...]
	# likelihood
	y_samples = tfd.Normal(y_hat, sigma_samples[tf.newaxis, ...]).sample()
	return beta_samples, c_samples, sigma_samples, y_hat, y_samples

def backward_logprob(beta, c, sigma):
	# likelihood for coefficient
	beta_logprob = tfd.Normal(0., 10.).log_prob(beta)
	# likelihood for intercept
	c_logprob = tfd.Normal(0., 10.).log_prob(c)
	# likelihood for noise
	sigma_logprob = tfd.Gamma(2., 5.).log_prob(sigma)
	# linear function
	y_hat = beta[tf.newaxis, ...] * X + c[tf.newaxis, ...]
	# likelihood
	y_logprob = tfd.Normal(y_hat, sigma[tf.newaxis, ...]).log_prob(y) # not the same shape
	return beta_logprob + c_logprob + sigma_logprob + tf.reduce_sum(y_logprob, axis = 0)

# grid search
eps_conditional = 1.
beta_min, beta_max = 0., 1.
c_min, c_max = 1., 2.
beta_grid, c_grid = np.meshgrid(
	np.linspace(beta_min, beta_max, N),
	np.linspace(c_min, c_max, N)
)
log_prob_grid = np.reshape(
	backward_logprob(*[tf.cast(x, tf.float32) for x in [beta_grid.flatten(), c_grid.flatten(), np.ones(N**2)*eps_conditional]]),
	[N, N]
)
plt.imshow(log_prob_grid, origin='lower', cmap='viridis', extent=[beta_min, beta_max, c_min, c_max], aspect='auto')
plt.colorbar(label = "log prob")
plt.scatter(beta, c)
plt.xlabel('Coefficient')
plt.ylabel('Intercept')
