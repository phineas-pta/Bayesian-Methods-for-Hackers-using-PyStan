# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt

count_data = np.array([
    13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
    19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
])
mdl_data = {"N": len(count_data), "obs": count_data}

# original model: slower
sm = pystan.StanModel(model_name = "std_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> obs[N];
	}

	transformed data {
		real alpha = 8.;
		real beta = .3;
	}

	parameters { // discrete parameters impossible
		real<lower=0> lambda1;
		real<lower=0> lambda2;
		real<lower=0> tau;
	}

	model {
		lambda1 ~ gamma(alpha, beta);
		lambda2 ~ gamma(alpha, beta);
		tau ~ uniform(1, N);

		for (i in 1:N) {
			if (i < tau) {
				obs[i] ~ poisson(lambda1);
			} else {
				obs[i] ~ poisson(lambda2);
			}
		}
	}
""")
optim = sm.optimizing(data = mdl_data)
fit = sm.sampling(
	data = mdl_data, n_jobs = -1, # parallel
	iter = 50000, chains = 3, warmup = 10000, thin = 5
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace)

# model copied from stan docs: faster a lot
sm_modif = pystan.StanModel(model_name = "modif_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> obs[N];
	}

	transformed data {
		real alpha = 8.;
		real beta = .3;
		real log_unif = -log(N);
	}

	parameters { // no info about tau
		real<lower=0> lambda1;
		real<lower=0> lambda2;
	}

	transformed parameters { // marginalize out the discrete parameter
		vector[N] lp = rep_vector(log_unif, N);
		vector[N+1] lp1;
		vector[N+1] lp2;
		lp1[1] = 0;
		lp2[1] = 0;
		for (i in 1:N) { // dynamic programming workaround to avoid nested loop
			lp1[i+1] = lp1[i] + poisson_lpmf(obs[i] | lambda1);
			lp2[i+1] = lp2[i] + poisson_lpmf(obs[i] | lambda2);
		}
		lp = rep_vector(log_unif + lp2[N+1], N) + head(lp1, N) - head(lp2, N);
	}

	model {
		lambda1 ~ gamma(alpha, beta);
		lambda2 ~ gamma(alpha, beta);
		target += log_sum_exp(lp);
	}

	generated quantities { // generate tau here
		int<lower=1,upper=N> tau = categorical_logit_rng(lp);
	}
""")

fit_modif = sm_modif.sampling(
	data = mdl_data, n_jobs = -1, pars = ["lambda1", "lambda2", "tau"],
	iter = 50000, chains = 3, warmup = 10000, thin = 5
)
print(fit_modif.stansummary())
