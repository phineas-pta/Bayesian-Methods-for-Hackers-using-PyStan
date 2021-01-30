# -*- coding: utf-8 -*-

import numpy as np, pystan, arviz as az, matplotlib.pyplot as plt
rng = np.random.default_rng(seed = 123) # newly introduced type of random generator

N, p, coin = 1000, .1, .5 # 1000 subjects, 100 cheaters, coin flip
true_answers = rng.binomial(1, p, size = N) # unknown
n1_coin_flip = rng.binomial(1, coin, size = N) # unknown
n2_coin_flip = rng.binomial(1, coin, size = N) # unknown
observed_yes = n1_coin_flip * true_answers + (1 - n1_coin_flip) * n2_coin_flip
n_yes = np.sum(observed_yes)

# EXPLAINATION why prob_yes = .5*prob_cheat + .5² (0.5 = prob flip coin)
# ┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
# |             |                  └ 2nd flip = heads » answer = YES
# |             └ 1st flip = heads                    » answer = no
# └ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
#               |                  └ 2nd flip = heads » answer = YES
#               └ 1st flip = heads                    » answer = YES

sm = pystan.StanModel(model_name = "simple_mdl", model_code = """
	functions { // No lower-bound or upper-bound constraints are allowed
		real flip_rng(int N, real prob_coin) { // name must end with `_rng` when use other `_rng`
			int coin_flip;
			coin_flip = binomial_rng(N, prob_coin);
			return coin_flip * 1. / N; // trick to make int->real
		}
	}

	data {
		int<lower=0> N;
		int<lower=0, upper=N> occur;
	}

	transformed data {
		real<lower=0, upper=1> prob_coin = .5;
		real<lower=0, upper=1> flip1 = flip_rng(N, prob_coin);
		real<lower=0, upper=1> flip2 = flip_rng(N, prob_coin);
	}

	parameters { // discrete parameters impossible
		real<lower=0, upper=1> prob_cheat;
	}

	transformed parameters {
		real<lower=0, upper=1> prob_yes = flip1 * prob_cheat + (1 - flip1) * flip2;
	}

	model {
		occur ~ binomial(N, prob_yes);
	}
""")

fit = sm.sampling(
	data = {"N": N, "occur": n_yes}, pars = ["prob_cheat"], n_jobs = -1, # parallel
	iter = 50000, chains = 10, warmup = 10000, thin = 5
)
print(fit)
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace)
