# -*- coding: utf-8 -*-

import numpy as np, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
rng = np.random.default_rng(seed = 123) # newly introduced type of random generator

N, p, coin = 1000, .1, .5 # 1000 subjects, 100 cheaters, coin flip
true_answers = rng.binomial(1, p, size = N) # unknown
n1_coin_flip = rng.binomial(1, coin, size = N) # unknown
n2_coin_flip = rng.binomial(1, coin, size = N) # unknown
observed_yes = n1_coin_flip * true_answers + (1 - n1_coin_flip) * n2_coin_flip
n_yes = np.sum(observed_yes)
mdl_data = {"N": N, "occur": n_yes}

# EXPLAINATION why prob_yes = .5*prob_cheat + .5² (0.5 = prob flip coin)
# ┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
# |             |                  └ 2nd flip = heads » answer = YES
# |             └ 1st flip = heads                    » answer = no
# └ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
#               |                  └ 2nd flip = heads » answer = YES
#               └ 1st flip = heads                    » answer = YES

modelfile = "cheating.stan"
with open(modelfile, "w") as file: file.write("""
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
var_name = ["prob_cheat"]

sm = CmdStanModel(stan_file = modelfile)

# maximum likelihood estimation
optim = sm.optimize(data = mdl_data).optimized_params_pd
optim[optim.columns[~optim.columns.str.startswith("lp")]]

# variational inference
vb = sm.variational(data = mdl_data)
vb.variational_sample.columns = vb.variational_params_dict.keys()
vb_name = vb.variational_params_pd.columns[~vb.variational_params_pd.columns.str.startswith(("lp", "log_"))]
vb.variational_params_pd[vb_name]
vb.variational_sample[vb_name]

# Markov chain Monte Carlo
fit = sm.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit.draws().shape # iterations, chains, parameters
fit.summary().loc[vb_name] # pandas DataFrame
print(fit.diagnose())

posterior = {k: fit.stan_variable(k) for k in var_name}

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[vb_name] # pandas DataFrame
az.plot_trace(az_trace, var_names = var_name)
