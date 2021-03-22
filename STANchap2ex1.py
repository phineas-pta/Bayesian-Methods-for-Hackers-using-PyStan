# -*- coding: utf-8 -*-

import numpy as np, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

count_data = np.array([
    13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
    19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
])
mdl_data = {"N": len(count_data), "obs": count_data}

# original model: slower
modelfile = "count.stan"
with open(modelfile, "w") as file: file.write("""
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

		for (i in 1:N) obs[i] ~ poisson(i < tau ? lambda1 : lambda2);
	}
""")

sm = CmdStanModel(stan_file = modelfile)

# model copied from stan docs: a lot faster
modelfile_modif = "count_modif.stan"
with open(modelfile_modif, "w") as file: file.write("""
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

sm_modif = CmdStanModel(stan_file = modelfile_modif)
var_name = ["lambda1", "lambda2", "tau"]

# maximum likelihood estimation
optim_modif = sm_modif.optimize(data = mdl_data).optimized_params_pd
optim_modif[optim_modif.columns[~optim_modif.columns.str.startswith("lp")]]

# variational inference
vb_modif = sm_modif.variational(data = mdl_data)
vb_modif.variational_sample.columns = vb_modif.variational_params_dict.keys()
vb_name = vb_modif.variational_params_pd.columns[~vb_modif.variational_params_pd.columns.str.startswith(("lp", "log_"))]
vb_modif.variational_params_pd[vb_name]
vb_modif.variational_sample[vb_name]

# Markov chain Monte Carlo
fit_modif = sm_modif.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_modif.draws().shape # iterations, chains, parameters
fit_modif.summary().loc[vb_name] # pandas DataFrame
print(fit_modif.diagnose())

posterior = {k: fit_modif.stan_variable(k) for k in var_name}

az_trace = az.from_cmdstanpy(fit_modif)
az.summary(az_trace).loc[vb_name] # pandas DataFrame
az.plot_trace(az_trace, var_names = var_name)
