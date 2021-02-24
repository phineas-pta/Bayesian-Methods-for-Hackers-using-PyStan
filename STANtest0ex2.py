# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt

baseball = pd.read_csv("data/EfronMorrisBB.txt", sep = "\t")
mdl_data = {"N": len(baseball), "at_bats": baseball["At-Bats"].values, "hits": baseball["Hits"].values}
sm = pystan.StanModel(model_name = "std_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> at_bats[N];
		int<lower=0> hits[N];
	}

	parameters { // discrete parameters impossible
		real log_kappa;
		real<lower=0, upper=1> phi;
		vector<lower=0, upper=1>[N] thetas;
	}

	transformed parameters {
		real<lower=0> kappa = exp(log_kappa);
		real<lower=0> alpha = kappa * phi;
		real<lower=0> beta = kappa * (1 - phi);
	}

	model {
		log_kappa ~ exponential(1.5);
		phi ~ uniform(0, 1);
		thetas ~ beta(alpha, beta);
		hits ~ binomial(at_bats, thetas);
	}
""")
optim = sm.optimizing(data = mdl_data)
fit = sm.sampling(
	data = mdl_data, pars = ["kappa", "phi", "thetas"], n_jobs = -1, # parallel
	iter = 50000, chains = 2, warmup = 10000, thin = 5, # not too much chains for a smaller plot
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace, var_names = ["phi", "kappa"])

az.plot_forest(az_trace, var_names = ["thetas"])
plt.gca().set_yticklabels(baseball.apply(lambda x: x['FirstName'] + " " + x['LastName'], axis=1).tolist())
