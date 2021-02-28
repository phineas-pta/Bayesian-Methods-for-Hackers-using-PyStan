# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

baseball = pd.read_csv("data/EfronMorrisBB.txt", sep = "\t")
mdl_data = {"N": len(baseball), "at_bats": baseball["At-Bats"].values, "hits": baseball["Hits"].values}

modelfile = "baseball.stan"
with open(modelfile, "w") as file: file.write("""
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
var_name = ["kappa", "phi", "thetas"]

sm = CmdStanModel(stan_file = modelfile)
optim_raw = sm.optimize(data = mdl_data).optimized_params_dict
optim = {k: optim_raw[k] for k in var_name}
fit = sm.sample(
	data = mdl_data, show_progress = True, chains = 2, # not too much chains for a smaller plot
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit.draws().shape # iterations, chains, parameters
fit.summary().loc[var_name] # pandas DataFrame
fit.diagnose()

posterior = {k: fit_modif.stan_variable(k) for k in var_name}

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[var_name] # pandas DataFrame
az.plot_trace(az_trace, var_names = ["phi", "kappa"])

az.plot_forest(az_trace, var_names = ["thetas"])
plt.gca().set_yticklabels(baseball.apply(lambda x: x['FirstName'] + " " + x['LastName'], axis=1).tolist())
