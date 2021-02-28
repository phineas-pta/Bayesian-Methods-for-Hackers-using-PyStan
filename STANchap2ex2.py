# -*- coding: utf-8 -*-

import numpy as np, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
rng = np.random.default_rng(seed = 123) # newly introduced type of random generator

pA, N = .05, 1500
occurrences = rng.binomial(N, pA)
mdl_data = {"N": N, "occur": occurrences}

modelfile = "ABtesting.stan"
with open(modelfile, "w") as file: file.write("""
	data {
		int<lower=0> N;
		int<lower=0, upper=N> occur;
	}

	parameters { // discrete parameters impossible
		real<lower=0, upper=1> probA;
	}

	model {
		occur ~ binomial(N, probA);
	}
""")

sm = CmdStanModel(stan_file = modelfile)
optim = sm.optimize(data = mdl_data).optimized_params_dict
fit = sm.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit.draws().shape # iterations, chains, parameters
fit.summary() # pandas DataFrame
fit.diagnose()

posterior = fit.stan_variables()

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace) # pandas DataFrame
az.plot_trace(az_trace)
