# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt
rng = np.random.default_rng(seed = 123) # newly introduced type of random generator

pA, N = .05, 1500
occurrences = rng.binomial(N, pA)
mdl_data = {"N": N, "occur": occurrences}

sm = pystan.StanModel(model_name = "simple_mdl", model_code = """
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
optim = sm.optimizing(data = mdl_data)
fit = sm.sampling(
	data = mdl_data, n_jobs = -1, # parallel
	iter = 50000, chains = 10, warmup = 10000, thin = 5
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace)
