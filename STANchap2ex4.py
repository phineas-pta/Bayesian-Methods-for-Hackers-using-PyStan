# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, seaborn as sns, matplotlib.pyplot as plt

raw_data = pd.read_csv("https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv")
raw_data["Date"] = pd.to_datetime(raw_data["Date"], infer_datetime_format=True)

# pop last row -> drop missing -> change dtype
challenger = raw_data.drop(raw_data.tail(1).index).dropna().astype({"Damage Incident": 'int32'})

sm = pystan.StanModel(model_name = "OLS", model_code = """
	data {
		int<lower=0> N;
		vector[N] temp;
		int<lower=0> dam;
	}

	parameters { // discrete parameters impossible
		real alpha;
		real beta;
	}

	transformed parameters {
		vector[N] prob;
		prob = 1 ./ (1 + exp(beta * temp + alpha)); // element-wise
	}

	model {
		alpha ~ normal(0, 1000);
		beta ~ normal(0, 1000);
		dam ~ binomial(N, prob);
	}
""")

fit = sm.sampling(
	data = {'N': len(challenger), 'dam': challenger["Damage Incident"].sum(), 'temp': challenger["Temperature"].values},
	pars = ["alpha", "beta"], iter = 50000, chains = 10, warmup = 10000, thin = 10, n_jobs = -1 # parallel
)
print(fit)
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace)
