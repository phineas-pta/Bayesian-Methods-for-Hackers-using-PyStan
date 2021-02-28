# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, arviz as az, seaborn as sns, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

raw_data = pd.read_csv("data/challenger_data.csv")
raw_data["Date"] = pd.to_datetime(raw_data["Date"], infer_datetime_format=True)

# pop last row -> drop missing -> change dtype
challenger = raw_data.drop(raw_data.tail(1).index).dropna().astype({"Damage Incident": 'int32'})

mdl_data = {'N': len(challenger), 'dam': challenger["Damage Incident"].sum(), 'temp': challenger["Temperature"].values}

modelfile = "challenger.stan"
with open(modelfile, "w") as file: file.write("""
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
var_name = ["alpha", "beta"]

sm = CmdStanModel(stan_file = modelfile)
optim_raw = sm.optimize(data = mdl_data).optimized_params_dict
optim = {k: optim_raw[k] for k in var_name}
fit = sm.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit.draws().shape # iterations, chains, parameters
fit.summary().loc[var_name] # pandas DataFrame
fit.diagnose()

posterior = {k: fit.stan_variable(k) for k in var_name}

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[var_name] # pandas DataFrame
az.plot_trace(az_trace, var_names = var_name)
