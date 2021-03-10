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
		vector[N] prob = 1 ./ (1 + exp(beta * temp + alpha)); // element-wise
	}

	model {
		alpha ~ normal(0, 1000);
		beta ~ normal(0, 1000);
		dam ~ binomial(N, prob);
	}
""")
var_name = ["alpha", "beta"]

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
