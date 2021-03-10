# -*- coding: utf-8 -*-

import numpy as np, scipy, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

data = np.loadtxt("data/mixture_data.csv")
mdl_data = {"N": len(data), "obs": data}

modelfile = "mixture.stan"
with open(modelfile, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector[N] obs;
	}

	transformed data {
		int<lower=0> n_class = 2;
	}

	parameters { // discrete parameters impossible
		simplex[n_class] class_prob;
		ordered[n_class] centers;
		vector<lower=0>[n_class] sigmas;
	}

	model {
		centers[1] ~ normal(120, 10);
		centers[2] ~ normal(190, 10);

		sigmas ~ uniform(0, 100);

		for (n in 1:N) { // marginalize out the discrete parameter
			vector[n_class] lps = log(class_prob);
			for (k in 1:n_class)
				lps[k] += normal_lpdf(obs[n] | centers[k], sigmas[k]);
			target += log_sum_exp(lps);
		}
	}
""")

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

posterior = fit.stan_variables()

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[vb_name] # pandas DataFrame
az.plot_trace(az_trace)

# Compute the un-normalized log probabilities for each cluster
cluster_0_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(data, axis = 1),
	loc = np.expand_dims(posterior["centers"][:, 0], axis = 0),
	scale = np.expand_dims(posterior["sigmas"][:, 0], axis = 0)
) + np.log(posterior["class_prob"][:, 0])
cluster_1_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(data, axis = 1),
	loc = np.expand_dims(posterior["centers"][:, 1], axis = 0),
	scale = np.expand_dims(posterior["sigmas"][:, 1], axis = 0)
) + np.log(posterior["class_prob"][:, 1])

# Bayes rule to compute the assignment probability: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_1 = cluster_1_log_prob - np.logaddexp(cluster_0_log_prob, cluster_1_log_prob)

# Average across the MCMC chain
log_p_assign_1bis = scipy.special.logsumexp(log_p_assign_1, axis=-1) - np.log(log_p_assign_1.shape[-1])

p_assign_1 = np.exp(log_p_assign_1bis)
assign_trace = p_assign_1[np.argsort(data)]

plt.figure(figsize = (8, 6), tight_layout = True)
plt.scatter(data[np.argsort(data)], assign_trace, c = 1 - assign_trace, cmap = "RdBu")
plt.title("Probability of data point belonging to cluster 1")
plt.ylabel("probability")
plt.xlabel("value of data point")
