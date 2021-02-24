# -*- coding: utf-8 -*-

import numpy as np, scipy, pystan, arviz as az, matplotlib.pyplot as plt

data = np.loadtxt("data/mixture_data.csv")
mdl_data = {"N": len(data), "obs": data}

sm = pystan.StanModel(model_name = "std_mdl", model_code = """
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
optim = sm.optimizing(data = mdl_data)
fit = sm.sampling(
	data = mdl_data, n_jobs = -1, # parallel
	iter = 50000, chains = 3, warmup = 10000, thin = 5
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit, compact = True)
az.summary(az_trace)
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
