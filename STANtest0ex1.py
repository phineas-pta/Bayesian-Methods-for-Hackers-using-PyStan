# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, arviz as az, pystan, scipy

#%% data

dfhogg = pd.DataFrame(dict(
	x = [201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146],
	y = [592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344],
	sigma_x = [9, 4, 11, 7, 5, 9, 4, 4, 11, 7, 5, 5, 5, 6, 6, 5, 9, 8, 6, 5.],
	sigma_y = [61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22],
	rho_xy = [-.84, .31, .64, -.27, -.33, .67, -.02, -.05, -.84, -.69, .3, -.46, -.03, .5, .73, -.52, .9, .4, -.78, -.56]
), dtype = "float32")
dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")

X, Y, sigmaY = dfhogg['x'].values, dfhogg['y'].values, dfhogg['sigma_y'].values
model_data_dict = {"N": len(dfhogg), "X": X, "Y": Y, "sigmaY": sigmaY}
model_data_stan = """
	data {
		int<lower=0> N;
		vector[N] X;
		vector[N] Y;
		vector<lower=0>[N] sigmaY;
	}
"""
model_trans_params_stan = """
	transformed parameters {
		vector[N] Yhat = b0 + b1 * X;
	}
"""
var_name = ["b0", "b1"]

#%% models

# Simple Linear Model with no Outlier Correction
sm_ols = pystan.StanModel(model_name = "mdl_ols", model_code = model_data_stan + """
	parameters { // discrete parameters impossible
		real b0;
		real b1;
	}
""" + model_trans_params_stan + """
	model {
		Y ~ normal(Yhat, sigmaY);
	}
""")
optim_ols = sm_ols.optimizing(data = model_data_dict)
fit_ols = sm_ols.sampling(
	data = model_data_dict, pars = var_name,
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1 # parallel
)
print(fit_ols.stansummary())
sample_ols = fit_ols.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace_ols = az.from_pystan(posterior = fit_ols)
az.summary(az_trace_ols)
az.plot_trace(az_trace_ols)

gd = sns.jointplot(
	x = sample_ols["b0"], y = sample_ols["b1"],
	marginal_kws={"kde": True, "kde_kws": {"cut": 1}},
)
gd.plot_joint(sns.kdeplot, zorder = 2, n_levels = 10, cmap = "gray_r")
gd.fig.suptitle("Posterior joint distribution (OLS)", y = 1.02)

# Simple Linear Model with Robust Student-T Likelihood: outliers to have a smaller influence in the likelihood estimation
sm_studentt = pystan.StanModel(model_name = "mdl_studentt", model_code = model_data_stan + """
	parameters { // discrete parameters impossible
		real b0;
		real b1;
		real<lower=0> df;
	}
""" + model_trans_params_stan + """
	model {
		df ~ inv_gamma(1, 1);
		Y ~ student_t(df, Yhat, sigmaY);
	}
""")
optim_studentt = sm_studentt.optimizing(data = model_data_dict)
fit_studentt = sm_studentt.sampling(
	data = model_data_dict, pars = var_name + ["df"],
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1 # parallel
)
print(fit_studentt.stansummary())
sample_studentt = fit_studentt.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace_studentt = az.from_pystan(posterior = fit_studentt)
az.summary(az_trace_studentt)
az.plot_trace(az_trace_studentt)

# Linear Model with Custom Likelihood to Distinguish Outliers: Hogg Method
# idea: mixture model whereby datapoints can be: normal linear model vs outlier (for convenience also be linear)
sm_hogg = pystan.StanModel(model_name = "mdl_hogg", model_code = model_data_stan + """
	parameters { // discrete parameters impossible
		real b0;
		real b1;
		real Y_outlier; // mean for all outliers
		real<lower=0> sigmaY_outlier; // additional variance for outliers
		simplex[2] cluster_prob; // mixture ratio
	}
""" + model_trans_params_stan + """
	model {
		b0 ~ normal(0, 5); // weakly informative Normal priors (L2 ridge reg) for inliers
		b1 ~ normal(0, 5); // likewise

		Y_outlier ~ normal(0, 10);
		sigmaY_outlier ~ normal(0, 10); // half-normal because of above constraint

		for (n in 1:N) { // custom mixture model: cluster 1 = inlier, 2 = outlier
			real cluster1 = log(cluster_prob[1]) + normal_lpdf(Y[n] | Yhat[n], sigmaY[n]);
			real cluster2 = log(cluster_prob[2]) + normal_lpdf(Y[n] | Y_outlier, sigmaY[n] + sigmaY_outlier);
			target += log_sum_exp(cluster1, cluster2);
		}
	}
""")
optim_hogg = sm_hogg.optimizing(data = model_data_dict)
fit_hogg = sm_hogg.sampling(
	data = model_data_dict, pars = var_name + ["Y_outlier", "sigmaY_outlier", "cluster_prob"],
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1 # parallel
)
print(fit_hogg.stansummary())
sample_hogg = fit_hogg.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace_hogg = az.from_pystan(posterior = fit_hogg)
az.summary(az_trace_hogg)
az.plot_trace(az_trace_hogg)

#%% some plots

xrange = np.array([X.min() - 1, X.max()])
dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b", figsize = (8, 6))
for x, y, z in zip([sample_ols, sample_studentt, sample_hogg], ["r", "g", "m"], ["OLS", "Student-T", "Hogg method"]):
	b0, b1 = x["b0"].mean(), x["b1"].mean()
	plt.plot(xrange, b0 + b1 * xrange, c = y, label = f"{z}: $ y = {b0:.1f} + {b1:.1f}x $")
plt.legend(loc = "lower right")

for var in var_name:
	a, b, c = sample_ols[var], sample_studentt[var], sample_hogg[var]
	data = pd.DataFrame(data = {
		"y": np.concatenate((a, b, c)),
		"model": np.repeat([
			f"OLS: {a.mean():.1f}", f"Student-T: {b.mean():.1f}", f"Hogg method: {c.mean():.1f}"
		], 24000) # nb of data pts
	})
	g = sns.displot(data, x = "y", hue = "model", bins = "sqrt", kde = True, palette = ["r", "g", "m"])
	g.fig.suptitle(f"Posterior distribution: {var}")
	g.fig.set_figwidth(8)
	g.fig.set_figheight(6)

#%% declare outliers

# Compute the un-normalized log probabilities for each cluster
cluster_0_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(Y, axis = 1),
	loc = sample_hogg["b0"] + sample_hogg["b1"] * np.expand_dims(X, axis = 1),
	scale = np.expand_dims(sigmaY, axis = 1)
) + np.log(sample_hogg["cluster_prob"][:, 0])
cluster_1_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(Y, axis = 1),
	loc = sample_hogg["Y_outlier"],
	scale = np.expand_dims(sigmaY, axis = 1) + sample_hogg["sigmaY_outlier"]
) + np.log(sample_hogg["cluster_prob"][:, 1])

# Bayes rule to compute the assignment probability: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_1 = cluster_1_log_prob - np.logaddexp(cluster_0_log_prob, cluster_1_log_prob)

# Average across the MCMC chain
log_p_assign_1bis = scipy.special.logsumexp(log_p_assign_1, axis=-1) - np.log(log_p_assign_1.shape[-1])

p_assign_1 = np.exp(log_p_assign_1bis)
dfhogg["is_outlier"] = [f"{100*i:.2f} %" for i in p_assign_1]
dfhogg["classed_as_outlier"] = p_assign_1 >= .95
