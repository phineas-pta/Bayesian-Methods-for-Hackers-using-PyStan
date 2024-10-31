# -*- coding: utf-8 -*-

# src:
# - https://github.com/astroML/astroML/blob/main/astroML/datasets/hogg2010test.py
# - https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust-with-outlier-detection.html

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, arviz as az, scipy
from matplotlib.patches import Ellipse
from cmdstanpy import CmdStanModel

#%% data

dfhogg = pd.DataFrame(dict(
	x       = [ 201, 244,  47,  287,  203,  58,  210,  202,  198,  158, 165,  201,  157, 131, 166,  160, 186, 125, 218,   146],
	y       = [ 592, 401, 583,  402,  495, 173,  479,  504,  510,  416, 393,  442,  317, 311, 400,  337, 423, 334, 533,   344],
	sigma_x = [   9,   4,  11,    7,    5,   9,    4,    4,   11,    7,   5,    5,    5,   6,   6,    5,   9,   8,   6,     5],
	sigma_y = [  61,  25,  38,   15,   21,  15,   27,   14,   30,   16,  14,   25,   52,  16,  34,   31,  42,  26,  16,    22],
	rho_xy  = [-.84, .31, .64, -.27, -.33, .67, -.02, -.05, -.84, -.69,  .3, -.46, -.03,  .5, .73, -.52,  .9,  .4, -.78, -.56]
), dtype = "float32")

# scatter plot with error bars
dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")
# scatter plot with error ellipsis
plt.scatter(dfhogg["x"].values, dfhogg["y"].values)
ax = plt.gca()
for i in range(len(dfhogg)):
	x = dfhogg.iloc[i,0]
	y = dfhogg.iloc[i,1]
	sigma_x = dfhogg.iloc[i,2]
	sigma_y = dfhogg.iloc[i,3]
	rho_xy = dfhogg.iloc[i,4]
	cov = rho_xy * sigma_x * sigma_y
	cov_mat = np.array([[sigma_x**2, cov], [cov, sigma_y**2]])
	evals, evecs = scipy.linalg.eigh(cov_mat, eigvals = (0,0)) # eigenvalue, eigenvector
	angle = np.degrees(np.arctan2(evecs[1][0], evecs[0][0])) # article by Vincent Spruyt
	ellipse = Ellipse((x,y), width=2*sigma_x, height=2*sigma_y, angle=angle, edgecolor="b", facecolor="none")
	ax.add_patch(ellipse)

X, Y, sigmaY = dfhogg["x"].values, dfhogg["y"].values, dfhogg["sigma_y"].values
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

#%% models to deal with outliers

# Simple Linear Model with no Outlier Correction
modelfile_ols = "mdl_ols.stan"
with open(modelfile_ols, "w") as file: file.write(model_data_stan + """
	parameters { // discrete parameters impossible
		real b0;
		real b1;
	}
""" + model_trans_params_stan + """
	model {
		Y ~ normal(Yhat, sigmaY);
	}
""")

sm_ols = CmdStanModel(stan_file = modelfile_ols)
fit_ols = sm_ols.sample(
	data = model_data_dict, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_ols.draws().shape # iterations, chains, parameters
fit_ols.summary().loc[var_name] # pandas DataFrame
print(fit_ols.diagnose())

posterior_ols = {k: fit_ols.stan_variable(k) for k in var_name}

az_trace_ols = az.from_cmdstanpy(fit_ols)
az.summary(az_trace_ols).loc[var_name] # pandas DataFrame
az.plot_trace(az_trace_ols, var_names = var_name)

gd = sns.jointplot(
	x = posterior_ols["b0"], y = posterior_ols["b1"],
	marginal_kws = {"kde": True, "kde_kws": {"cut": 1}},
)
gd.plot_joint(sns.kdeplot, zorder = 2, n_levels = 10, cmap = "gray_r")
gd.figure.suptitle("Posterior joint distribution (OLS)", y = 1.02)

# Simple Linear Model with Robust Student-T Likelihood: outliers to have a smaller influence in the likelihood estimation
modelfile_studentt = "mdl_studentt.stan"
with open(modelfile_studentt, "w") as file: file.write(model_data_stan + """
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
var_name_studentt = var_name + ["df"]

sm_studentt = CmdStanModel(stan_file = modelfile_studentt)
fit_studentt = sm_studentt.sample(
	data = model_data_dict, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_studentt.draws().shape # iterations, chains, parameters
fit_studentt.summary().loc[var_name_studentt] # pandas DataFrame
print(fit_studentt.diagnose())

posterior_studentt = {k: fit_studentt.stan_variable(k) for k in var_name_studentt}

az_trace_studentt = az.from_cmdstanpy(fit_studentt)
az.summary(az_trace_studentt).loc[var_name_studentt] # pandas DataFrame
az.plot_trace(az_trace_studentt, var_names = var_name_studentt)

# Linear Model with Custom Likelihood to Distinguish Outliers: Hogg Method
# idea: mixture model whereby datapoints can be: normal linear model vs outlier (for convenience also be linear)
modelfile_hogg = "mdl_hogg.stan"
with open(modelfile_hogg, "w") as file: file.write(model_data_stan + """
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
var_name_hogg_array = var_name + ["Y_outlier", "sigmaY_outlier", "cluster_prob[1]", "cluster_prob[2]"]
var_name_hogg_combi = var_name + ["Y_outlier", "sigmaY_outlier", "cluster_prob"]

sm_hogg = CmdStanModel(stan_file = modelfile_hogg)
fit_hogg = sm_hogg.sample(
	data = model_data_dict, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_hogg.draws().shape # iterations, chains, parameters
fit_hogg.summary().loc[var_name_hogg_array] # pandas DataFrame
print(fit_hogg.diagnose())

posterior_hogg = {k: fit_hogg.stan_variable(k) for k in var_name_hogg_combi}

az_trace_hogg = az.from_cmdstanpy(fit_hogg)
az.summary(az_trace_hogg) # pandas DataFrame
az.plot_trace(az_trace_hogg, var_names = var_name_hogg_combi)

#%% some plots

Xrange = np.array([X.min() - 1, X.max()])
dfhogg.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b", figsize = (8, 6))
for x, y, z in zip(
	[posterior_ols, posterior_studentt, posterior_hogg],
	["r", "g", "m"],
	["OLS", "Student-T", "Hogg method"]
):
	b0, b1 = x["b0"].mean(), x["b1"].mean()
	plt.plot(Xrange, b0 + b1 * Xrange, c = y, label = f"{z}: $ y = {b0:.1f} + {b1:.1f}x $")
plt.legend(loc = "lower right")

for var in var_name:
	a, b, c = posterior_ols[var], posterior_studentt[var], posterior_hogg[var]
	data = pd.DataFrame(data = {
		"y": np.concatenate((a, b, c)),
		"model": np.repeat([
			f"OLS: {a.mean():.1f}", f"Student-T: {b.mean():.1f}", f"Hogg method: {c.mean():.1f}"
		], 40000) # nb of data pts: 4chains × 50000 iter ÷ 5 thin
	})
	g = sns.displot(data, x = "y", hue = "model", bins = "sqrt", kde = True, palette = ["r", "g", "m"])
	g.figure.suptitle(f"Posterior distribution: {var}")
	g.figure.set_figwidth(8)
	g.figure.set_figheight(6)

#%% declare outliers

# Compute the un-normalized log probabilities for each cluster
cluster_0_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(Y, axis = 1),
	loc = posterior_hogg["b0"] + posterior_hogg["b1"] * np.expand_dims(X, axis = 1),
	scale = np.expand_dims(sigmaY, axis = 1)
) + np.log(posterior_hogg["cluster_prob"][:, 0])
cluster_1_log_prob = scipy.stats.norm.logpdf(
	np.expand_dims(Y, axis = 1),
	loc = posterior_hogg["Y_outlier"],
	scale = np.expand_dims(sigmaY, axis = 1) + posterior_hogg["sigmaY_outlier"]
) + np.log(posterior_hogg["cluster_prob"][:, 1])

# Bayes rule to compute the assignment probability: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_1 = cluster_1_log_prob - np.logaddexp(cluster_0_log_prob, cluster_1_log_prob)

# Average across the MCMC chain
log_p_assign_1bis = scipy.special.logsumexp(log_p_assign_1, axis=-1) - np.log(log_p_assign_1.shape[-1])

p_assign_1 = np.exp(log_p_assign_1bis)
dfhogg["is_outlier"] = [f"{100*i:.2f} %" for i in p_assign_1]
dfhogg["classed_as_outlier"] = p_assign_1 >= .95

#%% remove outliers + full model with all variances

# this section not exist in PyMC docs, refer to original paper

no_outliers = dfhogg[~dfhogg.classed_as_outlier]

mdl_full_data = dict(
	N = len(no_outliers),
	X = no_outliers["x"].values,
	Y = no_outliers["y"].values,
	sigmaX = no_outliers["sigma_x"].values,
	sigmaY = no_outliers["sigma_y"].values,
	rhoXY = no_outliers["rho_xy"].values
)

# linear model with gaussian uncertainties
modelfile_full = "mdl_full.stan"
with open(modelfile_full, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector[N] X;
		vector[N] Y;
		vector<lower=0>[N] sigmaX;
		vector<lower=0>[N] sigmaY;
		vector<lower=-1,upper=1>[N] rhoXY;
	}

	transformed data {
		array[N] vector[2] Z; // data pt in vector form
		array[N] matrix[2,2] S; // each data point’s covariance matrix
		for (i in 1:N) {
			Z[i] = [X[i], Y[i]]';
			real covXY = rhoXY[i] * sigmaX[i] * sigmaY[i];
			S[i] = [[sigmaX[i]^2, covXY], [covXY, sigmaY[i]^2]];
		}
	}

	parameters { // discrete parameters impossible
		real m; // slope
		real b; // intercept
	}

	model {
		for (i in 1:N) {
			real Y_hat_i = m * X[i] + b;
			vector[2] Z_hat_i = [X[i], Y_hat_i]';
			Z[i] ~ multi_normal(Z_hat_i, S[i]);
		}
	}
""")
var_name_full = ["m", "b"]

sm_full = CmdStanModel(stan_file = modelfile_full)
fit_full = sm_full.sample(
	data = mdl_full_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_full.draws().shape # iterations, chains, parameters
fit_full.summary().loc[var_name_full] # pandas DataFrame
print(fit_full.diagnose())

posterior_full = {k: fit_full.stan_variable(k) for k in var_name_full}

az_trace_full = az.from_cmdstanpy(fit_full)
az.summary(az_trace_full) # pandas DataFrame
az.plot_trace(az_trace_full, var_names = var_name_full)

b0, b1 = posterior_full["b"].mean(), posterior_full["m"].mean() # name changed
no_outliers.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")
plt.plot(Xrange, b0 + b1 * Xrange, label = f"$ y = {b0:.1f} + {b1:.1f}x $")
plt.legend(loc = "lower right")

#%% full model with intrinsic scatter

# linear model with arbitrary two-dimensional uncertainties
modelfile_full_intrinsic = "mdl_full_intrinsic.stan"
with open(modelfile_full_intrinsic, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector[N] X;
		vector[N] Y;
		vector<lower=0>[N] sigmaX;
		vector<lower=0>[N] sigmaY;
		vector<lower=-1,upper=1>[N] rhoXY;
	}

	transformed data {
		real angle90 = pi()/2; // a cste
		array[N] vector[2] Z; // data pt in vector form
		array[N] matrix[2,2] S; // each data point’s covariance matrix
		for (i in 1:N) {
			Z[i] = [X[i], Y[i]]';
			real covXY = rhoXY[i]*sigmaX[i]*sigmaY[i];
			S[i] = [[sigmaX[i]^2, covXY], [covXY, sigmaY[i]^2]];
		}
	}

	parameters { // discrete parameters impossible
		real<lower=-angle90,upper=angle90> theta; // angle of the fitted line
		real b; // intercept
		real<lower=0> V; // intrinsic Gaussian variance orthogonal to the line
	}

	transformed parameters {
		vector[2] v = [-sin(theta), cos(theta)]'; // unit vector orthogonal to the line
		vector[N] lp; // log prob
		for (i in 1:N) {
			real delta = dot_product(v, Z[i]) - b*v[2]; // orthogonal displacement of each data point from the line
			real sigma2 = quad_form(S[i], v); // orthogonal variance of projection of each data point to the line
			real tmp = sigma2 + V; // intermediary result
			lp[i] = .5*(log(tmp) + delta^2/tmp); // sum(lp) is faster than +=
		}
	}

	model {
		theta ~ uniform(-angle90, angle90);
		target += -sum(lp); // ATTENTION sign
	}

	generated quantities {
		real m = tan(theta); // slope
		real move_up = sqrt(V) / v[2];
	}
""")
var_name_full_intrinsic = var_name_full + ["theta", "V", "move_up"]

sm_full_intrinsic = CmdStanModel(stan_file = modelfile_full_intrinsic)
fit_full_intrinsic = sm_full_intrinsic.sample(
	data = mdl_full_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_full_intrinsic.draws().shape # iterations, chains, parameters
fit_full_intrinsic.summary().loc[var_name_full_intrinsic] # pandas DataFrame
print(fit_full_intrinsic.diagnose())

posterior_full_intrinsic = {k: fit_full_intrinsic.stan_variable(k) for k in var_name_full_intrinsic}

az_trace_full_intrinsic = az.from_cmdstanpy(fit_full_intrinsic)
az.summary(az_trace_full_intrinsic) # pandas DataFrame
az.plot_trace(az_trace_full_intrinsic, var_names = var_name_full_intrinsic)

b0, b1 = posterior_full_intrinsic["b"].mean(), posterior_full_intrinsic["m"].mean()
move_up = posterior_full_intrinsic["move_up"].mean()
no_outliers.plot.scatter(x = "x", y = "y", xerr = "sigma_x", yerr = "sigma_y", c = "b")
plt.plot(Xrange, b0 + b1 * Xrange, label = f"$ y = {b0:.1f} + {b1:.1f}x $")
plt.plot(Xrange, b0 + b1 * Xrange + move_up, linestyle = "--")
plt.plot(Xrange, b0 + b1 * Xrange - move_up, linestyle = "--")
plt.legend(loc = "lower right")
