# -*- coding: utf-8 -*-

# src: https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html

import numpy as np, pandas as pd, arviz as az, matplotlib.pyplot as plt, seaborn as sns
from cmdstanpy import CmdStanModel

#%% data

srrs2 = pd.read_csv("data/srrs2.dat")
srrs2["fips"] = srrs2["stfips"]*1000 + srrs2["cntyfips"]

cty = pd.read_csv("data/cty.dat")
cty["fips"] = cty["stfips"]*1000 + cty["ctfips"]

srrs_mn = srrs2[srrs2.state=="MN"].merge(cty[cty.st=="MN"][["fips", "Uppm"]], on="fips").drop_duplicates(subset="idnum")
srrs_mn["county"] = srrs_mn["county"].map(str.strip) # remove blank spaces

mn_counties = srrs_mn["county"].unique()
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
radon = srrs_mn["activity"].values

county = srrs_mn["county"].replace(county_lookup).values
N = len(srrs_mn)
log_radon = np.log(radon + .1) # +0.1 to make log scale
floor_measure = srrs_mn["floor"].values.astype("float")
counties = len(mn_counties)
u = np.log(srrs_mn["Uppm"].values)
xbar = srrs_mn.groupby("county")["floor"].mean().rename(county_lookup).values
x_mean = xbar[county]
county += 1 # Stan is 1-based index

# +0.1 to make log scale
sns.displot(log_radon, bins = "sqrt", kde = True)
sns.displot(radon + .1, bins = "sqrt", kde = True, log_scale = True)

#%% models

# pooled model
modelfile_pooled = "mdl_pooled.stan"
with open(modelfile_pooled, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector[N] x;
		vector[N] y;
	}

	parameters { // discrete parameters impossible
		vector[2] beta;
		real<lower=0> sigma;
	}

	transformed parameters {
		vector[N] y_hat = beta[1] + beta[2] * x;
	}

	model {
		y ~ normal(y_hat, sigma);
	}
""")
sm_pooled = CmdStanModel(stan_file = modelfile_pooled)
fit_pooled = sm_pooled.sample(
	data = {"N": N, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_pooled.summary() # pandas DataFrame

# unpooled model
modelfile_unpooled = "mdl_unpooled.stan"
with open(modelfile_unpooled, "w") as file: file.write("""
	data {
		int<lower=0> N;
		int<lower=0> J;
		array[N] int<lower=1, upper=J> county;
		vector[N] x;
		vector[N] y;
	}

	parameters { // discrete parameters impossible
		vector[J] a;
		real beta;
		real<lower=0, upper=100> sigma;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = beta * x[i] + a[county[i]];
	}

	model {
		y ~ normal(y_hat, sigma);
	}
""")
sm_unpooled = CmdStanModel(stan_file = modelfile_unpooled)
fit_unpooled = sm_unpooled.sample(
	data = {"N": N, "J": counties, "county": county, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_unpooled.summary() # pandas DataFrame

# partial pooling model
modelfile_partial_pooling = "mdl_partial_pooling.stan"
with open(modelfile_partial_pooling, "w") as file: file.write("""
	data {
		int<lower=0> N;
		int<lower=0> J;
		array[N] int<lower=1, upper=J> county;
		vector[N] y;
	}

	parameters {
		vector[J] a;
		real mu_a;
		real<lower=0, upper=100> sigma_a;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a[county[i]];
	}

	model {
		mu_a ~ normal(0, 1);
		a ~ normal (10 * mu_a, sigma_a);
		y ~ normal(y_hat, sigma_y);
	}
""")
sm_partial_pooling = CmdStanModel(stan_file = modelfile_partial_pooling)
fit_partial_pooling = sm_partial_pooling.sample(
	data = {"N": N, "J": counties, "county": county, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_partial_pooling.summary() # pandas DataFrame

# varying intercept model
modelfile_varying_intercept = "mdl_varying_intercept.stan"
with open(modelfile_varying_intercept, "w") as file: file.write("""
	data {
		int<lower=0> J;
		int<lower=0> N;
		array[N] int<lower=1, upper=J> county;
		vector[N] x;
		vector[N] y;
	}

	parameters {
		vector[J] a;
		real b;
		real mu_a;
		real<lower=0, upper=100> sigma_a;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a[county[i]] + x[i] * b;
	}

	model {
		sigma_a ~ uniform(0, 100);
		a ~ normal (mu_a, sigma_a);
		b ~ normal (0, 1);
		sigma_y ~ uniform(0, 100);
		y ~ normal(y_hat, sigma_y);
	}
""")
sm_varying_intercept = CmdStanModel(stan_file = modelfile_varying_intercept)
fit_varying_intercept = sm_varying_intercept.sample(
	data = {"N": N, "J": counties, "county": county, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_varying_intercept.summary() # pandas DataFrame

# varying slope model
modelfile_varying_slope = "mdl_varying_slope.stan"
with open(modelfile_varying_slope, "w") as file: file.write("""
	data {
		int<lower=0> J;
		int<lower=0> N;
		array[N] int<lower=1, upper=J> county;
		vector[N] x;
		vector[N] y;
	}

	parameters {
		real a;
		vector[J] b;
		real mu_b;
		real<lower=0, upper=100> sigma_b;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a + x[i] * b[county[i]];
	}

	model {
		sigma_b ~ uniform(0, 100);
		b ~ normal (mu_b, sigma_b);
		a ~ normal (0, 1);
		sigma_y ~ uniform(0, 100);
		y ~ normal(y_hat, sigma_y);
	}
""")
sm_varying_slope = CmdStanModel(stan_file = modelfile_varying_slope)
fit_varying_slope = sm_varying_slope.sampling(
	data = {"N": N, "J": counties, "county": county, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_varying_slope.summary() # pandas DataFrame

# varying intercept and slope model
modelfile_varying_intercept_slope = "mdl_varying_intercept_slope.stan"
with open(modelfile_varying_intercept_slope, "w") as file: file.write("""
	data {
		int<lower=0> N;
		int<lower=0> J;
		vector[N] y;
		vector[N] x;
		array[N] int<lower=1, upper=J> county;
	}

	parameters {
		real<lower=0> sigma;
		real<lower=0> sigma_a;
		real<lower=0> sigma_b;
		vector[J] a;
		vector[J] b;
		real mu_a;
		real mu_b;
	}

	transformed parameters {
		vector[N] y_hat = a[county] + b[county] .* x;
	}

	model {
		mu_a ~ normal(0, 100);
		mu_b ~ normal(0, 100);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(mu_b, sigma_b);
		y ~ normal(y_hat, sigma);
	}
""")
sm_varying_intercept_slope = CmdStanModel(stan_file = modelfile_varying_intercept_slope)
fit_varying_intercept_slope = sm_varying_intercept_slope.sample(
	data = {"N": N, "J": counties, "county": county, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_varying_intercept_slope.summary() # pandas DataFrame

# hierarchical intercept model
modelfile_hierarchical_intercept = "mdl_hierarchical_intercept.stan"
with open(modelfile_hierarchical_intercept, "w") as file: file.write("""
	data {
		int<lower=0> J;
		int<lower=0> N;
		array[N] int<lower=1, upper=J> county;
		vector[N] u;
		vector[N] x;
		vector[N] y;
	}

	parameters {
		vector[J] a;
		vector[2] b;
		real mu_a;
		real<lower=0, upper=100> sigma_a;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2];
	}

	model {
		mu_a ~ normal(0, 1);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(0, 1);
		y ~ normal(y_hat, sigma_y);
	}
""")
sm_hierarchical_intercept = CmdStanModel(stan_file = modelfile_hierarchical_intercept)
fit_hierarchical_intercept = sm_hierarchical_intercept.sample(
	data = {"N": N, "J": counties, "county": county, "u": u, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_hierarchical_intercept.summary() # pandas DataFrame

# contextual effect
modelfile_contextual_effect = "mdl_contextual_effect.stan"
with open(modelfile_contextual_effect, "w") as file: file.write("""
	data {
		int<lower=0> J;
		int<lower=0> N;
		array[N] int<lower=1, upper=J> county;
		vector[N] u;
		vector[N] x;
		vector[N] x_mean;
		vector[N] y;
	}

	parameters {
		vector[J] a;
		vector[3] b;
		real mu_a;
		real<lower=0, upper=100> sigma_a;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2] + x_mean[i] * b[3];
	}
	model {
		mu_a ~ normal(0, 1);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(0, 1);
		y ~ normal(y_hat, sigma_y);
	}
""")
sm_contextual_effect = CmdStanModel(stan_file = modelfile_contextual_effect)
fit_contextual_effect = sm_contextual_effect.sample(
	data = {"N": N, "J": counties, "county": county, "u": u, "x_mean": x_mean, "x": floor_measure, "y": log_radon},
	show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
fit_contextual_effect.summary() # pandas DataFrame

# prediction
stl = "ST LOUIS"
i_stl = county_lookup[stl]
modelfile_contextual_pred = "mdl_contextual_pred.stan"
with open(modelfile_contextual_pred, "w") as file: file.write("""
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=0, upper=J> stl;
		real u_stl;
		real xbar_stl;
		array[N] int<lower=1, upper=J> county;
		vector[N] u;
		vector[N] x;
		vector[N] x_mean;
		vector[N] y;
	}

	parameters {
		vector[J] a;
		vector[3] b;
		real mu_a;
		real<lower=0, upper=100> sigma_a;
		real<lower=0, upper=100> sigma_y;
	}

	transformed parameters {
		vector[N] y_hat;
		for (i in 1:N) y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2] + x_mean[i] * b[3];
		real stl_mu = a[stl+1] + u_stl * b[1] + b[2] + xbar_stl * b[3];
	}

	model {
		mu_a ~ normal(0, 1);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(0, 1);
		y ~ normal(y_hat, sigma_y);
	}

	generated quantities {
		real y_stl = normal_rng(stl_mu, sigma_y);
	}
""")
sm_contextual_pred = CmdStanModel(stan_file = modelfile_contextual_pred)
fit_contextual_pred = sm_contextual_pred.sample(
	data = {
		"N": N, "J": counties, "county": county, "u": u, "x_mean": x_mean, "x": floor_measure, "y": log_radon,
		"stl": i_stl, "u_stl": np.unique(u[srrs_mn.county == stl])[0], "xbar_stl": xbar[i_stl]
	}, show_progress = True, chains = 4, iter_sampling = 50000, iter_warmup = 10000, thin = 5
)
sample_contextual_pred = np.exp(fit_contextual_pred.stan_variable("y_stl"))
sns.displot(sample_contextual_pred, bins = "sqrt", kde = True)
