# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt, seaborn as sns

#%% data

srrs2 = pd.read_csv("data/srrs2.dat")
srrs2['fips'] = srrs2['stfips']*1000 + srrs2['cntyfips']

cty = pd.read_csv("data/cty.dat")
cty['fips'] = cty['stfips']*1000 + cty['ctfips']

srrs_mn = srrs2[srrs2.state=='MN'].merge(cty[cty.st=='MN'][['fips', 'Uppm']], on='fips').drop_duplicates(subset='idnum')
srrs_mn['county'] = srrs_mn['county'].map(str.strip) # remove blank spaces

mn_counties = srrs_mn['county'].unique()
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
radon = srrs_mn['activity'].values

county = srrs_mn['county'].replace(county_lookup).values
N = len(srrs_mn)
log_radon = np.log(radon + .1)
floor_measure = srrs_mn['floor'].values.astype('float')
counties = len(mn_counties)
u = np.log(srrs_mn['Uppm'].values)
xbar = srrs_mn.groupby('county')['floor'].mean().rename(county_lookup).values
x_mean = xbar[county]
county += 1 # Stan is 1-based index

# +0.1 to make log scale
sns.displot(log_radon, bins = "sqrt", kde = True)
sns.displot(radon + .1, bins = "sqrt", kde = True, log_scale = True)

#%% models

# pooled model
sm_pooled = pystan.StanModel(model_name = "pooled_mdl", model_code = """
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
fit_pooled = sm_pooled.sampling(
	data = {'N': N, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_pooled.stansummary())

# unpooled model
sm_unpooled = pystan.StanModel(model_name = "unpooled_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> J;
		int<lower=1, upper=J> county[N];
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
		for (i in 1:N)
			y_hat[i] = beta * x[i] + a[county[i]];
	}

	model {
		y ~ normal(y_hat, sigma);
	}
""")
fit_unpooled = sm_unpooled.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_unpooled.stansummary())

# partial pooling model
sm_partial_pooling = pystan.StanModel(model_name = "partial_pooling_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> J;
		int<lower=1, upper=J> county[N];
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
		for (i in 1:N)
			y_hat[i] = a[county[i]];
	}

	model {
		mu_a ~ normal(0, 1);
		a ~ normal (10 * mu_a, sigma_a);
		y ~ normal(y_hat, sigma_y);
	}
""")
fit_partial_pooling = sm_partial_pooling.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_partial_pooling.stansummary())

# varying intercept model
sm_varying_intercept = pystan.StanModel(model_name = "varying_intercept_mdl", model_code = """
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=1, upper=J> county[N];
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
		for (i in 1:N)
			y_hat[i] = a[county[i]] + x[i] * b;
	}

	model {
		sigma_a ~ uniform(0, 100);
		a ~ normal (mu_a, sigma_a);
		b ~ normal (0, 1);
		sigma_y ~ uniform(0, 100);
		y ~ normal(y_hat, sigma_y);
	}
""")
fit_varying_intercept = sm_varying_intercept.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_varying_intercept.stansummary())

# varying slope model
sm_varying_slope = pystan.StanModel(model_name = "varying_slope_mdl", model_code = """
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=1, upper=J> county[N];
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
		for (i in 1:N)
			y_hat[i] = a + x[i] * b[county[i]];
	}

	model {
		sigma_b ~ uniform(0, 100);
		b ~ normal (mu_b, sigma_b);
		a ~ normal (0, 1);
		sigma_y ~ uniform(0, 100);
		y ~ normal(y_hat, sigma_y);
	}
""")
fit_varying_slope = sm_varying_slope.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_varying_slope.stansummary())

# varying intercept and slope model
sm_varying_intercept_slope = pystan.StanModel(model_name = "varying_intercept_slope_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> J;
		vector[N] y;
		vector[N] x;
		int<lower=1, upper=J> county[N];
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

	model {
		mu_a ~ normal(0, 100);
		mu_b ~ normal(0, 100);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(mu_b, sigma_b);
		y ~ normal(a[county] + b[county] .* x, sigma);
	}
""")
fit_varying_intercept_slope = sm_varying_intercept_slope.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_varying_intercept_slope.stansummary())

# hierarchical intercept model
sm_hierarchical_intercept = pystan.StanModel(model_name = "hierarchical_intercept_mdl", model_code = """
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=1, upper=J> county[N];
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
		vector[N] m;
		for (i in 1:N) {
			m[i] = a[county[i]] + u[i] * b[1];
			y_hat[i] = m[i] + x[i] * b[2];
		}
	}

	model {
		mu_a ~ normal(0, 1);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(0, 1);
		y ~ normal(y_hat, sigma_y);
	}
""")
fit_hierarchical_intercept = sm_hierarchical_intercept.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'u': u, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_hierarchical_intercept.stansummary())

# contextual effect
sm_contextual_effect = pystan.StanModel(model_name = "contextual_effect_mdl", model_code = """
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=1, upper=J> county[N];
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
		for (i in 1:N)
			y_hat[i] = a[county[i]] + u[i]*b[1] + x[i]*b[2] + x_mean[i]*b[3];
	}
	model {
		mu_a ~ normal(0, 1);
		a ~ normal(mu_a, sigma_a);
		b ~ normal(0, 1);
		y ~ normal(y_hat, sigma_y);
	}
""")
fit_contextual_effect = sm_contextual_effect.sampling(
	data = {'N': N, 'J': counties, 'county': county, 'u': u, 'x_mean': x_mean, 'x': floor_measure, 'y': log_radon},
	iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1
)
print(fit_contextual_effect.stansummary())

# prediction
stl = 'ST LOUIS'
i_stl = county_lookup[stl]
sm_contextual_pred = pystan.StanModel(model_name = "contextual_pred_mdl", model_code = """
	data {
		int<lower=0> J;
		int<lower=0> N;
		int<lower=0, upper=J> stl;
		real u_stl;
		real xbar_stl;
		int<lower=1, upper=J> county[N];
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
		real stl_mu;
		for (i in 1:N)
			y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2] + x_mean[i] * b[3];
		stl_mu = a[stl+1] + u_stl * b[1] + b[2] + xbar_stl * b[3];
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
fit_contextual_pred = sm_contextual_pred.sampling(
	data = {
		'N': N, 'J': counties, 'county': county, 'u': u, 'x_mean': x_mean, 'x': floor_measure, 'y': log_radon,
		'stl': i_stl, 'u_stl': np.unique(u[srrs_mn.county == stl])[0], 'xbar_stl': xbar[i_stl]
	}, iter = 50000, chains = 3, warmup = 10000, thin = 5, n_jobs = -1, pars = ["y_stl"]
)
print(fit_contextual_pred.stansummary())

sample_contextual_pred = fit_contextual_pred.extract(permuted = True) # all chains are merged and warmup samples are discarded
sns.displot(sample_contextual_pred['y_stl'], bins = "sqrt", kde = True)
