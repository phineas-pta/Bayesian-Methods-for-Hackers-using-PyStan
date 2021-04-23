# -*- coding: utf-8 -*-

import numpy as np, arviz as az
from cmdstanpy import CmdStanModel

# data for model fitting
N = 5
time_since_joined = np.array([4.5, 6., 7., 12., 18.])
slack_comments = np.array([7500, 10100, 18600, 25200, 27500])
github_commits = np.array([25, 32, 49, 66, 96])
names = ["Alice", "Bob", "Cole", "Danielle", "Erika"]

# data for out of sample predictions
N_pred = 2
candidate_devs_time = np.array([3.6, 5.1])
candidate_devs = ["Francis", "Gerard"]

# prior distrib
modelfile_prior = "prior.stan"
with open(modelfile_prior, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector<lower=0>[N] time_since_joined;
	}

	generated quantities {
		real b0 = normal_rng(0, 200);
		real b1 = normal_rng(0, 200);
		real<lower=0> b_sigma = abs(normal_rng(0, 300));
		real log_b_sigma = log(b_sigma);

		real c0 = normal_rng(0, 10);
		real c1 = normal_rng(0, 10);
		real<lower=0> c_sigma = fabs(normal_rng(0, 6));
		real log_c_sigma = log(b_sigma);

		// must be array
		real slack_comments_hat[N] = normal_rng(b0 + b1 * time_since_joined, b_sigma);
		real github_commits_hat[N] = normal_rng(c0 + c1 * time_since_joined, c_sigma);
	}
""")

sm_prior = CmdStanModel(stan_file = modelfile_prior)
prior = sm_prior.sample(
	data = {"N": N, "time_since_joined": time_since_joined},
	iter_sampling = 150, chains = 1, iter_warmup = 0, fixed_param = True
)

# posterior distrib
modelfile_posterior = "posterior.stan"
with open(modelfile_posterior, "w") as file: file.write("""
	data {
		int<lower=0> N;
		vector<lower=0>[N] time_since_joined;
		vector<lower=0>[N] slack_comments;
		vector<lower=0>[N] github_commits;

		// out of sample prediction
		int<lower=0> N_pred;
		vector<lower=0>[N_pred] time_since_joined_pred;
	}

	parameters {
		real b0;
		real b1;
		real log_b_sigma;

		real c0;
		real c1;
		real log_c_sigma;
	}

	transformed parameters {
		real<lower=0> b_sigma = exp(log_b_sigma);
		real<lower=0> c_sigma = exp(log_c_sigma);
	}

	model {
		b0 ~ normal(0, 200);
		b1 ~ normal(0, 200);
		slack_comments ~ normal(b0 + b1 * time_since_joined, b_sigma);
		github_commits ~ normal(c0 + c1 * time_since_joined, c_sigma);
	}

	generated quantities {
		// elementwise log likelihood: type real???
		real log_likelihood_slack_comments = normal_lpdf(slack_comments | b0 + b1 * time_since_joined, b_sigma);
		real log_likelihood_github_commits = normal_lpdf(github_commits | c0 + c1 * time_since_joined, c_sigma);

		// posterior predictive: must be array
		real slack_comments_hat[N] = normal_rng(b0 + b1 * time_since_joined, b_sigma);
		real github_commits_hat[N] = normal_rng(c0 + c1 * time_since_joined, c_sigma);

		// out of sample prediction: must be array
		real slack_comments_pred[N_pred] = normal_rng(b0 + b1 * time_since_joined_pred, b_sigma);
		real github_commits_pred[N_pred] = normal_rng(c0 + c1 * time_since_joined_pred, c_sigma);
	}
""")

sm_posterior = CmdStanModel(stan_file = modelfile_posterior)
posterior = sm_posterior.sample(data={
	"N": N, "time_since_joined": time_since_joined,
	"slack_comments": slack_comments, "github_commits": github_commits,
	"N_pred" : N_pred, "time_since_joined_pred" : candidate_devs_time
}, iter_sampling = 200, chains = 4)

# save to arviz
var_name = ["slack_comments","github_commits"]
idata_stan = az.from_cmdstanpy( # DOES NOT WORK, problems everywhere, idk why
	posterior = posterior, prior = prior,
	posterior_predictive = [i + "_hat" for i in var_name],
	prior_predictive = [i + "_hat" for i in var_name],
	observed_data = var_name,
	constant_data = ["time_since_joined"],
	log_likelihood = {i: "log_likelihood_" + i for i in var_name},
	predictions = [i + "_pred" for i in var_name],
	predictions_constant_data = ["time_since_joined_pred"],
	coords = {"developer": names, "candidate developer" : candidate_devs},
	dims = {
		"slack_comments": ["developer"],
		"github_commits" : ["developer"],
		"slack_comments_hat": ["developer"],
		"github_commits_hat": ["developer"],
		"time_since_joined": ["developer"],
		"slack_comments_pred" : ["candidate developer"],
		"github_commits_pred" : ["candidate developer"],
		"time_since_joined_pred" : ["candidate developer"],
	}
)
