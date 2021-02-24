# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt

# data for model fitting
N = 5
time_since_joined = np.array([4.5, 6., 7., 12., 18.])
slack_comments = np.array([7500, 10100, 18600, 25200, 27500])
github_commits = np.array([25, 32, 49, 66, 96])
names = np.array(["Alice", "Bob", "Cole", "Danielle", "Erika"])

# data for out of sample predictions
N_pred = 2
candidate_devs_time = np.array([3.6, 5.1])
candidate_devs = np.array(["Francis", "Gerard"])

# prior distrib
sm_prior = pystan.StanModel(model_code = """
	data {
		int<lower=0> N;
		real time_since_joined[N];
	}

	generated quantities {
		real b0;
		real b1;
		real log_b_sigma;
		real<lower=0> b_sigma;

		real c0;
		real c1;
		real log_c_sigma;
		real<lower=0> c_sigma;

		vector[N] slack_comments_hat;
		vector[N] github_commits_hat;

		b0 = normal_rng(0, 200);
		b1 = normal_rng(0, 200);
		b_sigma = abs(normal_rng(0, 300));
		log_b_sigma = log(b_sigma);

		c0 = normal_rng(0, 10);
		c1 = normal_rng(0, 10);
		c_sigma = fabs(normal_rng(0, 6));
		log_c_sigma = log(b_sigma);

		for (n in 1:N) {
			slack_comments_hat[n] = normal_rng(b0 + b1 * time_since_joined[n], b_sigma);
			github_commits_hat[n] = normal_rng(c0 + c1 * time_since_joined[n], c_sigma);
		}
	}
""")
prior = sm_prior.sampling(
	data={"N": N, "time_since_joined": time_since_joined},
	iter=150, chains=1, algorithm='Fixed_param', warmup=0
)

# posterior distrib
sm = pystan.StanModel(model_code="""
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
		b0 ~ normal(0,200);
		b1 ~ normal(0,200);
		b_sigma ~ normal(0,300);
		slack_comments ~ normal(b0 + b1 * time_since_joined, b_sigma);
		github_commits ~ normal(c0 + c1 * time_since_joined, c_sigma);
	}

	generated quantities {
		// elementwise log likelihood
		vector[N] log_likelihood_slack_comments;
		vector[N] log_likelihood_github_commits;

		// posterior predictive
		vector[N] slack_comments_hat;
		vector[N] github_commits_hat;

		// out of sample prediction
		vector[N_pred] slack_comments_pred;
		vector[N_pred] github_commits_pred;

		// posterior predictive
		for (n in 1:N) {
			log_likelihood_slack_comments[n] = normal_lpdf(slack_comments[n] | b0 + b1 * time_since_joined[n], b_sigma);
			slack_comments_hat[n] = normal_rng(b0 + b1 * time_since_joined[n], b_sigma);

			log_likelihood_github_commits[n] = normal_lpdf(github_commits[n] | c0 + c1 * time_since_joined[n], c_sigma);
			github_commits_hat[n] = normal_rng(c0 + c1 * time_since_joined[n], c_sigma);
		}

		// out of sample prediction
		for (n in 1:N_pred) {
			slack_comments_pred[n] = normal_rng(b0 + b1 * time_since_joined_pred[n], b_sigma);
			github_commits_pred[n] = normal_rng(c0 + c1 * time_since_joined_pred[n], c_sigma);
		}
	}
""")
posterior = sm.sampling(data={
	"N": N, "time_since_joined": time_since_joined,
	"slack_comments": slack_comments, "github_commits": github_commits,
	"N_pred" : N_pred, "time_since_joined_pred" : candidate_devs_time
}, iter=200, chains=4)

# save to arviz
var_name = ["slack_comments","github_commits"]
idata_stan = az.from_pystan(
	posterior=posterior, prior=prior,
	posterior_predictive=[i + "_hat" for i in var_name],
	prior_predictive=[i + "_hat" for i in var_name],
	observed_data=var_name,
	constant_data=["time_since_joined"],
	log_likelihood={i: "log_likelihood_" + i for i in var_name},
	predictions=[i + "_pred" for i in var_name],
	predictions_constant_data=["time_since_joined_pred"],
	coords={"developer": names, "candidate developer" : candidate_devs},
	dims={
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
