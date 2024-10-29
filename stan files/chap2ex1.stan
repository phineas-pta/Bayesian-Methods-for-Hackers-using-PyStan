/*
You are given a series of daily text-message counts from a user of your system. You are curious to know if the user’s text-messaging habits have changed over time, either gradually or suddenly. How can you model this?

text-message count of day i: C_i ~ Poisson(λ)
λ = λ1 if day < τ, λ2 if day ≥ τ, with τ = switchpoint (user’s text-messaging habits change), if λ1 = λ2 then no change
λ1 ~ Γ(α, β) and λ2 ~ Γ(α, β)
τ ~ Unif(1, N)

this code is an optimized version to avoid `if` condition in model, see stan doc for details: https://mc-stan.org/docs/stan-users-guide/latent-discrete.html
*/

data {
	int<lower=0> N;
	array[N] int<lower=0> obs;
}

transformed data {
	real alpha = 8.;
	real bbeta = .3; // `beta` is built-in distrib fx
	real log_unif = -log(N);
}

parameters { // no info about tau
	real<lower=0> lambda1;
	real<lower=0> lambda2;
}

transformed parameters { // marginalize out the discrete parameter
	vector[N] lp = rep_vector(log_unif, N);
	vector[N+1] lp1;
	vector[N+1] lp2;
	lp1[1] = 0;
	lp2[1] = 0;
	for (i in 1:N) { // dynamic programming workaround to avoid nested loop
		lp1[i+1] = lp1[i] + poisson_lpmf(obs[i] | lambda1);
		lp2[i+1] = lp2[i] + poisson_lpmf(obs[i] | lambda2);
	}
	lp = rep_vector(log_unif + lp2[N+1], N) + head(lp1, N) - head(lp2, N);
}

model {
	lambda1 ~ gamma(alpha, bbeta);
	lambda2 ~ gamma(alpha, bbeta);
	target += log_sum_exp(lp);
}

generated quantities { // generate tau here
	int<lower=1,upper=N> tau = categorical_logit_rng(lp);
}
