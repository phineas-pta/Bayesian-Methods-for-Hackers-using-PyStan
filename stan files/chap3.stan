/*
data generation algorithm:
For each data point, choose cluster 1 with probability p, else choose cluster 2.
Draw a random variate from a Normal distribution with parameters μ_i and σ_i where i was cluster chosen previously in step 1.
*/

data {
	int<lower=0> N;
	vector[N] obs;
}

transformed data {
	int<lower=0> n_class = 2;
}

parameters {
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
