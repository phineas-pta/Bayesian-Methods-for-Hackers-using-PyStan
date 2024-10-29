/*
src: https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
*/

data {
	int<lower=0> N;
	array[N] int<lower=0> at_bats;
	array[N] int<lower=0> hits;
}

parameters {
	real log_kappa;
	real<lower=0, upper=1> phi;
	vector<lower=0, upper=1>[N] thetas;
}

transformed parameters {
	real<lower=0> kappa = exp(log_kappa);
	real<lower=0> alpha = kappa * phi;
	real<lower=0> bbeta = kappa * (1 - phi); // `beta` is built-in distrib fx
}

model {
	log_kappa ~ exponential(1.5);
	phi ~ uniform(0, 1);
	thetas ~ bbeta(alpha, bbeta);
	hits ~ binomial(at_bats, thetas);
}
