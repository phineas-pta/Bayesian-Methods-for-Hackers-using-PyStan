/*
src: https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html

study of radon levels in 80,000 houses: 2 important predictors:
- measurement in basement or first floor (radon higher in basements)
- county uranium level (positive correlation with radon levels)
focus on modeling radon levels in Minnesota.
*/

data {
	int<lower=0> J; // county count
	int<lower=0> N;
	array[N] int<lower=1, upper=J> county; // county ID
	vector[N] u; // uranium measurement at the county level
	vector[N] x; // 0 for basement and 1 for 1st floor
	vector[N] x_mean;
	vector[N] y; // logarithm of the radon measurement in house
}

parameters {
	vector[J] a; // average log radon level among all the houses in 1 county
	vector[3] b;
	real mu_a;
	real<lower=0, upper=100> sigma_a; // variance among the average log radon levels of the different counties
	real<lower=0, upper=100> sigma_y; // within-county variance in log radon measurements
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
