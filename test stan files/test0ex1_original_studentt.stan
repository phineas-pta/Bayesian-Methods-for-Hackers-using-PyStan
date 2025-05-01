/*
Hogg 2010 data: Linear Model with Robust Student-T Likelihood: outliers to have a smaller influence in the likelihood estimation
*/

data {
	int<lower=0> N;
	vector[N] X;
	vector[N] Y;
	vector<lower=0>[N] sigmaY;
}

parameters {
	real b0; // intercept
	real b1; // slope
	real<lower=0> df;  // degree of freedom
}

transformed parameters {
	vector[N] Yhat = b0 + b1 * X;
}

model {
	df ~ inv_gamma(1, 1);
	Y ~ student_t(df, Yhat, sigmaY);
}
