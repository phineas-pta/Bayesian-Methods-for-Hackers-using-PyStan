/*
Hogg 2010 data: simple ordinary least square model with no outlier correction
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
}

transformed parameters {
	vector[N] Yhat = b0 + b1 * X;
}

model {
	Y ~ normal(Yhat, sigmaY);
}
