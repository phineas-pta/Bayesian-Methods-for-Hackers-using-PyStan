/*
Hogg 2010 data without 3 outliers: linear model with gaussian uncertainties
*/

data {
	int<lower=0> N;
	vector[N] X;
	vector[N] Y;
	vector<lower=0>[N] sigmaX;
	vector<lower=0>[N] sigmaY;
	vector<lower=-1,upper=1>[N] rhoXY;
}

transformed data {
	array[N] vector[2] Z; // data pt in vector form
	array[N] matrix[2,2] S; // each data point’s covariance matrix
	for (i in 1:N) {
		Z[i] = [X[i], Y[i]]';
		real covXY = rhoXY[i] * sigmaX[i] * sigmaY[i];
		S[i] = [[sigmaX[i]^2, covXY], [covXY, sigmaY[i]^2]];
	}
}

parameters {
	real m; // slope
	real b; // intercept
}

transformed parameters {
	vector[N] Y_hat = m * X + b;
	array[N] vector[2] Z_hat;
	for (i in 1:N) Z_hat[i] = [X[i], Y_hat[i]]';
}

model {
	for (i in 1:N) Z[i] ~ multi_normal(Z_hat[i], S[i]);
}

generated quantities {
	vector[N] log_lik; // pointwise log-likelihood
	for (i in 1:N) log_lik[i] = multi_normal_lpdf(Z[i] | Z_hat[i], S[i]);
}
