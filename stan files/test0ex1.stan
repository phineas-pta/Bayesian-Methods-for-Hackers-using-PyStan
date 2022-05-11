/*
cd /opt/cmdstan && make -j 6 ~/coder/test0ex1 && cd ~/coder

./test0ex1 optimize data file=test0ex1.data.json

./test0ex1 sample
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5
	data file=test0ex1.data.json
	output file=test0ex1_fit.csv diagnostic_file=test0ex1_dia.csv
	refresh=0 num_threads=4

/opt/cmdstan/bin/stansummary test0ex1_fit_*.csv

./test0ex1 variational data file=test0ex1.data.json
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

parameters { // discrete parameters impossible
	real m; // slope
	real b; // intercept
}

model {
	for (i in 1:N) {
		real Y_hat_i = m * X[i] + b;
		vector[2] Z_hat_i = [X[i], Y_hat_i]';
		Z[i] ~ multi_normal(Z_hat_i, S[i]);
	}
}