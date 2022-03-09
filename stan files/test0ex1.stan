/*
cd /opt/cmdstan && make -j 4 ~/code_playground/test0ex1 && cd ~/code_playground

./test0ex1 optimize data file=test0ex1.data.json

./test0ex1 sample
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5
	data file=test0ex1.data.json
	output file=test0ex1_fit.csv diagnostic_file=test0ex1_dia.csv refresh=0

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
	real angle90 = pi()/2; // a cste
	array[N] vector[2] Z; // data pt in vector form
	array[N] matrix[2,2] S; // each data point’s covariance matrix
	for (i in 1:N) {
		Z[i] = [X[i], Y[i]]';
		real covXY = rhoXY[i]*sigmaX[i]*sigmaY[i];
		S[i] = [[sigmaX[i]^2, covXY], [covXY, sigmaY[i]^2]];
	}
}

parameters { // discrete parameters impossible
	real<lower=-angle90,upper=angle90> theta; // angle of the fitted line
	real b; // intercept
}

transformed parameters {
	vector[2] v = [-sin(theta), cos(theta)]'; // unit vector orthogonal to the line
	vector[N] lp; // log prob
	for (i in 1:N) {
		real delta = dot_product(v, Z[i]) - b*v[2]; // orthogonal displacement of each data point from the line
		real sigma2 = quad_form(S[i], v); // orthogonal variance of projection of each data point to the line
		lp[i] = delta^2/2/sigma2; // sum(lp) is faster than +=
	}
}

model {
	theta ~ uniform(-angle90, angle90);
	target += -sum(lp); // ATTENTION sign
}

generated quantities {
	real m = tan(theta); // slope
}
