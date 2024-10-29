/*
Hogg 2010 data without 3 outliers: linear model with arbitrary two-dimensional uncertainties
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
	real<lower=-angle90,upper=angle90> theta; // angle of the fitted line, use this instead of slope
	real b; // intercept
	real<lower=0> V; // intrinsic Gaussian variance orthogonal to the line
}

transformed parameters {
	vector[2] v = [-sin(theta), cos(theta)]'; // unit vector orthogonal to the line
	vector[N] lp; // log prob
	for (i in 1:N) {
		real delta = dot_product(v, Z[i]) - b*v[2]; // orthogonal displacement of each data point from the line
		real sigma2 = quad_form(S[i], v); // orthogonal variance of projection of each data point to the line
		real tmp = sigma2 + V; // intermediary result
		lp[i] = .5*(log(tmp) + delta^2/tmp); // sum(lp) is faster than +=
	}
}

model {
	theta ~ uniform(-angle90, angle90);
	target += -sum(lp); // ATTENTION sign
}

generated quantities {
	real m = tan(theta); // slope
	real move_up = sqrt(V) / v[2];
}
