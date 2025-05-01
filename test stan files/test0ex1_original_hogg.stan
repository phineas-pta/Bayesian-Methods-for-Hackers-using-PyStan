/*
Hogg 2010 data: Hogg Method: Linear Model with Custom Likelihood to Distinguish Outliers
idea: mixture model whereby datapoints can be: normal linear model vs outlier (for convenience also be linear)
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
	real Y_outlier; // mean for all outliers
	real<lower=0> sigmaY_outlier; // additional variance for outliers
	simplex[2] cluster_prob; // mixture ratio
}

transformed parameters {
	vector[N] Yhat = b0 + b1 * X;
}

model {
	b0 ~ normal(0, 5); // weakly informative Normal priors (L2 ridge reg) for inliers
	b1 ~ normal(0, 5); // likewise

	Y_outlier ~ normal(0, 10);
	sigmaY_outlier ~ normal(0, 10); // half-normal because of above constraint

	for (n in 1:N) { // custom mixture model: cluster 1 = inlier, 2 = outlier
		real cluster1 = log(cluster_prob[1]) + normal_lpdf(Y[n] | Yhat[n], sigmaY[n]);
		real cluster2 = log(cluster_prob[2]) + normal_lpdf(Y[n] | Y_outlier, sigmaY[n] + sigmaY_outlier);
		target += log_sum_exp(cluster1, cluster2);
	}
}
