/*
modelling daily stock return

this code is an optimized version with Bartlett decomposition, see stan doc for details: https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html
*/

data { // avoid putting data in matrix except for linear algebra
	int<lower=0> N;
	int<lower=0> N_stocks;
	array[N] row_vector[N_stocks] observations;;
}

transformed data {
	int<lower=2+N_stocks> df = 10;
	row_vector[N_stocks] expert_mus = [-.03, .05, .03, -.02];
	matrix<lower=0>[N_stocks, N_stocks] expert_sigmas = diag_matrix(square([.04, .03, .02, .01]'));
	cholesky_factor_cov[N_stocks] L = cholesky_decompose(expert_sigmas);
}

parameters {
	row_vector[N_stocks] locs;
	vector[N_stocks] c;
	vector[N_stocks * (N_stocks - 1) %/% 2] z;
}

transformed parameters {
	matrix[N_stocks, N_stocks] A;
	{ // extra layer of brackes let us define a local int for the loop
		int count = 1;
		for (j in 1:(N_stocks-1)) {
			for (i in (j+1):N_stocks) {
				A[i,j] = z[count];
				count += 1;
			}
			for (i in 1:(j-1)) A[i,j] = 0;
			A[j, N_stocks] = 0;
			A[j,j] = sqrt(c[j]);
		}
		A[N_stocks, N_stocks] = sqrt(c[N_stocks]);
	}
}

model {
	for (i in 1:N_stocks) c[i] ~ chi_square(df - i + 1);
	z ~ std_normal();
	locs ~ normal(expert_mus, 1);
	observations ~ multi_normal_cholesky(locs, L*A);
}
