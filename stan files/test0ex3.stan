/*
cd /opt/cmdstan && make -j 4 ~/code_playground/test0ex3 && cd ~/code_playground

./test0ex3 optimize data file=test0ex3.data.json

./test0ex3 sample
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5
	data file=test0ex3.data.json
	output file=test0ex3_fit.csv diagnostic_file=test0ex3_dia.csv refresh=0

/opt/cmdstan/bin/stansummary test0ex3_fit_*.csv

./test0ex3 variational data file=test0ex3.data.json

for i in {1..4}
	do
		./test0ex3 generate_quantities
			fitted_params=test0ex3_fit_${i}.csv
			data file=test0ex3.data.json
			output file=test0ex3_ppc_${i}.csv &
	done
*/

data {
	int<lower=0> J;
	int<lower=0> N;
	int<lower=0, upper=J> stl;
	real u_stl;
	real xbar_stl;
	array[N] int<lower=1, upper=J> county;
	vector[N] u;
	vector[N] x;
	vector[N] x_mean;
	vector[N] y;
}

parameters {
	vector[J] a;
	vector[3] b;
	real mu_a;
	real<lower=0, upper=100> sigma_a;
	real<lower=0, upper=100> sigma_y;
}

transformed parameters {
	vector[N] y_hat;
	for (i in 1:N) y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2] + x_mean[i] * b[3];
	real stl_mu = a[stl+1] + u_stl * b[1] + b[2] + xbar_stl * b[3];
}

model {
	mu_a ~ normal(0, 1);
	a ~ normal(mu_a, sigma_a);
	b ~ normal(0, 1);
	y ~ normal(y_hat, sigma_y);
}

generated quantities {
	real y_stl = normal_rng(stl_mu, sigma_y);
}
