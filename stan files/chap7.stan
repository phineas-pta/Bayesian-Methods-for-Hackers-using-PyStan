/*
cd /opt/cmdstan && make -j 4 ~/code_playground/chap7 && cd ~/code_playground

./chap7 optimize data file=chap7.data.json

./chap7 sample
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5
	data file=chap7.data.json
	output file=chap7_fit.csv diagnostic_file=chap7_dia.csv refresh=0

/opt/cmdstan/bin/stansummary chap7_fit_*.csv

./chap7 variational data file=chap7.data.json

for i in {1..4}
	do
		./chap7 generate_quantities
			fitted_params=chap7_fit_${i}.csv
			data file=chap7.data.json
			output file=chap7_ppc_${i}.csv &
	done
*/

data {
	int N; // the number of training observations
	int N2; // the number of test observations
	int K; // the number of features
	array[N] int y; // the response
	matrix[N,K] X; // the model matrix
	matrix[N2,K] new_X; // the matrix for the predicted values
}

parameters { // regression parameters
	real alpha;
	vector[K] beta;
}

transformed parameters {
	vector[N] linpred = alpha + X * beta;
}

model {
	alpha ~ cauchy(0, 10); // prior for the intercept following Gelman 2008
	beta ~ student_t(1, 0, 0.03);
	y ~ bernoulli_logit(linpred);
}

generated quantities { // y values predicted by the model
	vector[N2] y_pred = alpha + new_X * beta;
}
