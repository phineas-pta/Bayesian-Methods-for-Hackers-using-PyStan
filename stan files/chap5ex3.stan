/*
cd /opt/cmdstan && make -j 4 ~/code_playground/chap5ex3 && cd ~/code_playground

./chap5ex3 optimize data file=chap5ex3.data.json

./chap5ex3 sample
	num_chains=4 num_samples=50000 num_warmup=10000 thin=5
	init=chap5ex3.init.json
	data file=chap5ex3.data.json
	output file=chap5ex3_fit.csv diagnostic_file=chap5ex3_dia.csv refresh=0
	num_threads=2

/opt/cmdstan/bin/stansummary chap5ex3_fit_*.csv

./chap5ex3 variational data file=chap5ex3.data.json
*/

functions {
	real f_dist(row_vector glxy_pos, row_vector halo_pos, real cste) {
		// max of either the Euclidian distance or the constant
		return fmax(distance(glxy_pos, halo_pos), cste); // ATTENTION: `max` for vector/matrix/array
	}
	row_vector tangential_distance(row_vector glxy_pos, row_vector halo_pos) {
		row_vector[2] delta = glxy_pos - halo_pos;
		real angle = 2 * atan2(delta[2], delta[1]); // angle of the galaxy with respect to the dark matter centre
		return [-cos(angle), -sin(angle)];
	}
	real partial_sum( // model very slow => parallelization (reduce with summation)
		array[] row_vector ellipticity_slice, int start, int end, int N_halos,
		array[] row_vector galaxies, array[] row_vector halo_pos, row_vector mass_halos, row_vector cste_f_dist
	) {
		real res = 0;
		for (i in start:end) {
			row_vector[2] ellpty_mvn_loc = [0, 0];
			for (j in 1:N_halos)
				ellpty_mvn_loc += tangential_distance(galaxies[i], halo_pos[j]) * mass_halos[j] / f_dist(galaxies[i], halo_pos[j], cste_f_dist[j]);
			res += normal_lpdf(ellipticity_slice[i-start+1] | ellpty_mvn_loc, 0.223607);
		}
		return res;
	}
}

data { // avoid putting data in matrix except for linear algebra
	int<lower=0> N;
	array[N] row_vector<lower=0, upper=4200>[2] galaxies; // galaxy position (x, y)
	array[N] row_vector<lower=-1, upper=1>[2] ellipticity; // 2 ellpty
}

transformed data {
	int N_halos = 3;
	row_vector[N_halos] cste_f_dist = [240, 70, 70]; // one large & 2 smaller
}

parameters { // discrete parameters impossible
	real<lower=0> mass_large; // large halo mass, log uniform does not work ?
	array[N_halos] row_vector<lower=0, upper=4200>[2] halo_pos; // halo position (x, y)
}

transformed parameters {
	row_vector[N_halos] mass_halos = [mass_large, 20, 20]; // one large & 2 smaller
}

model {
	mass_large ~ uniform(40, 180);
	for (j in 1:N_halos) halo_pos[j] ~ uniform(0, 4200); // use `j` to have same annotation below
	target += reduce_sum(
		partial_sum, ellipticity, 1, N_halos, // grainsize = 1 => estimated automatically
		galaxies, halo_pos, mass_halos, cste_f_dist
	);
}
