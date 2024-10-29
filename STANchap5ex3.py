# -*- coding: utf-8 -*-

# src:
# - http://www.kaggle.com/c/DarkWorlds
# - http://www.timsalimans.com/observing-dark-worlds

import numpy as np, pandas as pd, arviz as az, matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from matplotlib.patches import Ellipse

halo_data = pd.read_csv("data/DarkWorlds/Training_halos.csv")
SkyID = 215 # take an example from train data
data_sky = pd.read_csv(f"data/DarkWorlds/Train_Skies/Training_Sky{SkyID}.csv")
data_sky[["e1", "e2"]].std()

# plot sky + halo
def draw_sky(galaxies: pd.DataFrame, SkyID: int):
	fig = plt.figure(figsize = (9, 9))
	ax = fig.add_subplot(111, aspect = "equal")

	for row in galaxies.itertuples(index = False):
		d = np.sqrt(row.e1 ** 2 + row.e2 ** 2)
		a, b = 1 / (1 - d), 1 / (1 + d)
		theta = np.degrees(.5*np.arctan2(row.e2, row.e1))
		ax.add_patch(Ellipse(xy = (row.x, row.y), width = 40 * a, height = 40 * b, angle = theta))

	halos = halo_data.iloc[SkyID - 1]
	for i in range(halos.numberHalos):
		print(halos[f"halo_x{i+1}"], halos[f"halo_y{i+1}"])
		ax.scatter(halos[f"halo_x{i+1}"], halos[f"halo_y{i+1}"], c = "k", s = 70)

	ax.autoscale_view(tight = True)
	return fig
draw_sky(data_sky, SkyID)

# model
halo_data["numberHalos"].max()
mdl_data = {
	"N": len(data_sky),
	"galaxies": data_sky[["x", "y"]].values,
	"ellipticity": data_sky[["e1", "e2"]].values
}

modelfile = "DarkWorldsTimSalimans.stan"
with open(modelfile, "w") as file: file.write("""
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
		int grainsize = 1; // grainsize should be estimated automatically
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
			partial_sum, ellipticity, grainsize, N_halos,
			galaxies, halo_pos, mass_halos, cste_f_dist
		);
	}
""")
nchain = 4
var_name = ["halo_pos"]

sm = CmdStanModel(stan_file = modelfile, cpp_options = {"STAN_THREADS": True}) # parallelization

# maximum likelihood estimation
optim = sm.optimize(data = mdl_data).optimized_params_pd
optim[optim.columns[~optim.columns.str.startswith("lp")]]

# variational inference
vb = sm.variational(data = mdl_data)
vb.variational_sample.columns = vb.variational_params_dict.keys()
vb_name = vb.variational_params_pd.columns[~vb.variational_params_pd.columns.str.startswith(("lp", "log_"))]
vb.variational_params_pd[vb_name]
vb.variational_sample[vb_name]

# Markov chain Monte Carlo
fit = sm.sample( # very very slow
	data = mdl_data, show_progress = True, chains = nchain, # adapt_delta = .95,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5, threads_per_chain = 2, # parallelization
	inits = {"mass_large": 80, "halo_pos": [[1000, 500], [2100, 1500], [3500, 4000]]}
	# must have init, otherwise failed
)

fit.draws().shape # iterations, chains, parameters
fit.summary().loc[vb_name] # pandas DataFrame
print(fit.diagnose())

posterior = {k: fit.stan_variable(k) for k in var_name}

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[vb_name] # pandas DataFrame
az.plot_trace(az_trace, var_names = var_name)

draw_sky(data_sky, SkyID)
for i, val in enumerate(["#F15854", "#B276B2", "#FAA43A"]):
	plt.scatter(posterior["halo_pos"][:,i,0], posterior["halo_pos"][:,i,1], alpha = 0.015, c = val)
