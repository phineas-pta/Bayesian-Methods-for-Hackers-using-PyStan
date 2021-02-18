# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

halo_data = pd.read_csv("DarkWorlds/Training_halos.csv")
SkyID = 215 # take an example from train data
data_sky = pd.read_csv(f"DarkWorlds/Train_Skies/Training_Sky{SkyID}.csv")
data_sky[["e1", "e2"]].std()

# plot sky + halo
def draw_sky(galaxies: pd.DataFrame, SkyID: int):
	fig = plt.figure(figsize = (9, 9))
	ax = fig.add_subplot(111, aspect = 'equal')

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
sm = pystan.StanModel(model_name = "Tim_Salimans", model_code = """
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
	}

	data { // avoid putting data in matrix except for linear algebra
		int<lower=0> N;
		row_vector<lower=0, upper=4200>[2] galaxies[N]; // galaxy position (x, y)
		row_vector<lower=-1, upper=1>[2] ellipticity[N]; // 2 ellpty
	}

	transformed data {
		int N_halos = 3;
		row_vector[N_halos] cste_f_dist = [240, 70, 70]; // one large & 2 smaller
	}

	parameters { // discrete parameters impossible
		real<lower=0> mass_large; // large halo mass, log uniform does not work ?
		row_vector<lower=0, upper=4200>[2] halo_pos[N_halos]; // halo position (x, y)
	}

	transformed parameters {
		row_vector[N_halos] mass_halos = [mass_large, 20, 20]; // one large & 2 smaller
	}

	model {
		mass_large ~ uniform(40, 180);
		for (j in 1:N_halos) halo_pos[j] ~ uniform(0, 4200); // use `j` to have same annotation below
		for (i in 1:N) {
			row_vector[2] ellpty_mvn_loc = [0, 0];
			for (j in 1:N_halos)
				ellpty_mvn_loc += tangential_distance(galaxies[i], halo_pos[j]) * mass_halos[j] / f_dist(galaxies[i], halo_pos[j], cste_f_dist[j]);
			ellipticity[i] ~ normal(ellpty_mvn_loc, 0.223607);
		}
	}
""")
nchain = 3
fit = sm.sampling( # very very slow
	data = mdl_data, pars = ["halo_pos"], # control = {"adapt_delta": .95},
	iter = 50000, chains = nchain, warmup = 10000, thin = 5, n_jobs = -1, # parallel
	init = [{"mass_large": 80, "halo_pos": [[1000, 500], [2100, 1500], [3500, 4000]]}] * nchain # must have init, otherwise failed
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

az_trace = az.from_pystan(posterior = fit)
az.summary(az_trace)
az.plot_trace(az_trace)

draw_sky(data_sky, SkyID)
for i, val in enumerate(["#F15854", "#B276B2", "#FAA43A"]):
	plt.scatter(posterior["halo_pos"][:,i,0], posterior["halo_pos"][:,i,1], alpha = 0.015, c = val)
