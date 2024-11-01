# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, arviz as az, prince, matplotlib.pyplot as plt, seaborn as sns
from cmdstanpy import CmdStanModel

#%% load data

data = pd.read_csv("data/overfitting.csv", index_col = "case_id")
data.columns
data.info()

feature_names = data.columns.str.startswith("var_")
predictors = data[data.columns[feature_names]]
labels = data["Target_Practice"]

ix_training = data.train == 1
training_data = predictors[ix_training]
training_labels = labels[ix_training]

ix_testing = data.train == 0
testing_data = predictors[ix_testing]
testing_labels = labels[ix_testing]

sns.displot(training_data.values.flatten(), bins = "sqrt", kde = True)

pca = prince.PCA(n_components = 2, as_array = False).fit(training_data)
pca.plot_row_coordinates(training_data, color_labels = training_labels)
pca.column_correlations(training_data).plot.scatter(x = 0, y = 1) # weird column name

#%% Roshan Sharma model

mdl_data = { # problem with JSON dump => cast to python native type
	"N": ix_training.sum().tolist(),
	"N2": ix_testing.sum().tolist(),
	"K": feature_names.sum().tolist(),
	"y": training_labels.values.tolist(),
	"X": training_data.values.tolist(),
	"new_X": testing_data.values.tolist(),
}

modelfile = "OverfittingRoshanSharma.stan"
with open(modelfile, "w") as file: file.write("""
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
		vector[K] bbeta; // `beta` is built-in distrib fx
	}

	transformed parameters {
		vector[N] linpred = alpha + X * bbeta;
	}

	model {
		alpha ~ cauchy(0, 10); // prior for the intercept following Gelman 2008
		bbeta ~ student_t(1, 0, 0.03);
		y ~ bernoulli_logit(linpred);
	}

	generated quantities { // y values predicted by the model
		vector[N2] y_pred = alpha + new_X * bbeta;
	}
""")
var_name_array = ["alpha"] + [f"bbeta[{i+1}]" for i in range(mdl_data["K"])]
var_name_combi = ["alpha", "bbeta"]

sm = CmdStanModel(stan_file = modelfile)

# maximum likelihood estimation
optim = sm.optimize(data = mdl_data).optimized_params_pd
optim[optim.columns[~optim.columns.str.startswith("lp")]]
plt.plot(optim[var_name_array[1:]].values[0])

# variational inference
vb = sm.variational(data = mdl_data)
vb.variational_sample.columns = vb.variational_params_dict.keys()
vb_name = vb.variational_params_pd.columns[~vb.variational_params_pd.columns.str.startswith(("lp", "log_"))]
vb.variational_params_pd[var_name_array]
vb.variational_sample[var_name_array]

# Markov chain Monte Carlo
fit = sm.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit.draws().shape # iterations, chains, parameters
fit.summary().loc[var_name_array] # pandas DataFrame
print(fit.diagnose())

# posterior = {k: fit_modif.stan_variable(k) for k in var_name_combi}

az_trace = az.from_cmdstanpy(fit)
az.summary(az_trace).loc[var_name_combi] # pandas DataFrame
az.plot_trace(az_trace, var_names = ["alpha"])
az.plot_forest(az_trace, var_names = ["bbeta"])

sample_pred = fit.stan_variable("y_pred")

# Tim Salimans model: DOES NOT WORK yet
# need to figure out how to marginalize all discrete params
