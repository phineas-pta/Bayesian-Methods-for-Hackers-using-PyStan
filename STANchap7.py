# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, prince

data = pd.read_csv("overfitting.csv", index_col = 'case_id')
data.columns
predictors = data[data.columns[data.columns.str.startswith("var_")]]
labels = data["Target_Practice"]

ix_training = data.train == 1
training_data = predictors[ix_training]
training_labels = labels[ix_training]

ix_testing = data.train == 0
testing_data = predictors[ix_testing]
testing_labels = labels[ix_testing]

sns.histplot(training_data.values.flatten(), bins = "sqrt", kde = True)

pca = prince.PCA(n_components = 2, as_array = False).fit(training_data)
pca.plot_row_coordinates(training_data, color_labels = training_labels)
pca.column_correlations(training_data).plot.scatter(x = 0, y = 1) # weird column name

mdl_data = {
	'N': len(ix_training),
	'N2': len(ix_testing),
	'K': len(data.columns.str.startswith("var_")),
	'y': training_labels.values,
	'X': training_data.values,
	'new_X': testing_labels.values,
}

sm = pystan.StanModel(model_name = "Roshan Sharma", model_code = """
	data {
		int N; // the number of training observations
		int N2; // the number of test observations
		int K; // the number of features
		int y[N]; // the response
		matrix[N,K] X; // the model matrix
		matrix[N2,K] new_X; // the matrix for the predicted values
	}

	parameters {
		real alpha;
		vector[K] beta; // the regression parameters
	}

	transformed parameters {
		vector[N] linpred = alpha + X * beta;
	}

	model {
		alpha ~ cauchy(0, 10); // prior for the intercept following Gelman 2008
		beta ~ student_t(1, 0, 0.03);
		y ~ bernoulli_logit(linpred);
	}

	generated quantities {
		vector[N2] y_pred = alpha + new_X * beta; // the y values predicted by the model
	}
""")

# DOES NOT WORK
# need to figure out how to marginalize all discrete params
pystan.StanModel(model_name = "Tim_Salimans", model_code = """
	data { // avoid putting data in matrix except for linear algebra
		int<lower=0> N;
		int<lower=0> N_var;
		row_vector<lower=0, upper=1>[N] T;
		row_vector<lower=0, upper=1>[N_var] obs[N];
	}

	parameters { // discrete parameters impossible
		vector<lower=0, upper=1>[N_var] includes;
	}

	transformed parameters {
		real n_incl = sum(includes);
		row_vector<lower=0, upper=1>[N_var] coef;
		row_vector<lower=0>[N_var] Y[N] = coef * obs * includes;
	}

	model {
		includes ~ bernoulli(.5);
		coef ~ uniform(0, 1);
		T = Y < mean(Y) ? 1 : 0;
	}
""")
