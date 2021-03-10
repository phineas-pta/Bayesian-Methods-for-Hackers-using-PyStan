# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, seaborn as sns, pandas_datareader.data as web
from cmdstanpy import CmdStanModel
from matplotlib import pyplot as plt, ticker as mtick

#%% load data

stock_names = ["GOOG", "AAPL", "AMZN", "TSLA"]
stock_df = pd.DataFrame()
for stock in stock_names:
	stock_data = web.DataReader(stock, 'yahoo', "2015-09-01", "2018-04-27")
	stock_df[stock] = stock_data["Open"]
stock_df.index = stock_data.index
stock_returns = stock_df.pct_change()[1:]

pd.DataFrame({
	"mu": [-.03, .05, .03, -.02],
	"sigma": [.04, .03, .02, .01]
}, index = stock_names)

stock_returns.mean()
stock_returns.cov()

#%% some plot

cum_returns = np.cumprod(1 + stock_returns) - 1
cum_returns.index = stock_returns.index
cum_returns.plot()
plt.legend(loc = "upper left")
plt.ylabel("Return of $1 on first date")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1))

meltt = stock_returns.melt(var_name = "column")
sns.displot(meltt, x = 'value', hue = 'column', bins = "sqrt", kde = True)
g = sns.FacetGrid(meltt, col = 'column', col_wrap = 2)
g.map(sns.histplot, 'value', bins = "sqrt", kde = True)

#%% model

mdl_data = {"N": len(stock_returns), "N_stocks": len(stock_names), "observations": stock_returns.values}
modelfile = "stocks.stan"
with open(modelfile, "w") as file: file.write("""
	data { // avoid putting data in matrix except for linear algebra
		int<lower=0> N;
		int<lower=0> N_stocks;
		row_vector[N_stocks] observations[N];
	}

	transformed data {
		int<lower=2+N_stocks> df = 10;
		row_vector[N_stocks] expert_mus = [-.03, .05, .03, -.02];
		matrix<lower=0>[N_stocks, N_stocks] expert_sigmas = diag_matrix(square([.04, .03, .02, .01]'));
	}

	parameters { // discrete parameters impossible
		row_vector[N_stocks] locs;
		cov_matrix[N_stocks] covs;
	}

	transformed parameters { // Cholesky form, more numerically stabilized
		cholesky_factor_cov[N_stocks] L = cholesky_decompose(covs);
	}

	model {
		locs ~ normal(expert_mus, 1);
		covs ~ wishart(df, expert_sigmas);
		observations ~ multi_normal_cholesky(locs, L); // failed to initialize if not use Cholesky
	}
""")

sm = CmdStanModel(stan_file = modelfile)

#%% reparameterization for more efficient computation: Bartlett decomposition

modelfile_repar = "stocks_repar.stan"
with open(modelfile_repar, "w") as file: file.write("""
	data { // avoid putting data in matrix except for linear algebra
		int<lower=0> N;
		int<lower=0> N_stocks;
		row_vector[N_stocks] observations[N];
	}

	transformed data {
		int<lower=2+N_stocks> df = 10;
		row_vector[N_stocks] expert_mus = [-.03, .05, .03, -.02];
		matrix<lower=0>[N_stocks, N_stocks] expert_sigmas = diag_matrix(square([.04, .03, .02, .01]'));
		cholesky_factor_cov[N_stocks] L = cholesky_decompose(expert_sigmas);
	}

	parameters { // discrete parameters impossible
		row_vector[N_stocks] locs;
		vector[N_stocks] c;
		vector[N_stocks * (N_stocks - 1) / 2] z;
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
""")

Xrange = range(1, 5)
var_name_repar_array = [f"locs[{i}]" for i in Xrange] + [f"A[{i},{i}]" for i in Xrange]
var_name_repar_combi = ["locs", "A"]

sm_repar = CmdStanModel(stan_file = modelfile_repar)

# maximum likelihood estimation
optim_repar = sm_repar.optimize(data = mdl_data).optimized_params_pd
optim_repar[var_name_repar_array]

# variational inference
vb_repar = sm_repar.variational(data = mdl_data)
vb_repar.variational_sample.columns = vb_repar.variational_params_dict.keys()
vb_name_repar = vb_repar.variational_params_pd.columns[~vb_repar.variational_params_pd.columns.str.startswith(("lp", "log_"))]
vb_repar.variational_params_pd[var_name_repar_array]
vb_repar.variational_sample[var_name_repar_array]

# Markov chain Monte Carlo
fit_repar = sm_repar.sample(
	data = mdl_data, show_progress = True, chains = 4,
	iter_sampling = 50000, iter_warmup = 10000, thin = 5
)

fit_repar.draws().shape # iterations, chains, parameters
fit_repar.summary().loc[var_name_repar_array] # pandas DataFrame
print(fit_repar.diagnose())

posterior_repar = {k: fit_repar.stan_variable(k) for k in var_name_repar_combi}

# re-compose the covariance matrix
L_repar = np.linalg.cholesky(np.diag([.04, .03, .02, .01]))
f = lambda a: L_repar @ a @ a.T @ L_repar.T
posterior_repar["covs"] = np.array([f(a) for a in posterior_repar["A"]])

colors = ['#5DA5DA', '#F15854', '#B276B2', '#60BD68']
fig = plt.figure(figsize = (16, 9))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
for i in range(len(stock_names)):
	sns.kdeplot(posterior_repar["locs"][:,i], color = colors[i], ax = ax1)
	sns.kdeplot(posterior_repar["covs"][:,i,i], color = colors[i], ax = ax2)
ax1.legend(labels = stock_names)
ax1.set_title(r"$ \mu $")
ax2.legend(labels = stock_names)
ax2.set_title(r"$ \sigma $")
