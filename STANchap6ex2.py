# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, pystan, arviz as az, seaborn as sns, pandas_datareader.data as web
from matplotlib import pyplot as plt, ticker as mtick

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

mdl_data = {"N": len(stock_returns), "N_stocks": len(stock_names), "observations": stock_returns.values}
sm = pystan.StanModel(model_name = "simple_mdl", model_code = """
	data {
		int<lower=0> N;
		int<lower=0> N_stocks;
		matrix[N, N_stocks] observations;
	}

	transformed data {
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
		covs ~ wishart(10, expert_sigmas);
		for (i in 1:N) observations[i] ~ multi_normal_cholesky(locs, L); // failed to initialize if not use Cholesky
	}
""")
fit = sm.sampling(
	data = mdl_data, pars = ["locs", "covs"], n_jobs = -1, # parallel
	iter = 50000, chains = 3, warmup = 10000, thin = 5
)
print(fit.stansummary())
fit.extract(permuted = False).shape # iterations, chains, parameters
posterior = fit.extract(permuted = True) # all chains are merged and warmup samples are discarded

fig = plt.figure(figsize = (16, 9))
for i, val in enumerate(stock_names):

	ax = fig.add_subplot(2, 4, i+1)
	sns.histplot(posterior["locs"][:,i], bins = "sqrt", kde = True, ax = ax)
	ax.set_title(f"$ {val}: \\mu $")

	ax = fig.add_subplot(2, 4, 4+i+1)
	sns.histplot(posterior["covs"][:,i,i], bins = "sqrt", kde = True, ax = ax)
	ax.set_title(f"$ {val}: \\sigma $")

