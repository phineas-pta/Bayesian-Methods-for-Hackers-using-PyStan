using StatsPlots, ArviZ
using Stan.StanSample: SampleModel, Sample, stan_sample, read_summary, read_samples
using Stan.StanDiagnose: DiagnoseModel, stan_diagnose, read_diagnose
using Stan.StanOptimize: OptimizeModel, stan_optimize, read_optimize
using Stan.StanVariational: VariationalModel, stan_variational, read_variational, read_summary

count_data = [
    13, 24,  8, 24,  7, 35, 14, 11, 15, 11, 22, 22, 11, 57, 11, 19, 29,  6, 19, 12, 22, 12, 18, 72, 32,  9,  7, 13,
    19, 23, 27, 20,  6, 17, 13, 10, 14,  6, 16, 15,  7,  2, 15, 15, 19, 70, 49,  7, 53, 22, 21, 31, 19, 11, 18, 20,
    12, 35, 17, 23, 17,  4,  2, 31, 30, 13, 27,  0, 39, 37,  5, 14, 13, 22,
];
mdl_data = Dict("N" => length(count_data), "obs" => count_data);
var_name = ["lambda1", "lambda2", "tau"];
var_sym = Symbol.(var_name);
mdl_str = "
	data {
		int<lower=0> N;
		array[N] int<lower=0> obs;
	}

	transformed data {
		real alpha = 8.;
		real beta = .3;
		real log_unif = -log(N);
	}

	parameters { // no info about tau
		real<lower=0> lambda1;
		real<lower=0> lambda2;
	}

	transformed parameters { // marginalize out the discrete parameter
		vector[N] lp = rep_vector(log_unif, N);
		vector[N+1] lp1;
		vector[N+1] lp2;
		lp1[1] = 0;
		lp2[1] = 0;
		for (i in 1:N) { // dynamic programming workaround to avoid nested loop
			lp1[i+1] = lp1[i] + poisson_lpmf(obs[i] | lambda1);
			lp2[i+1] = lp2[i] + poisson_lpmf(obs[i] | lambda2);
		}
		lp = rep_vector(log_unif + lp2[N+1], N) + head(lp1, N) - head(lp2, N);
	}

	model {
		lambda1 ~ gamma(alpha, beta);
		lambda2 ~ gamma(alpha, beta);
		target += log_sum_exp(lp);
	}

	generated quantities { // generate tau here
		int<lower=1,upper=N> tau = categorical_logit_rng(lp);
	}
";

sm = Dict(
	"sample" => SampleModel(
		"mdl", mdl_str, [4], # n_chains
		method = Sample(num_samples = 50000, num_warmup = 10000, thin = 5)
	),
	"diagnose" => DiagnoseModel("mdl", mdl_str),
	"optimize" => OptimizeModel("mdl", mdl_str),
	"variational" => VariationalModel("mdl", mdl_str),
);

rc = Dict( # run code?
	"sample" => stan_sample(sm["sample"], data = mdl_data),
	"diagnose" => stan_diagnose(sm["diagnose"], data = mdl_data),
	"optimize" => stan_optimize(sm["optimize"], data = mdl_data),
	"variational" => stan_variational(sm["variational"], data = mdl_data),
);

summ_sample = read_summary(sm["sample"]); # DataFrame
summ_variational = read_summary(sm["variational"]); # DataFrame
summ_sample[summ_sample.parameters .∈ Ref(var_sym),:]
summ_variational[summ_variational.parameters .∈ Ref(var_sym),:]

samples = read_samples(sm["sample"]); # named tuple
optim, cnames = read_optimize(sm["optimize"]);
diags = read_diagnose(sm["diagnose"]);
vi_samples, vi_names = read_variational(sm["variational"]);

idata = from_namedtuple(samples);
summarystats(idata)
plot_trace(idata, var_names = var_name)

optim_data = Dict(i => optim[i] for i ∈ var_name);
vi_data = Dict(i => dropdims(vi_samples[:,findall(x -> x == i, vi_names),:], dims=2) for i ∈ var_name);
samples_data = Dict(i => reshape(samples[Symbol(i)], :, 4) for i ∈ var_name);

for i ∈ var_name
	density(samples_data[i]) |> display
end
