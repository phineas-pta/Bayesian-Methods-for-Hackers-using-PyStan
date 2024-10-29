# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf, seaborn as sns, arviz as az, tensorflow_probability as tfp
tf.config.optimizer.set_jit(True)
tfd, tfb = tfp.distributions, tfp.bijectors

def snsdistplot(data):
	g = sns.displot(data, bins = "sqrt", kde = True)
	g.fig.set_figwidth(8)
	g.fig.set_figheight(6)
	return g

data = np.loadtxt("data/mixture_data.csv")
snsdistplot(data)

var_name = ["prob", "centers", "sigmas"]

@tfd.JointDistributionCoroutineAutoBatched
def mdl_batch():
	prob = yield tfd.Uniform(low = 0., high = 1., name = var_name[0])
	centers = yield tfd.Normal(loc = [120., 190.], scale = [10.]*2, name = var_name[1])
	sigmas = yield tfd.Uniform(low = [0.]*2, high = [100.]*2, name = var_name[2])
	categories = tfd.Categorical(probs = [prob, 1. - prob]) # assignments to a group
	values = tfd.Normal(loc = centers, scale = sigmas) # group
	mixture = tfd.MixtureSameFamily(mixture_distribution = categories, components_distribution = values)
	yield tfd.Sample(mixture, sample_shape = len(data), name = "obs")

# a helper function in McMC chain
def trace_fn(current_state, kernel_results):
	mdr = kernel_results.inner_results.inner_results
	return mdr.target_log_prob, mdr.leapfrogs_taken, mdr.has_divergence, mdr.energy, mdr.log_accept_ratio

@tf.function(autograph = False, experimental_compile = True) # speed up a lot the McMC sampling
def run_mcmc( # pass numeric arguments as Tensors whenever possible
	init_state, unconstraining_bijectors,
	num_steps = 50000, burnin = 10000,
	num_leapfrog_steps = 3, step_size = .5
):
	kernel0 = tfp.mcmc.NoUTurnSampler(
		target_log_prob_fn = lambda *args: mdl_batch.log_prob(obs=data, *args),
		step_size = step_size
	)
	kernel1 = tfp.mcmc.TransformedTransitionKernel(
		inner_kernel= kernel0,
		bijector = unconstraining_bijectors
	)
	kernel2 = tfp.mcmc.DualAveragingStepSizeAdaptation( # pkr = previous kernel results
		inner_kernel = kernel1,
		num_adaptation_steps = int(0.8*burnin),
		step_size_setter_fn = lambda pkr, new_step_size: pkr._replace(inner_results = pkr.inner_results._replace(step_size=new_step_size)),
		step_size_getter_fn = lambda pkr: pkr.inner_results.step_size,
		log_accept_prob_getter_fn = lambda pkr: pkr.inner_results.log_accept_ratio
	)
	# tf.get_logger().setLevel("ERROR") # multiple chains
	return tfp.mcmc.sample_chain( # ATTENTION: 2 values to unpack
		num_results = num_steps,
		num_burnin_steps = burnin,
		current_state = init_state,
		kernel = kernel2,
		trace_fn = trace_fn
	)

nchain = 4
init_state = [mdl_batch.sample(nchain)._asdict()[_] for _ in var_name]
unconstraining_bijectors = [tfb.Identity()]*len(var_name)
samples, sampler_stat = run_mcmc(init_state, unconstraining_bijectors)

#%% using the pymc3 naming convention, with log_likelihood instead of lp so that ArviZ can compute loo and waic
sample_stats_name = ["log_likelihood", "tree_size", "diverging", "energy", "mean_tree_accept"]

sample_stats = {k: v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}
posterior = {k:np.swapaxes(v.numpy(), 1, 0) for k, v in zip(var_name, samples)}
az_trace = az.from_dict(posterior = posterior, sample_stats = sample_stats)

snsdistplot(posterior["prob"])
snsdistplot(posterior["centers"][:, 0])

# put the data into a tensor
datatf = tf.constant(data, dtype = tf.float32)[:, tf.newaxis]

# This produces a cluster per MCMC chain
rv_clusters_1 = tfd.Normal(posterior["centers"][:, 0], posterior["sigmas"][:, 0])
rv_clusters_2 = tfd.Normal(posterior["centers"][:, 1], posterior["sigmas"][:, 1])

# Compute the un-normalized log probabilities for each cluster
cluster_1_log_prob = rv_clusters_1.log_prob(datatf) + tf.math.log(posterior["prob"])
cluster_2_log_prob = rv_clusters_2.log_prob(datatf) + tf.math.log(1. - posterior["prob"])

# Bayes rule to compute the assignment probability: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_1 = cluster_1_log_prob - tf.math.reduce_logsumexp(tf.stack([cluster_1_log_prob, cluster_2_log_prob], axis=-1), -1)

# Average across the MCMC chain
log_p_assign_1bis = tf.math.reduce_logsumexp(log_p_assign_1, -1) - tf.math.log(tf.cast(log_p_assign_1.shape[-1], tf.float32))

p_assign_1 = tf.exp(log_p_assign_1bis)
p_assign = tf.stack([p_assign_1, 1 - p_assign_1], axis=-1)

assign_trace = log_p_assign_1bis.numpy()[np.argsort(data)]
plt.scatter(data[np.argsort(data)], assign_trace, cmap = "RdBu",c = (1 - assign_trace), s = 50)
plt.title("Probability of data point belonging to cluster 0")
plt.ylabel("probability")
plt.xlabel("value of data point")
