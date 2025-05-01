/*
src:
- https://www1.swarthmore.edu/NatSci/peverso1/Sports%20Data/JamesSteinData/Efron-Morris%20Baseball/EfronMorrisBB.txt
- https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
- https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html

Suppose you are tasked with estimating baseball batting skills for several players.
One such performance metric is batting average.
Since players play a different number of games and bat in different positions in the order, each player has a different number of at-bats.
However, you want to estimate the skill of all players, including those with a relatively small number of batting opportunities.
The idea of hierarchical partial pooling is to model the global performance, and use that estimate to parameterize a population of players that accounts for differences among the players’ performances.
*/

data {
	int<lower=0> N;
	array[N] int<lower=0> at_bats;
	array[N] int<lower=0> hits;
}

parameters {
	real log_kappa; // reparameterization for better sampling because Pareto distribution has very long tails
	real<lower=0, upper=1> phi; // hidden factor related to the expected performance for all players
	vector<lower=0, upper=1>[N] thetas; // chance of success
}

transformed parameters {
	real<lower=0> kappa = exp(log_kappa); // hyperparameter to account for the variance in the population batting averages
	real<lower=0> alpha = kappa * phi;
	real<lower=0> bbeta = kappa * (1 - phi); // `beta` is built-in distrib fx
}

model {
	log_kappa ~ exponential(1.5); // hyperprior
	phi ~ uniform(0, 1);
	thetas ~ beta(alpha, bbeta);
	hits ~ binomial(at_bats, thetas);
}
