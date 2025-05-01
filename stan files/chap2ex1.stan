/*
Assume that there is some true 0 < p_A < 1 probability that users who, upon shown site A, eventually purchase from the site. This is the true effectiveness of site A. Currently, this quantity is unknown to us.

Suppose site A was shown to N people, and n people purchased from the site. One might conclude hastily that p_A = n / N. Unfortunately, the observed frequency  does not necessarily equal p_A - there is a difference between the observed frequency and the true frequency of an event. We are interested in using what we know, N (the total trials administered) and n (the number of conversions), to estimate what p_A, the true frequency of buyers, might be.
*/

data {
	int<lower=0> N;
	int<lower=0, upper=N> occur;
}

parameters {
	real<lower=0, upper=1> probA;
}

model {
	occur ~ binomial(N, probA);
}
