/*
On January 28, 1986, the 25th flight of the USA space shuttle program ended in disaster when one of the rocket boosters of the Shuttle Challenger exploded shortly after lift-off, killing all 7 crew members.
The presidential commission on the accident concluded that it was caused by the failure of an O-ring in a field joint on the rocket booster, and that this failure was due to a faulty design that made the O-ring unacceptably sensitive to a number of factors including outside temperature.
Of the previous 24 flights, data were available on failures of O-rings on 23, (one was lost at sea), and these data were discussed on the evening preceding the Challenger launch, but unfortunately only the data corresponding to the 7 flights on which there was a damage incident were considered important and these were thought to show no obvious trend.

observation: probability of damage incidents occurring increases as the outside temperature decreases:
probability = 1 / (1 + exp(α + β * temperature))
*/

data {
	int<lower=0> N;
	vector[N] temp; // temperature
	int<lower=0> dam; // damage or not
}

parameters {
	real alpha;
	real bbeta; // `beta` is built-in distrib fx
}

transformed parameters {
	vector[N] prob = 1 ./ (1 + exp(bbeta * temp + alpha)); // element-wise
}

model {
	alpha ~ normal(0, 1000);
	bbeta ~ normal(0, 1000);
	dam ~ binomial(N, prob);
}
