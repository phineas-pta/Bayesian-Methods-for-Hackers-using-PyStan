/*
In the interview process for each student, the student flips a coin, hidden from the interviewer.
The student agrees to answer honestly if the coin comes up heads.
Otherwise, if the coin comes up tails, the student (secretly) flips the coin again, and answers “Yes, I did cheat” if the coin flip lands heads, and “No, I did not cheat”, if the coin flip lands tails.
This way, the interviewer does not know if a “Yes” was the result of a guilty plea, or a Heads on a second coin toss.
Thus privacy is preserved and the researchers receive honest answers.

┬ cheat = no  ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
|             |                  └ 2nd flip = heads » answer = YES
|             └ 1st flip = heads                    » answer = no
└ cheat = yes ┬ 1st flip = tails ┬ 2nd flip = tails » answer = no
              |                  └ 2nd flip = heads » answer = YES
              └ 1st flip = heads                    » answer = YES
►►► prob_yes = .5 × prob_cheat + .5² (0.5 = prob flip coin)
*/

data {
	int<lower=0> N;
	int<lower=0, upper=N> occur;
}

transformed data {
	real<lower=0, upper=1> prob_coin = .5;
	real<lower=0, upper=1> flip1 = binomial_rng(N, prob_coin) * 1. / N; // trick to make int->real
	real<lower=0, upper=1> flip2 = binomial_rng(N, prob_coin) * 1. / N;;
}

parameters {
	real<lower=0, upper=1> prob_cheat;
}

transformed parameters {
	real<lower=0, upper=1> prob_yes = flip1 * prob_cheat + (1 - flip1) * flip2;
}

model {
	occur ~ binomial(N, prob_yes);
}
