/*
src: https://www.kaggle.com/c/overfitting

In order to achieve this we have created a simulated data set with 200 variables and 20,000 cases.
An ‘equation’ based on this data was created in order to generate a Target to be predicted.
Given the all 20,000 cases, the problem is very easy to solve – but you only get given the Target value of 250 cases – the task is to build a model that gives the best predictions on the remaining 19,750 cases.
*/

data {
	int N; // the number of training observations
	int N2; // the number of test observations
	int K; // the number of features
	array[N] int y; // the response
	matrix[N,K] X; // the model matrix
	matrix[N2,K] new_X; // the matrix for the predicted values
}

parameters {
	real alpha; // intercept
	vector[K] bbeta; // slopes, `beta` is built-in distrib fx
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
