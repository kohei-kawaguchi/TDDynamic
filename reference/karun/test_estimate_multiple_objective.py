# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleAVI,
    EstimatorForEquilibriumMultipleSemiGradient,
    MLEModel,
)
import polars as pl
from td_dynamic.karun.multiple.predictor_multiple import (
  LinearRegressionPredictor,
  CCPLogisticRegressionPredictor
)
from td_dynamic.karun.utils import read_pickle_from_s3, write_pickle_to_s3

# set constants ----------------------------------------------------

degree = 0
predictor_type = "oracle"
num_iteration = 1

prefix = "output/estimate_multiple_objective/"
bucket_name = "football-markov"

# load data --------------------------------------------------------

equilibrium = read_pickle_from_s3(bucket=bucket_name, prefix="output/simulate_multiple/", file_name="equilibrium.pkl")

# evaluate the objective function ----------------------------------

## initialize estimators
predictor_list = [
    LinearRegressionPredictor(equilibrium=equilibrium, predictor_type=predictor_type, degree=degree, i=i)
  for i in range(equilibrium.num["firm"])  
]
ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type=predictor_type, degree=degree)
    for i in range(equilibrium.num["firm"])
]
avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=predictor_list, ccp_predictor_list=ccp_predictor_list)
semi_gradient_estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)

## estiamte h and g for all actions --------------------------------

action_set = avi_estimator.make_action_set(result=equilibrium.result, num_action=equilibrium.num["action"], num_firm=equilibrium.num["firm"])

df = avi_estimator.expand_result_action(result=equilibrium.result, num_action=equilibrium.num["action"], num_firm=equilibrium.num["firm"])

h_semi, _, _ = semi_gradient_estimator.estimate_h(predictor_type=predictor_type, degree=degree)
g_semi, _, _ = semi_gradient_estimator.estimate_g(predictor_type=predictor_type, degree=degree)

h_all_semi, g_all_semi = semi_gradient_estimator.estimate_h_g_all()

h, predictor_list_h = avi_estimator.estimate_h(
    initial_h=h_semi,
    num_iteration=num_iteration,
)
g, predictor_list_g = avi_estimator.estimate_g(
    initial_g=g_semi,
    num_iteration=num_iteration,
)

h_all, g_all = avi_estimator.estimate_h_g_all(predictor_list_h=predictor_list_h, predictor_list_g=predictor_list_g)

## evaluate the loglikelihood --------------------------------------

x = np.array(
    [
        avi_estimator.equilibrium.param["param_payoff"]["beta"],
        avi_estimator.equilibrium.param["param_payoff"]["alpha"],
        avi_estimator.equilibrium.param["param_payoff"]["lambda"],
    ]
)

### use avi h and g -----------------------------------------------

# MLE
mle_model = MLEModel(
    endog=np.random.normal(size=equilibrium.result.shape[0]),
    h=h,
    g=g,
    h_all=h_all,
    g_all=g_all,
    result=equilibrium.result,
    action_set=action_set
)

denominator, benchmark = mle_model.compute_denominator(x=x)

denominator, benchmark

numerator = mle_model.compute_numerator(x=x, benchmark=benchmark)

numerator

likelihood = mle_model.compute_likelihood_individual(numerator=numerator, denominator=denominator)

likelihood

loglikelihood = mle_model.compute_loglikelihood(params=x)

loglikelihood

# Define a range of values around each element of x
x_range = np.linspace(x * 0.1, x * 3, 100)

# Initialize lists to store loglikelihood values
loglikelihood_values_theta = []
loglikelihood_values_alpha = []
loglikelihood_values_lambda = []

# Calculate loglikelihood for varying beta
for beta in x_range[:, 0]:
    x_temp = np.array([beta, x[1], x[2]])
    loglikelihood_temp = mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_theta.append(loglikelihood_temp)

for alpha in x_range[:, 1]:
    x_temp = np.array([x[0], alpha, x[2]])
    loglikelihood_temp = mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_alpha.append(loglikelihood_temp)

# Calculate loglikelihood for varying lambda
for lambda_val in x_range[:, 1]:
    x_temp = np.array([x[0], x[1], lambda_val])
    loglikelihood_temp = mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_lambda.append(loglikelihood_temp)


# Create the plot
# should be maximized at the true value
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot for beta
ax1.plot(x_range[:, 0], loglikelihood_values_theta)
ax1.set_title("Log-likelihood vs beta")
ax1.set_xlabel("beta")
ax1.set_ylabel("Log-likelihood")
ax1.axvline(x=x[0], color="r", linestyle="--", label="True value")
ax1.legend()

# Plot for alpha
ax2.plot(x_range[:, 1], loglikelihood_values_alpha)
ax2.set_title("Log-likelihood vs alpha")
ax2.set_xlabel("alpha")
ax2.set_ylabel("Log-likelihood")
ax2.axvline(x=x[1], color="r", linestyle="--", label="True value")
ax2.legend()

# Plot for lambda
ax3.plot(x_range[:, 1], loglikelihood_values_lambda)
ax3.set_title("Log-likelihood vs Lambda")
ax3.set_xlabel("Lambda")
ax3.set_ylabel("Log-likelihood")
ax3.axvline(x=x[2], color="r", linestyle="--", label="True value")
ax3.legend()

plt.tight_layout()
plt.show()


### use semi-gradient method to estimate h and g ----------------------

# MLE
mle_model_semi = MLEModel(
    endog=np.random.normal(size=equilibrium.result.shape[0]),
    h=h_semi,
    g=g_semi,
    h_all=h_all_semi,
    g_all=g_all_semi,
    result=equilibrium.result,
    action_set=action_set
)

denominator, benchmark = mle_model_semi.compute_denominator(x=x)

denominator, benchmark

numerator = mle_model_semi.compute_numerator(x=x, benchmark=benchmark)

numerator

likelihood = mle_model_semi.compute_likelihood_individual(numerator=numerator, denominator=denominator)

likelihood

loglikelihood = mle_model_semi.compute_loglikelihood(params=x)

loglikelihood

# Define a range of values around each element of x
x_range = np.linspace(x * 0.1, x * 3, 100)

# Initialize lists to store loglikelihood values
loglikelihood_values_theta = []
loglikelihood_values_alpha = []
loglikelihood_values_lambda = []

# Calculate loglikelihood for varying beta
for beta in x_range[:, 0]:
    x_temp = np.array([beta, x[1], x[2]])
    loglikelihood_temp = mle_model_semi.compute_loglikelihood(params=x_temp)
    loglikelihood_values_theta.append(loglikelihood_temp)

for alpha in x_range[:, 1]:
    x_temp = np.array([x[0], alpha, x[2]])
    loglikelihood_temp = mle_model_semi.compute_loglikelihood(params=x_temp)
    loglikelihood_values_alpha.append(loglikelihood_temp)

# Calculate loglikelihood for varying lambda
for lambda_val in x_range[:, 1]:
    x_temp = np.array([x[0], x[1], lambda_val])
    loglikelihood_temp = mle_model_semi.compute_loglikelihood(params=x_temp)
    loglikelihood_values_lambda.append(loglikelihood_temp)


# Create the plot
# should be maximized at the true value
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot for beta
ax1.plot(x_range[:, 0], loglikelihood_values_theta)
ax1.set_title("Log-likelihood vs beta")
ax1.set_xlabel("beta")
ax1.set_ylabel("Log-likelihood")
ax1.axvline(x=x[0], color="r", linestyle="--", label="True value")
ax1.legend()

# Plot for alpha
ax2.plot(x_range[:, 1], loglikelihood_values_alpha)
ax2.set_title("Log-likelihood vs alpha")
ax2.set_xlabel("alpha")
ax2.set_ylabel("Log-likelihood")
ax2.axvline(x=x[1], color="r", linestyle="--", label="True value")
ax2.legend()

# Plot for lambda
ax3.plot(x_range[:, 1], loglikelihood_values_lambda)
ax3.set_title("Log-likelihood vs Lambda")
ax3.set_xlabel("Lambda")
ax3.set_ylabel("Log-likelihood")
ax3.axvline(x=x[2], color="r", linestyle="--", label="True value")
ax3.legend()

plt.tight_layout()
plt.show()

# estimate the parameter --------------------------------------------

result_semi_mle = mle_model_semi.fit(start_params=x)
result_semi_mle.summary()

semi_gradient_estimator.estimate_params(predictor_type=predictor_type, degree=degree)

result_mle = mle_model.fit(start_params=x)
result_mle.summary()

avi_estimator.estimate_params(initial_h=h_semi, initial_g=g_semi, num_iteration=num_iteration)
