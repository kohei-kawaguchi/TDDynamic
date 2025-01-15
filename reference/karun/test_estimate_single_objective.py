# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from td_dynamic.karun.single.estimate_single import (
    EstimatorForEquilibriumSingleAVI,
    EstimatorForEquilibriumSingleSemiGradient,
    MLEModel,
)
import polars as pl
from td_dynamic.karun.single.predictor_single import LinearRegressionPredictor
from td_dynamic.karun.utils import read_pickle_from_s3

# set constants ----------------------------------------------------

degree = 0
predictor_type = "oracle"
num_iteration = 50

prefix = "output/estimate_single_objective/"
bucket_name = "football-markov"

# load data --------------------------------------------------------

equilibrium = read_pickle_from_s3(bucket=bucket_name, prefix="output/simulate_single/", file_name="equilibrium.pkl")

# compare the theoretical and empirical ccp ------------------------

ccp = (
    equilibrium.result.group_by(["state_0", "state_1", "action_index"])
    .agg(ccp_empirical=pl.col("action_index").count())
    .with_columns(ccp_empirical=pl.col("ccp_empirical") / pl.col("ccp_empirical").sum().over("state_0", "state_1"))
)

ccp_theoretical = (
    equilibrium.value["ccp_index"]
    .with_columns(ccp_theoretical=equilibrium.value["ccp_value"].flatten())
    .with_columns(action=pl.col("action").cast(pl.Int64))
    .rename({"action": "action_index"})
)

ccp = ccp.join(ccp_theoretical, on=["state_0", "state_1", "action_index"], how="left")

# Create a scatter plot of theoretical vs empirical CCPs
# perfect match is expecte
plt.figure(figsize=(10, 10))
plt.scatter(ccp["ccp_theoretical"], ccp["ccp_empirical"], alpha=0.5)
plt.plot([0, 1], [0, 1], "r--")  # Add a diagonal line for reference

plt.xlabel("Theoretical CCP")
plt.ylabel("Empirical CCP")
plt.title("Theoretical vs Empirical Conditional Choice Probabilities")
plt.tight_layout()
plt.show()

# evaluate the objective function ----------------------------------

## initialize estimators
predictor = LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=2)
avi_estimator = EstimatorForEquilibriumSingleAVI(equilibrium=equilibrium, predictor=predictor)
semi_gradient_estimator = EstimatorForEquilibriumSingleSemiGradient(equilibrium=equilibrium)
result_semi_gradient = semi_gradient_estimator.estimate_params(predictor_type="polynomial", degree=2)
## compute the likelihood ------------------------------------------

x = np.array(
    [
        equilibrium.param["param_payoff"]["beta"],
        equilibrium.param["param_payoff"]["lambda"],
    ]
)

param_payoff = {"beta": x[0], "lambda": x[1]}

h, h_model_list = avi_estimator.estimate_h(
    initial_h=semi_gradient_estimator.h,
    num_iteration=num_iteration,
)
g, g_model_list = avi_estimator.estimate_g(
    initial_g=semi_gradient_estimator.g,
    num_iteration=num_iteration,
)

h_semi, _, _ = semi_gradient_estimator.estimate_h(predictor_type=predictor_type, degree=degree)
g_semi, _, _ = semi_gradient_estimator.estimate_g(predictor_type=predictor_type, degree=degree)


# calculate the true value

choice_value = equilibrium.value["choice_value"]
choice_index = equilibrium.value["ccp_index"]

true_value = (
    pl.concat([choice_index, pl.DataFrame({"true": choice_value.flatten()})], how="horizontal")
    .rename({"action": "action_index"})
    .with_columns(pl.col("action_index").cast(pl.Int64))
)

# calculate the estimated value

estimated_value = np.dot(h, x) + g.flatten()

# compare true value with estimated values
# perfect match is expecte
comparison = equilibrium.result.join(
    true_value,
    on=["action_index"] + [col for col in true_value.columns if col.startswith("state_")],
).with_columns(pl.Series(name="estimate", values=estimated_value))

plt.scatter(comparison["true"], comparison["estimate"])
plt.xlabel("True Values")
plt.ylabel("Estimated Values")
plt.title("Scatter Plot between True and Estimated Values")
plt.plot(
    [comparison["true"].min(), comparison["true"].max()],
    [comparison["true"].min(), comparison["true"].max()],
    "k--",
    lw=1,
)
plt.show()

## compare with the true ccp -----------------------------------------

# perfect match is expected
comparison = (
    comparison.unique(subset=["state_0", "state_1", "action_index"])
    .with_columns(
        ccp_estimated=np.exp(pl.col("estimate")) / np.exp(pl.col("estimate")).sum().over("state_0", "state_1")
    )
    .join(ccp_theoretical, on=["state_0", "state_1", "action_index"])
)

plt.scatter(comparison["ccp_theoretical"], comparison["ccp_estimated"])
plt.xlabel("True CCP")
plt.ylabel("Estimated CCP")
plt.title("Scatter Plot between True and Estimated CCP")
plt.plot(
    [comparison["ccp_theoretical"].min(), comparison["ccp_theoretical"].max()],
    [comparison["ccp_theoretical"].min(), comparison["ccp_theoretical"].max()],
    "k--",
    lw=1,
)
plt.show()


# check that the ccp from h and g coincide with the theoretical ccp

df = equilibrium.result.select(
    "action_index",
    *[col for col in equilibrium.result.columns if col.startswith("state_")],
).unique()
action_state = df.select("action_index", *[col for col in df.columns if col.startswith("state_value_")])

h_check = [model.predict(np.array(action_state)) for model in h_model_list]
h_check = np.array(h_check).T

g_check = [model.predict(np.array(action_state)) for model in g_model_list]
g_check = np.array(g_check).T

v_check = np.dot(h_check, x) + g_check.flatten()
v_check = np.exp(v_check)

df = df.with_columns(ccp_hg=v_check).with_columns(
    ccp_hg=pl.col("ccp_hg") / pl.col("ccp_hg").sum().over("state_0", "state_1")
)

df = df.join(ccp_theoretical, on=["state_0", "state_1", "action_index"], how="left").sort(
    ["state_0", "state_1", "action_index"]
)

# Create a scatter plot of ccp_hg vs ccp_theoretical
# perfect match is expected
plt.figure(figsize=(10, 10))
plt.scatter(df["ccp_theoretical"], df["ccp_hg"], alpha=0.5)
plt.plot([0, 1], [0, 1], "r--")  # Add a diagonal line for reference

plt.xlabel("Theoretical CCP")
plt.ylabel("Estimated CCP (h_g)")
plt.title("Theoretical vs Estimated Conditional Choice Probabilities")
plt.tight_layout()
plt.show()

# basis = fit_basis_action_state(equilibrium=equilibrium, predictor_type=predictor_type, degree=degree)
# current_selector, next_selector = make_selector(equilibrium=equilibrium)
# basis_action_state = transform_basis_action_state(result=equilibrium.result, basis=basis)
# payoff_covariate = compute_payoff_covariate_from_result(equilibrium=equilibrium)

# # estimate omega

# omega = estimate_omega(
#     payoff_covariate=payoff_covariate,
#     basis_action_state=basis_action_state,
#     current_selector=current_selector,
#     next_selector=next_selector,
#     discount=equilibrium.discount,
# )

# ccp = estimate_ccp_count(equilibrium=equilibrium)
# e = compute_conditional_expected_shock_from_result(equilibrium=equilibrium, ccp=ccp)
# xi = estimate_xi(
#     e=e,
#     basis_action_state=basis_action_state,
#     current_selector=current_selector,
#     next_selector=next_selector,
#     discount=equilibrium.discount,
# )

h_all, g_all = avi_estimator.estimate_h_g_all(
    h_predictor_list=h_model_list,
    g_predictor_list=g_model_list,
)

h_all_semi, g_all_semi = semi_gradient_estimator.estimate_h_g_all()


# MLE
mle_model = MLEModel(
    endog=np.random.normal(size=equilibrium.result.shape[0]),
    h=h,
    g=g,
    h_all=h_all,
    g_all=g_all,
    result=equilibrium.result,
    action_set=equilibrium.action_set,
)

denominator, benchmark = mle_model.compute_denominator(x=x)

denominator, benchmark

numerator = mle_model.compute_numerator(x=x, benchmark=benchmark)

numerator

likelihood = mle_model.compute_likelihood_individual(numerator=numerator, denominator=denominator)

likelihood

loglikelihood = mle_model.compute_loglikelihood(params=x)


loglikelihood

# use avi to estimate h and g --------------------------------------

# Define a range of values around each element of x
x_range = np.linspace(x * 0.1, x * 3, 100)

# Initialize lists to store loglikelihood values
loglikelihood_values_theta = []
loglikelihood_values_lambda = []

# Calculate loglikelihood for varying beta
for beta in x_range[:, 0]:
    x_temp = np.array([beta, x[1]])
    loglikelihood_temp = mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_theta.append(loglikelihood_temp)


# Calculate loglikelihood for varying lambda
for lambda_val in x_range[:, 1]:
    x_temp = np.array([x[0], lambda_val])
    loglikelihood_temp = mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_lambda.append(loglikelihood_temp)


# Create the plot
# should be maximized at the true value
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot for beta
ax1.plot(x_range[:, 0], loglikelihood_values_theta)
ax1.set_title("Log-likelihood vs beta")
ax1.set_xlabel("beta")
ax1.set_ylabel("Log-likelihood")
ax1.axvline(x=x[0], color="r", linestyle="--", label="True value")
ax1.legend()

# Plot for lambda
ax2.plot(x_range[:, 1], loglikelihood_values_lambda)
ax2.set_title("Log-likelihood vs Lambda")
ax2.set_xlabel("Lambda")
ax2.set_ylabel("Log-likelihood")
ax2.axvline(x=x[1], color="r", linestyle="--", label="True value")
ax2.legend()

plt.tight_layout()
plt.show()


# use semi-gradient method to estimate h and g ----------------------
# MLE model
semi_mle_model = MLEModel(
    endog=np.random.normal(size=equilibrium.result.shape[0]),
    h=h_semi,
    g=g_semi,
    h_all=h_all_semi,
    g_all=g_all_semi,
    result=equilibrium.result,
    action_set=equilibrium.action_set,
)
denominator, benchmark = semi_mle_model.compute_denominator(x=x)

denominator, benchmark

numerator = semi_mle_model.compute_numerator(x=x, benchmark=benchmark)

numerator

likelihood = semi_mle_model.compute_likelihood_individual(numerator=numerator, denominator=denominator)

likelihood

loglikelihood = semi_mle_model.compute_loglikelihood(params=x)

loglikelihood

# Initialize lists to store loglikelihood values
loglikelihood_values_theta = []
loglikelihood_values_lambda = []

# Calculate loglikelihood for varying beta
for beta in x_range[:, 0]:
    x_temp = np.array([beta, x[1]])
    loglikelihood_temp = semi_mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_theta.append(loglikelihood_temp)


# Calculate loglikelihood for varying lambda
for lambda_val in x_range[:, 1]:
    x_temp = np.array([x[0], lambda_val])
    loglikelihood_temp = semi_mle_model.compute_loglikelihood(params=x_temp)
    loglikelihood_values_lambda.append(loglikelihood_temp)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot for beta
# should be maximized at the true value
ax1.plot(x_range[:, 0], loglikelihood_values_theta)
ax1.set_title("Log-likelihood vs beta")
ax1.set_xlabel("beta")
ax1.set_ylabel("Log-likelihood")
ax1.axvline(x=x[0], color="r", linestyle="--", label="True value")
ax1.legend()

# Plot for lambda
ax2.plot(x_range[:, 1], loglikelihood_values_lambda)
ax2.set_title("Log-likelihood vs Lambda")
ax2.set_xlabel("Lambda")
ax2.set_ylabel("Log-likelihood")
ax2.axvline(x=x[1], color="r", linestyle="--", label="True value")
ax2.legend()

plt.tight_layout()
plt.show()

# estimate the parameter --------------------------------------------

result_mle = mle_model.fit(start_params=x)
result_mle.summary()

result_semi_mle = semi_mle_model.fit(start_params=x)
result_semi_mle.summary()
