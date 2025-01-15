# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.single.estimate_single import (
    EstimatorForEquilibriumSingle,
    EstimatorForEquilibriumSingleAVI,
    EstimatorForEquilibriumSingleSemiGradient,
)
from td_dynamic.karun.single.predictor_single import (
    MLPPredictor,
    LinearRegressionPredictor,
    OracleLinearRegression,
)
from td_dynamic.karun.utils import read_pickle_from_s3
import torch

# set constants ----------------------------------------------------

prefix = "output/estimate_single_avi/"
bucket_name = "football-markov"
num_iteration = 50
predictor_type = "polynomial"
degree = 2

# load data --------------------------------------------------------


equilibrium = read_pickle_from_s3(bucket=bucket_name, prefix="output/simulate_single/", file_name="equilibrium.pkl")
predictor = LinearRegressionPredictor(equilibrium=equilibrium, predictor_type=predictor_type, degree=degree)
# predictor = MLPPredictor(
#     equilibrium=equilibrium,
#     predictor_type="oracle",
#     degree=2,
#     hidden_layer_sizes=(16,),
#     learning_rate=0.001,
#     batch_size=256,
#     num_epochs=2,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# )
avi_estimator = EstimatorForEquilibriumSingleAVI(equilibrium=equilibrium, predictor=predictor)

# estimate using semi gradient method ------------------------------
semi_gradient_estimator = EstimatorForEquilibriumSingleSemiGradient(equilibrium=equilibrium)
result_semi_gradient = semi_gradient_estimator.estimate_params(predictor_type="polynomial", degree=2)

initial_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)
initial_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)

# estimate using avi -----------------------------------------------

## estimate h -------------------------------------------------------

self = avi_estimator

# Prepare the data
action_state = self.equilibrium.result.select(
    "action_index",
    *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
)
action_state = np.array(action_state)
# Initialize a list to store predictors
predictor_list = [deepcopy(self.predictor) for _ in range(initial_h.shape[1])]

predictor_list = self.update_predictor(predictor_list=predictor_list, X=action_state, Y=initial_h)

predictor_list = self.initialize_h_predictor(initial_h=initial_h)

h, h_predictor_list = avi_estimator.estimate_h(initial_h=initial_h, num_iteration=num_iteration)

oracle_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)

# Flatten h and oracle_h for scatter plot
# may not match perfectly
h_flat = h.flatten()
oracle_h_flat = oracle_h.flatten()

plt.figure()
plt.scatter(oracle_h_flat, h_flat)
plt.xlabel("OracleActionState h Values")
plt.ylabel("Estimated h Values")
plt.title("Scatter Plot between OracleActionState h and Estimated h Values")
plt.plot(
    [oracle_h_flat.min(), oracle_h_flat.max()],
    [oracle_h_flat.min(), oracle_h_flat.max()],
    "k--",
    lw=1,
)
plt.show()

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)
oracle_avi_estimator = EstimatorForEquilibriumSingleAVI(
    equilibrium=equilibrium, predictor=OracleLinearRegression(basis=basis)
)

h, predictor_list = oracle_avi_estimator.estimate_h(
    initial_h=oracle_h,
    num_iteration=10,
)

# Flatten h and oracle_h for scatter plot
# perfect match is expected
h_flat = h.flatten()
oracle_h_flat = oracle_h.flatten()

plt.figure()
plt.scatter(oracle_h_flat, h_flat)
plt.xlabel("OracleActionState h Values")
plt.ylabel("Estimated h Values")
plt.title("Scatter Plot between OracleActionState h and Estimated h Values")
plt.plot(
    [oracle_h_flat.min(), oracle_h_flat.max()],
    [oracle_h_flat.min(), oracle_h_flat.max()],
    "k--",
    lw=1,
)
plt.show()

## estimate g ----------------------------------------------------- 

self = avi_estimator

predictor_list = self.initialize_g_predictor(initial_g=initial_g)

g, g_predictor_list = self.estimate_g(initial_g=initial_g, num_iteration=num_iteration)

# Assuming oracle_g is already defined or computed elsewhere in the code
oracle_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)

# Flatten g and oracle_g for scatter plot
g_flat = g.flatten()
oracle_g_flat = oracle_g.flatten()

# may not match perfectly
plt.figure()
plt.scatter(oracle_g_flat, g_flat)
plt.xlabel("OracleActionState g Values")
plt.ylabel("Estimated g Values")
plt.title("Scatter Plot between OracleActionState g and Estimated g Values")
plt.plot(
    [oracle_g_flat.min(), oracle_g_flat.max()],
    [oracle_g_flat.min(), oracle_g_flat.max()],
    "k--",
    lw=1,
)
plt.show()

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)
oracle_avi_estimator = EstimatorForEquilibriumSingleAVI(
    equilibrium=equilibrium, predictor=OracleLinearRegression(basis=basis)
)

g, predictor_list = oracle_avi_estimator.estimate_g(
    initial_g=oracle_g,
    num_iteration=10,
)

# Flatten g and oracle_g for scatter plot
# perfect match is expected
g_flat = g.flatten()
oracle_g_flat = oracle_g.flatten()

plt.figure()
plt.scatter(oracle_g_flat, g_flat)
plt.xlabel("OracleActionState g Values")
plt.ylabel("Estimated g Values")
plt.title("Scatter Plot between OracleActionState g and Estimated g Values")
plt.plot(
    [oracle_g_flat.min(), oracle_g_flat.max()],
    [oracle_g_flat.min(), oracle_g_flat.max()],
    "k--",
    lw=1,
)
plt.show()

## compare with the true choice value ------------------------------

# obtain the true value
choice_value = equilibrium.value["choice_value"]

true_value = (
    pl.concat(
        [
            equilibrium.value["ccp_index"],
            pl.DataFrame({"true": choice_value.flatten()}),
        ],
        how="horizontal",
    )
    .rename({"action": "action_index"})
    .with_columns(pl.col("action_index").cast(pl.Int32))
)

# calculate the estimated value

h, predictor_list = oracle_avi_estimator.estimate_h(
    initial_h=oracle_h,
    num_iteration=10,
)

g, predictor_list = oracle_avi_estimator.estimate_g(
    initial_g=oracle_g,
    num_iteration=10,
)

estimated_value = (
    np.dot(
        h,
        [
            equilibrium.param["param_payoff"]["beta"],
            equilibrium.param["param_payoff"]["lambda"],
        ],
    )
    + g.flatten()
)

# compare true value with estimated values

comparison = (
    equilibrium.result.with_columns(pl.col("action_index").cast(pl.Int32))
    .join(
        true_value,
        on=["action_index"] + [col for col in true_value.columns if col.startswith("state_")],
    )
    .with_columns(pl.Series(name="estimate", values=estimated_value))
)

# perfect match is expected
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

ccp = (
    equilibrium.value["ccp_index"]
    .with_columns(ccp_true=equilibrium.value["ccp_value"].flatten())
    .with_columns(
        pl.col("action").cast(pl.Int32),
    )
    .rename({"action": "action_index"})
)

comparison = (
    comparison.unique(subset=["state_0", "state_1", "action_index"])
    .with_columns(
        ccp_estimated=np.exp(pl.col("estimate")) / np.exp(pl.col("estimate")).sum().over("state_0", "state_1")
    )
    .join(ccp, on=["state_0", "state_1", "action_index"])
)

# perfect match is expected
plt.scatter(comparison["ccp_true"], comparison["ccp_estimated"])
plt.xlabel("True Values")
plt.ylabel("Estimated Values")
plt.title("Scatter Plot between True and Estimated Values")
plt.plot(
    [comparison["ccp_true"].min(), comparison["ccp_true"].max()],
    [comparison["ccp_true"].min(), comparison["ccp_true"].max()],
    "k--",
    lw=1,
)
plt.show()

# debug for estimating h ---------------------------------------------

## does the oracle regression perfectly fit the oracle h?
## perfect match is expected

Y, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)

# prepare the data
result = equilibrium.result
action_state = result.select("action_index", *[col for col in result.columns if col.startswith("state_value_")])
X = np.array(action_state)

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)

i = 0
y = np.array(Y)[:, i]
model = OracleLinearRegression(basis=basis)
model.fit(X, y)
model.coef_

# Compare raw y and predicted y
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Raw y")
plt.ylabel("Predicted y")
plt.title("Comparison of Raw y and Predicted y")
plt.tight_layout()
plt.show()

# perfect match is expected
i = 1
y = np.array(Y)[:, i]
model = OracleLinearRegression(basis=basis)
model.fit(X, y)
model.coef_

# Compare raw y and predicted y
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Raw y")
plt.ylabel("Predicted y")
plt.title("Comparison of Raw y and Predicted y")
plt.tight_layout()
plt.show()

## does the oracle h satiafies the bellman equation?
## approximately yes
## okay if the conditional mean below matches

oracle_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)

discount = equilibrium.discount
current_selector, next_selector = EstimatorForEquilibriumSingle.make_selector(equilibrium=equilibrium)
payoff_covariate = EstimatorForEquilibriumSingle.compute_payoff_covariate(action_state=action_state)
payoff_covariate_current = payoff_covariate[current_selector, :]
action_state_current = np.array(action_state)[current_selector, :]
action_state_next = np.array(action_state)[next_selector, :]
h_next = oracle_h[next_selector, :]
h_current = oracle_h[current_selector, :]

v_1 = h_current
v_2 = payoff_covariate_current + discount * h_next

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(v_1[:, i], v_2[:, i], alpha=0.5)
    min_val = min(v_1[:, i].min(), v_2[:, i].min())
    max_val = max(v_1[:, i].max(), v_2[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()

diff = (v_1 - v_2) / np.abs(np.mean(v_1, axis=0))
np.mean(diff, axis=0)
np.std(diff, axis=0)

# does the conditional mean match?
# perfect match is expected
num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    df = pl.DataFrame({"v_1": v_1[:, i], "v_2": v_2[:, i]}).group_by("v_1").agg(pl.col("v_2").mean())
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(df["v_1"], df["v_2"], alpha=0.5)
    min_val = min(df["v_1"].min(), df["v_2"].min())
    max_val = max(df["v_1"].max(), df["v_2"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()


## does the estimated h satisfy the bellman equation?
## approximately yes
## okay if the conditional mean below matches

h, predictor_list = oracle_avi_estimator.estimate_h(
    initial_h=oracle_h,
    num_iteration=10,
)

h_next = h[next_selector, :]
h_current = h[current_selector, :]

v_1 = h_current
v_2 = payoff_covariate_current + discount * h_next

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(v_1[:, i], v_2[:, i], alpha=0.5)
    min_val = min(v_1[:, i].min(), v_2[:, i].min())
    max_val = max(v_1[:, i].max(), v_2[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()

diff = (v_1 - v_2) / np.abs(np.mean(v_1, axis=0))
np.mean(diff, axis=0)
np.std(diff, axis=0)

# does the conditional mean match?
# perfect match is expected
num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    df = pl.DataFrame({"v_1": v_1[:, i], "v_2": v_2[:, i]}).group_by("v_1").agg(pl.col("v_2").mean())
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(df["v_1"], df["v_2"], alpha=0.5)
    min_val = min(df["v_1"].min(), df["v_2"].min())
    max_val = max(df["v_1"].max(), df["v_2"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()


## does the estimated h converge to the oracle h?
## perfect match is expected

oracle_h_current = oracle_h[current_selector, :]

# Compare h_current and oracle_h_current
num_columns = h_current.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of h_current and oracle_h_current for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(h_current[:, i], oracle_h_current[:, i], alpha=0.5)
    min_val = min(h_current[:, i].min(), oracle_h_current[:, i].min())
    max_val = max(h_current[:, i].max(), oracle_h_current[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("h_current")
    ax.set_ylabel("oracle_h_current")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()

# debug for estimating g ----------------------------------------------

## does the oracle regression perfectly fit the oracle h?
## perfect match is expected

Y, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)

# prepare the data
X = np.array(action_state)

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)

i = 0
y = np.array(Y)[:, i]
model = OracleLinearRegression(basis=basis)
model.fit(X, y)
model.coef_

# Compare raw y and predicted y
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Raw y")
plt.ylabel("Predicted y")
plt.title("Comparison of Raw y and Predicted y")
plt.tight_layout()
plt.show()

## does the oracle g satiafies the bellman equation?
## approximately yes
## okay if the conditional mean below matches

oracle_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)

discount = equilibrium.discount
ccp = EstimatorForEquilibriumSingle.estimate_ccp_count(equilibrium=equilibrium)
e = EstimatorForEquilibriumSingle.compute_conditional_expected_shock_from_result(equilibrium=equilibrium, ccp=ccp)

e_next = e[next_selector, :]
action_state_current = np.array(action_state)[current_selector, :]
action_state_next = np.array(action_state)[next_selector, :]

g_current = oracle_g[current_selector, :]
g_next = oracle_g[next_selector, :]

v_1 = g_current
v_2 = discount * e_next + discount * g_next

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(v_1[:, i], v_2[:, i], alpha=0.5)
    min_val = min(v_1[:, i].min(), v_2[:, i].min())
    max_val = max(v_1[:, i].max(), v_2[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()

diff = (v_1 - v_2) / np.abs(np.mean(v_1, axis=0))
np.mean(diff, axis=0)
np.std(diff, axis=0)

# does the conditional mean match?
# perfect match is expected

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    df = pl.DataFrame({"v_1": v_1[:, i], "v_2": v_2[:, i]}).group_by("v_1").agg(pl.col("v_2").mean())
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(df["v_1"], df["v_2"], alpha=0.5)
    min_val = min(df["v_1"].min(), df["v_2"].min())
    max_val = max(df["v_1"].max(), df["v_2"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()


## does the estimated g satisfy the bellman equation?
## approximately yes
## okay if the conditional mean below matches

g, predictor_list = oracle_avi_estimator.estimate_g(
    initial_g=oracle_g,
    num_iteration=10,
)

g_next = g[next_selector, :]
g_current = g[current_selector, :]

v_1 = g_current
v_2 = discount * e_next + discount * g_next

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(v_1[:, i], v_2[:, i], alpha=0.5)
    min_val = min(v_1[:, i].min(), v_2[:, i].min())
    max_val = max(v_1[:, i].max(), v_2[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()

diff = (v_1 - v_2) / np.abs(np.mean(v_1, axis=0))
np.mean(diff, axis=0)
np.std(diff, axis=0)

# does the conditional mean match?
# perfect match is expected

num_columns = v_1.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of v_1 and v_2 for each column")

for i in range(num_columns):
    df = pl.DataFrame({"v_1": v_1[:, i], "v_2": v_2[:, i]}).group_by("v_1").agg(pl.col("v_2").mean())
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(df["v_1"], df["v_2"], alpha=0.5)
    min_val = min(df["v_1"].min(), df["v_2"].min())
    max_val = max(df["v_1"].max(), df["v_2"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("v_1 (h_current)")
    ax.set_ylabel("v_2 (payoff_covariate_current + discount * h_next)")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()


## does the estimated g converge to the oracle g?
## perfect match is expected

oracle_g_current = oracle_g[current_selector, :]

# Compare h_current and oracle_h_current
num_columns = g_current.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
fig.suptitle("Comparison of h_current and oracle_h_current for each column")

for i in range(num_columns):
    ax = axes[i] if num_columns > 1 else axes
    ax.scatter(g_current[:, i], oracle_g_current[:, i], alpha=0.5)
    min_val = min(g_current[:, i].min(), oracle_g_current[:, i].min())
    max_val = max(g_current[:, i].max(), oracle_g_current[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("h_current")
    ax.set_ylabel("oracle_h_current")
    ax.set_title(f"Column {i}")

plt.tight_layout()
plt.show()
