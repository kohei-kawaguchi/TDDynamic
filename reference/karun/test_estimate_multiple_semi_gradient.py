# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleSemiGradient,
    EstimatorForEquilibriumMultiple,
)
from td_dynamic.karun.multiple.predictor_multiple import (
    OracleStateValue,
    CCPLogisticRegressionPredictor,
)
from td_dynamic.karun.utils import read_pickle_from_local
from td_dynamic.karun.utils import br

# set constants ----------------------------------------------------

degree = 1
predictor_type = "polynomial"

# load data -------------------------------------------------------

equilibrium = read_pickle_from_local("output/simulate_multiple/equilibrium.pkl")

# define estimator ------------------------------------------------

i = 0

action_state = equilibrium.action_state["action_state_value"]
state_value = np.unique(action_state[:, slice(equilibrium.num["firm"], action_state.shape[1])], axis=0)
basis = OracleStateValue(state_value=state_value)
basis.transform(state_value)

y = equilibrium.result.select(f"action_value_{i}").to_numpy().flatten()
X = equilibrium.result.select(*[col for col in equilibrium.result.columns if col.startswith("state_value_")]).to_numpy()

ccp_predictor = CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="polynomial", degree=1)
ccp_predictor.fit(X, y)
ccp_predictor.predict(X)
ccp_predictor.score(X, y)

ccp_predictor = CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
ccp_predictor.fit(X, y)
ccp_predictor.predict(X)
ccp_predictor.score(X, y)

# estimate ccp by logstic
ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
    for i in range(equilibrium.num["firm"])
]

estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)

# check data -------------------------------------------------------

## check the variation of the state --------------------------------

check = equilibrium.result.filter(
    (pl.col("t") != 0) & (pl.col("t") != equilibrium.num["period"] - 1)
)
check = check.select(
    *[col for col in check.columns if col.startswith("action_value_")],
    *[col for col in check.columns if col.startswith("state_value_")],
)
[
    np.unique(check[:, [i, *range(equilibrium.num["firm"], check.shape[1])]], axis=0).shape
    for i in range(equilibrium.num["firm"])
]

## compare the ccp joint --------------------------------------------

ccp_joint_empirical = equilibrium.result.group_by([*[col for col in equilibrium.result.columns if col.startswith("action_index_")], *[col for col in equilibrium.result.columns if col.startswith("state_index_")]]).agg(count=pl.col("action_index_0").count()).with_columns(ccp_empirical=pl.col("count") / pl.col("count").sum().over([col for col in equilibrium.result.columns if col.startswith("state_index_")])).sort([col for col in equilibrium.result.columns if col.startswith("state_index_")][::-1] + [col for col in equilibrium.result.columns if col.startswith("action_index_")])

ccp_joint_value, ccp_joint_index = equilibrium.compute_ccp_joint(ccp_marginal_value=equilibrium.value["ccp_marginal_value"])
ccp_joint_theoretical = ccp_joint_index.with_columns(ccp_theoretical=ccp_joint_value.flatten()).sort([col for col in ccp_joint_index.columns if col.startswith("state_")][::-1] + [col for col in ccp_joint_index.columns if col.startswith("action_")])

comparison = ccp_joint_empirical.join(ccp_joint_theoretical, on=[col for col in ccp_joint_empirical.columns if col.startswith("state_")][::-1] + [col for col in ccp_joint_empirical.columns if col.startswith("action_")])

# Create scatter plot comparing empirical vs theoretical CCPs
plt.figure(figsize=(8, 6))
plt.scatter(comparison["ccp_theoretical"], comparison["ccp_empirical"], alpha=0.5)
plt.plot([0, comparison["ccp_theoretical"].max()], [0, comparison["ccp_theoretical"].max()], 'k--') # 45 degree line
plt.xlabel("Theoretical CCP")
plt.ylabel("Empirical CCP")
plt.title("Comparison of Empirical vs Theoretical CCPs")
plt.grid(True)
plt.show()


# estimate by a semi gradient method -------------------------------

## estimate h: choice-specific payoff value ------------------------

# construct basis functions

basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type=predictor_type, degree=degree
)

basis

basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)

basis

current_selector, next_selector = EstimatorForEquilibriumMultiple.make_selector(equilibrium=equilibrium)

basis_action_state = EstimatorForEquilibriumMultipleSemiGradient.transform_basis_action_state(
    result=equilibrium.result, basis=basis
)

basis_action_state

# construct payoff covariate

result = equilibrium.result
action_state = result.select(
    *[col for col in result.columns if col.startswith("action_value_")],
    *[col for col in result.columns if col.startswith("state_value_")],
)
action_state = np.array(action_state)

payoff_covariate = EstimatorForEquilibriumMultiple.compute_payoff_covariate(action_state=action_state, num_firm=equilibrium.num["firm"])

payoff_covariate = EstimatorForEquilibriumMultipleSemiGradient.compute_payoff_covariate_from_result(
    equilibrium=equilibrium
)

# estimate omega

omega = EstimatorForEquilibriumMultipleSemiGradient.estimate_omega(
    payoff_covariate=payoff_covariate,
    basis_action_state=basis_action_state,
    current_selector=current_selector,
    next_selector=next_selector,
    discount=equilibrium.discount,
)

# estimate h: choice-specific payoff value

h, _, _ = estimator.estimate_h(predictor_type=predictor_type, degree=degree)

h, omega

## estimate g: choice-specific conditional error value ---------------

# estimate ccp by count

ccp = EstimatorForEquilibriumMultiple.estimate_ccp_count(equilibrium=equilibrium)

ccp

# Create polynomial features for state values
ccp_theoretical = [
    equilibrium.value["ccp_marginal_index"][i]
    .with_columns(ccp_theoretical=equilibrium.value["ccp_marginal_value"][i].flatten())
    for i in range(equilibrium.num["firm"])
]

comparison = [
    ccp[i].join(ccp_theoretical[i], on=[col for col in ccp[i].columns if col.startswith(("action_index_", "state_index_"))], how="left")
    for i in range(equilibrium.num["firm"])
]

# Create a scatter plot of theoretical vs empirical CCPs
# perfect match is expecte
for i in range(equilibrium.num["firm"]):
    plt.figure(figsize=(10, 10))
    plt.scatter(comparison[i]["ccp_theoretical"], comparison[i]["ccp"], alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--") 
    plt.xlabel("Theoretical CCP")
    plt.ylabel("Empirical CCP") 
    plt.title(f"Theoretical vs Empirical Conditional Choice Probabilities for Firm {i}")
    plt.tight_layout()
    plt.show()

ccp_logistic = EstimatorForEquilibriumMultiple.estimate_ccp(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)

for i in range(equilibrium.num["firm"]):
    y = equilibrium.result.select(f"action_value_{i}").to_numpy().flatten()
    X = equilibrium.result.select(*[col for col in equilibrium.result.columns if col.startswith("state_value_")]).to_numpy()

    # compare with the count
    ccp_count = equilibrium.result.join(
        ccp[i],
        on=[col for col in equilibrium.result.columns if col.startswith("state_index_")] + [f"action_index_{i}"],
        how="left",
    )
    ccp_predicted = ccp_predictor_list[i].predict(X)
    row_indices = np.arange(ccp_predicted.shape[0])
    ccp_predicted_selected = ccp_predicted[row_indices, y.astype(int)]
    ccp_count = ccp_count.with_columns(ccp_predicted=ccp_predicted_selected)

    # Create scatter plot comparing empirical CCPs vs logistic regression predicted CCPs
    plt.figure(figsize=(10, 10))
    plt.scatter(ccp_count["ccp"], ccp_count["ccp_predicted"], alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--", label="45° line")  # Add 45 degree reference line

    plt.xlabel("Count CCP")
    plt.ylabel("Logistic CCP")
    plt.title(f"Count vs Logistic CCP for Firm {i}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# compute conditional expected shock from result

e = EstimatorForEquilibriumMultiple.compute_conditional_expected_shock_from_result(equilibrium=equilibrium, ccp=ccp_logistic)

e

# estimate xi

xi = EstimatorForEquilibriumMultipleSemiGradient.estimate_xi(
    e=e,
    basis_action_state=basis_action_state,
    current_selector=current_selector,
    next_selector=next_selector,
    discount=equilibrium.discount,
)

# estimate g

g, _, _ = estimator.estimate_g(predictor_type=predictor_type, degree=degree)

g

## test functions ----------------------------------------------------

# calculate the oracle estimated value

basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)

basis_action_state = EstimatorForEquilibriumMultipleSemiGradient.transform_basis_action_state(
    result=equilibrium.result, basis=basis
)

ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
    for i in range(equilibrium.num["firm"])
]

oracle_h, _, _ = estimator.estimate_h(predictor_type="oracle", degree=0)

oracle_g, _, _ = estimator.estimate_g(predictor_type="oracle", degree=0)

for i in range(equilibrium.num["firm"]):
    # calculate the estimated value
    estimated_value = (
        np.dot(
            h[i],
            [
                equilibrium.param["param_payoff"]["beta"],
                equilibrium.param["param_payoff"]["alpha"],
                equilibrium.param["param_payoff"]["lambda"],
            ],
        )
        + g[i].flatten()
    )

    # obtain the true value
    choice_marginal_value = equilibrium.value["choice_marginal_value"][i]

    true_value = (
        pl.concat(
            [
                equilibrium.value["choice_marginal_index"][i],
                pl.DataFrame({"true": choice_marginal_value.flatten()}),
            ],
            how="horizontal",
        )
    )

    oracle_value = (
        np.dot(
            oracle_h[i],
            [
                equilibrium.param["param_payoff"]["beta"],
                equilibrium.param["param_payoff"]["alpha"],
                equilibrium.param["param_payoff"]["lambda"],
            ],
        )
        + oracle_g[i].flatten()
    )

    # compare true value with estimated values

    comparison = (
        equilibrium.result
        .join(
            true_value,
            on=[f"action_index_{i}"] + [col for col in true_value.columns if col.startswith("state_")],
        )
        .with_columns(
            pl.Series(name="estimate", values=estimated_value),
            pl.Series(name="oracle", values=oracle_value),
        )
    )

    # perfect match is not expected
    plt.scatter(comparison["true"], comparison["estimate"])
    plt.xlabel("True Values")
    plt.ylabel("Estimated Values")
    plt.title(f"Scatter Plot between True and Estimated Values for Firm {i}")
    plt.plot(
        [comparison["true"].min(), comparison["true"].max()],
        [comparison["true"].min(), comparison["true"].max()],
        "k--",
        lw=1,
    )
    plt.show()

    # perfect match is expected
    plt.scatter(comparison["true"], comparison["oracle"])
    plt.xlabel("True Values")
    plt.ylabel("OracleActionState Values")
    plt.title(f"Scatter Plot between True and OracleActionState Values for Firm {i}")
    plt.plot(
        [comparison["true"].min(), comparison["true"].max()],
        [comparison["true"].min(), comparison["true"].max()],
        "k--",
        lw=1,
    )
    plt.show()

