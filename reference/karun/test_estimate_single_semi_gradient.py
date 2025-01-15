# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.single.estimate_single import (
    EstimatorForEquilibriumSingleSemiGradient,
    EstimatorForEquilibriumSingle,
)
from td_dynamic.karun.single.estimate_single import EstimatorForEquilibriumSingleSemiGradient

from td_dynamic.karun.utils import read_pickle_from_s3

# set constants ----------------------------------------------------

degree = 2
predictor_type = "polynomial"

prefix = "output/estimate_single_semi_gradient/"
bucket_name = "football-markov"

# load data
equilibrium = read_pickle_from_s3(bucket=bucket_name, prefix="output/simulate_single/", file_name="equilibrium.pkl")
estimator = EstimatorForEquilibriumSingleSemiGradient(equilibrium=equilibrium)
result = estimator.estimate_params(predictor_type=predictor_type, degree=degree)


# estimate by a semi gradient method -------------------------------

## estimate h: choice-specific payoff value ------------------------

# construct basis functions

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type=predictor_type, degree=degree
)

basis

current_selector, next_selector = EstimatorForEquilibriumSingle.make_selector(equilibrium=equilibrium)

basis_action_state = EstimatorForEquilibriumSingleSemiGradient.transform_basis_action_state(
    result=equilibrium.result, basis=basis
)

basis_action_state

# construct payoff covariate

payoff_covariate = EstimatorForEquilibriumSingleSemiGradient.compute_payoff_covariate_from_result(
    equilibrium=equilibrium
)

# estimate omega

omega = EstimatorForEquilibriumSingleSemiGradient.estimate_omega(
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

# estimate ccp

ccp = EstimatorForEquilibriumSingle.estimate_ccp_count(equilibrium=equilibrium)

ccp

# compute conditional expected shock from result

e = EstimatorForEquilibriumSingle.compute_conditional_expected_shock_from_result(equilibrium=equilibrium, ccp=ccp)

e

# estimate xi

xi = EstimatorForEquilibriumSingleSemiGradient.estimate_xi(
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

# calculate the estimated value

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

# calculate the oracle estimated value

basis = EstimatorForEquilibriumSingleSemiGradient.fit_basis_action_state(
    equilibrium=equilibrium, predictor_type="oracle", degree=0
)

basis

basis_action_state = EstimatorForEquilibriumSingleSemiGradient.transform_basis_action_state(
    result=equilibrium.result, basis=basis
)

basis_action_state

oracle_h, _, _ = estimator.estimate_h(predictor_type="oracle", degree=0)

oracle_g, _, _ = estimator.estimate_g(predictor_type="oracle", degree=0)

oracle_value = (
    np.dot(
        oracle_h,
        [
            equilibrium.param["param_payoff"]["beta"],
            equilibrium.param["param_payoff"]["lambda"],
        ],
    )
    + oracle_g.flatten()
)

# compare true value with estimated values

comparison = (
    equilibrium.result.with_columns(pl.col("action_index").cast(pl.Int32))
    .join(
        true_value,
        on=["action_index"] + [col for col in true_value.columns if col.startswith("state_")],
    )
    .with_columns(
        pl.Series(name="estimate", values=estimated_value),
        pl.Series(name="oracle", values=oracle_value),
    )
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

# perfect match is expected
plt.scatter(comparison["true"], comparison["oracle"])
plt.xlabel("True Values")
plt.ylabel("OracleActionState Values")
plt.title("Scatter Plot between True and OracleActionState Values")
plt.plot(
    [comparison["true"].min(), comparison["true"].max()],
    [comparison["true"].min(), comparison["true"].max()],
    "k--",
    lw=1,
)
plt.show()
