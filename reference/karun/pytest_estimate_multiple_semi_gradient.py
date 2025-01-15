import pytest
import numpy as np
import polars as pl
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleSemiGradient,
    EstimatorForEquilibriumMultiple,
)
from td_dynamic.karun.utils import read_pickle_from_local
from td_dynamic.karun.multiple.predictor_multiple import CCPLogisticRegressionPredictor

@pytest.fixture
def setup_data():
    equilibrium = read_pickle_from_local("output/simulate_multiple/equilibrium.pkl")
    ccp_predictor_list = [
        CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
        for i in range(equilibrium.num["firm"])
    ]
    estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
    degree = 1
    predictor_type = "polynomial"
    return equilibrium, estimator, degree, predictor_type

def test_state_variation(setup_data):
    equilibrium, *_ = setup_data
    check = equilibrium.result.filter(
        (pl.col("t") != 0) & (pl.col("t") != equilibrium.num["period"] - 1)
    )
    check = check.select(
        *[col for col in check.columns if col.startswith("action_value_")],
        *[col for col in check.columns if col.startswith("state_value_")],
    )
    variations = [
        np.unique(check[:, [i, *range(equilibrium.num["firm"], check.shape[1])]], axis=0).shape
        for i in range(equilibrium.num["firm"])
    ]
    assert all(v[0] > 1 for v in variations), "Insufficient state variation"

def test_ccp_joint_comparison(setup_data):
    equilibrium, *_ = setup_data
    
    ccp_joint_empirical = equilibrium.result.group_by(
        [*[col for col in equilibrium.result.columns if col.startswith("action_index_")], 
         *[col for col in equilibrium.result.columns if col.startswith("state_index_")]]
    ).agg(count=pl.col("action_index_0").count()).with_columns(
        ccp_empirical=pl.col("count") / pl.col("count").sum().over(
            [col for col in equilibrium.result.columns if col.startswith("state_index_")]
        )
    )
    
    ccp_joint_value, ccp_joint_index = equilibrium.compute_ccp_joint(
        ccp_marginal_value=equilibrium.value["ccp_marginal_value"]
    )
    
    comparison = ccp_joint_empirical.join(
        ccp_joint_index.with_columns(ccp_theoretical=ccp_joint_value.flatten()),
        on=[col for col in ccp_joint_empirical.columns if col.startswith(("action_index_", "state_index_"))]
    )
    
    correlation = np.corrcoef(comparison["ccp_theoretical"], comparison["ccp_empirical"])[0,1]
    assert correlation > 0.9, f"CCP joint comparison correlation {correlation} is too low"

def test_basis_estimation(setup_data):
    equilibrium, _, degree, predictor_type = setup_data
    
    basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
        equilibrium=equilibrium, predictor_type=predictor_type, degree=degree
    )
    assert basis is not None, "Basis estimation failed"
    
    basis_action_state = EstimatorForEquilibriumMultipleSemiGradient.transform_basis_action_state(
        result=equilibrium.result, basis=basis
    )
    assert basis_action_state is not None, "Basis transformation failed"

def test_payoff_covariate(setup_data):
    equilibrium, *_ = setup_data
    
    payoff_covariate = EstimatorForEquilibriumMultipleSemiGradient.compute_payoff_covariate_from_result(
        equilibrium=equilibrium
    )
    assert payoff_covariate is not None, "Payoff covariate computation failed"

def test_omega_estimation(setup_data):
    equilibrium, _, degree, predictor_type = setup_data
    
    basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
        equilibrium=equilibrium, predictor_type=predictor_type, degree=degree
    )
    
    basis_action_state = EstimatorForEquilibriumMultipleSemiGradient.transform_basis_action_state(
        result=equilibrium.result, basis=basis
    )
    
    payoff_covariate = EstimatorForEquilibriumMultipleSemiGradient.compute_payoff_covariate_from_result(
        equilibrium=equilibrium
    )

    current_selector, next_selector = EstimatorForEquilibriumMultiple.make_selector(equilibrium=equilibrium)
    
    omega = EstimatorForEquilibriumMultipleSemiGradient.estimate_omega(
        payoff_covariate=payoff_covariate,
        basis_action_state=basis_action_state,
        current_selector=current_selector,
        next_selector=next_selector,
        discount=equilibrium.discount,
    )
    
    assert omega is not None, "Omega estimation failed"
    assert isinstance(omega[0], np.ndarray), "Omega elements should be numpy arrays"
    assert not np.isnan(omega).any(), "Omega contains NaN values"


def test_h_estimation(setup_data):
    equilibrium, estimator, degree, predictor_type = setup_data
    
    h, h_covariate, h_basis = estimator.estimate_h(predictor_type=predictor_type, degree=degree)
    assert h_covariate is not None, "H covariate computation failed"
    assert h_basis is not None, "H basis computation failed"
    assert all(h_i is not None for h_i in h), "H estimation failed"

def test_ccp_estimation(setup_data):
    equilibrium, estimator, degree, predictor_type = setup_data

    # Test CCP count estimation
    ccp = EstimatorForEquilibriumMultiple.estimate_ccp_count(equilibrium=equilibrium)
    assert ccp is not None, "CCP count estimation failed"
    assert len(ccp) == equilibrium.num["firm"], "CCP count length mismatch"

    # Test theoretical CCP computation
    ccp_theoretical = [
        equilibrium.value["ccp_marginal_index"][i]
        .with_columns(ccp_theoretical=equilibrium.value["ccp_marginal_value"][i].flatten())
        for i in range(equilibrium.num["firm"])
    ]
    assert all(ct is not None for ct in ccp_theoretical), "Theoretical CCP computation failed"

    # Test CCP comparison
    comparison = [
        ccp[i].join(ccp_theoretical[i], on=[col for col in ccp[i].columns if col.startswith(("action_index_", "state_index_"))], how="left")
        for i in range(equilibrium.num["firm"])
    ]
    assert all(comp is not None for comp in comparison), "CCP comparison failed"
    
    # Test correlation between theoretical and empirical CCPs
    for i in range(equilibrium.num["firm"]):
        correlation = np.corrcoef(comparison[i]["ccp_theoretical"], comparison[i]["ccp"])[0,1]
        assert correlation > 0.9, f"CCP correlation {correlation} for firm {i} is too low"

    # Test logistic CCP estimation
    ccp_logistic = EstimatorForEquilibriumMultiple.estimate_ccp(equilibrium=equilibrium, ccp_predictor_list=estimator.ccp_predictor_list)
    assert ccp_logistic is not None, "Logistic CCP estimation failed"

    # Test logistic CCP predictions
    for i in range(equilibrium.num["firm"]):
        y = equilibrium.result.select(f"action_value_{i}").to_numpy().flatten()
        X = equilibrium.result.select(*[col for col in equilibrium.result.columns if col.startswith("state_value_")]).to_numpy()

        ccp_count = equilibrium.result.join(
            ccp[i],
            on=[col for col in equilibrium.result.columns if col.startswith("state_index_")] + [f"action_index_{i}"],
            how="left",
        )
        ccp_predicted = estimator.ccp_predictor_list[i].predict(X)
        row_indices = np.arange(ccp_predicted.shape[0])
        ccp_predicted_selected = ccp_predicted[row_indices, y.astype(int)]
        ccp_count = ccp_count.with_columns(ccp_predicted=ccp_predicted_selected)

        # Test correlation between count and logistic CCPs
        correlation = np.corrcoef(ccp_count["ccp"], ccp_count["ccp_predicted"])[0,1]
        assert correlation > 0.9, f"Logistic vs count CCP correlation {correlation} for firm {i} is too low"


def test_conditional_expected_shock(setup_data):
    equilibrium, estimator, *_ = setup_data
    
    ccp = EstimatorForEquilibriumMultiple.estimate_ccp(equilibrium=equilibrium, ccp_predictor_list=estimator.ccp_predictor_list)
    e = EstimatorForEquilibriumMultiple.compute_conditional_expected_shock_from_result(
        equilibrium=equilibrium, ccp=ccp
    )
    
    assert e is not None, "Conditional expected shock computation failed"
    assert isinstance(e, list), "Expected shock should be a list"
    assert len(e) == equilibrium.num["firm"], "Expected shock list length mismatch"
    assert all(not np.isnan(e_i).any() for e_i in e), "Expected shock contains NaN values"

def test_xi_estimation(setup_data):
    equilibrium, estimator, degree, predictor_type = setup_data
    
    # Get required inputs
    ccp = EstimatorForEquilibriumMultiple.estimate_ccp(equilibrium=equilibrium, ccp_predictor_list=estimator.ccp_predictor_list)
    e = EstimatorForEquilibriumMultiple.compute_conditional_expected_shock_from_result(
        equilibrium=equilibrium, ccp=ccp
    )
    
    basis = EstimatorForEquilibriumMultipleSemiGradient.fit_basis_action_state(
        equilibrium=equilibrium, predictor_type=predictor_type, degree=degree
    )
    basis_action_state = EstimatorForEquilibriumMultipleSemiGradient.transform_basis_action_state(
        result=equilibrium.result, basis=basis
    )

    current_selector, next_selector = EstimatorForEquilibriumMultiple.make_selector(equilibrium=equilibrium)
    
    xi = EstimatorForEquilibriumMultipleSemiGradient.estimate_xi(
        e=e,
        basis_action_state=basis_action_state,
        current_selector=current_selector,
        next_selector=next_selector,
        discount=equilibrium.discount,
    )
    
    assert xi is not None, "Xi estimation failed"
    assert isinstance(xi[0], np.ndarray), "Xi elements should be numpy arrays"
    assert not np.isnan(xi).any(), "Xi contains NaN values"

def test_g_estimation(setup_data):
    equilibrium, estimator, degree, predictor_type = setup_data
    
    g, _, _ = estimator.estimate_g(predictor_type=predictor_type, degree=degree)
    
    assert g is not None, "G estimation failed"
    assert isinstance(g, list), "G should be a list"
    assert len(g) == equilibrium.num["firm"], "G list length mismatch"
    assert all(isinstance(g_i, np.ndarray) for g_i in g), "G elements should be numpy arrays"
    assert all(not np.isnan(g_i).any() for g_i in g), "G contains NaN values"


def test_value_function_estimation(setup_data):
    equilibrium, estimator, *_ = setup_data
    
    oracle_h, _, _ = estimator.estimate_h(predictor_type="oracle", degree=0)
    oracle_g, _, _ = estimator.estimate_g(predictor_type="oracle", degree=0)
    
    for i in range(equilibrium.num["firm"]):
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

        comparison = (
            equilibrium.result
            .join(
                true_value,
                on=[f"action_index_{i}"] + [col for col in true_value.columns if col.startswith("state_")],
            )
            .with_columns(
                pl.Series(name="oracle", values=oracle_value),
            )
        )
            
        correlation = np.corrcoef(comparison["true"], comparison["oracle"])[0,1]
        assert correlation > 0.95, f"Value function estimation correlation for firm {i} is too low"
