import pytest
import numpy as np
import torch
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleAVI,
    EstimatorForEquilibriumMultipleSemiGradient,
)
from td_dynamic.karun.multiple.predictor_multiple import (
    LinearRegressionPredictor,
    CCPLogisticRegressionPredictor,
)
from td_dynamic.karun.utils import read_pickle_from_s3

@pytest.fixture
def setup_data():
    bucket_name = "football-markov"
    equilibrium = read_pickle_from_s3(bucket=bucket_name, prefix="output/simulate_multiple/", file_name="equilibrium.pkl")
    
    # predictor_list = [
    #     MLPPredictor(
    #         equilibrium=equilibrium,
    #         predictor_type="oracle",
    #         degree=2,
    #         hidden_layer_sizes=(16,),
    #         learning_rate=0.001,
    #         batch_size=256,
    #         num_epochs=2,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         i=i,
    #     )
    #     for i in range(equilibrium.num["firm"])
    # ]

    predictor_list = [
      LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=0, i=i)
      for i in range(equilibrium.num["firm"])
    ]
    ccp_predictor_list = [
        CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
        for i in range(equilibrium.num["firm"])
    ]   
    avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=predictor_list, ccp_predictor_list=ccp_predictor_list)
    
    oracle_predictor_list = [
        LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=0, i=i)
        for i in range(equilibrium.num["firm"])
    ]
    
    oracle_avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=oracle_predictor_list, ccp_predictor_list=ccp_predictor_list)
    
    semi_gradient_estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
    
    return {
        'equilibrium': equilibrium,
        'avi_estimator': avi_estimator,
        'oracle_avi_estimator': oracle_avi_estimator,
        'semi_gradient_estimator': semi_gradient_estimator
    }

# Tests for h estimation
def test_initialize_h_predictor(setup_data):
    avi_estimator = setup_data['avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="polynomial", degree=2)
    predictor_list_h = avi_estimator.initialize_h_predictor(initial_h=initial_h)
    
    assert len(predictor_list_h) == len(initial_h)
    for i in range(len(initial_h)):
        assert len(predictor_list_h[i]) == initial_h[i].shape[1]

def test_update_h_predictor(setup_data):
    avi_estimator = setup_data['avi_estimator']
    equilibrium = setup_data['equilibrium']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="polynomial", degree=2)
    
    action_state = equilibrium.result.select(
        *[col for col in equilibrium.result.columns if col.startswith("action_value_")],
        *[col for col in equilibrium.result.columns if col.startswith("state_value_")],
    )
    
    payoff_covariate = avi_estimator.compute_payoff_covariate(
        action_state=np.array(action_state), 
        num_firm=equilibrium.num["firm"]
    )
    
    current_selector, next_selector = avi_estimator.make_selector(equilibrium=equilibrium)
    predictor_list_h = avi_estimator.initialize_h_predictor(initial_h=initial_h)
    
    updated_predictor_list = avi_estimator.update_h_predictor(
        predictor_list_h=predictor_list_h,
        action_state=action_state,
        payoff_covariate=payoff_covariate,
        current_selector=current_selector,
        next_selector=next_selector,
        discount=equilibrium.discount,
    )
    
    assert len(updated_predictor_list) == len(predictor_list_h)

def test_estimate_h(setup_data):
    avi_estimator = setup_data['avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="polynomial", degree=2)
    h, h_predictor_list = avi_estimator.estimate_h(initial_h=initial_h, num_iteration=2)
    
    assert len(h) == len(initial_h)
    assert len(h_predictor_list) == len(initial_h)
    for i in range(len(h)):
        assert h[i].shape == initial_h[i].shape

# Tests for g estimation
def test_initialize_g_predictor(setup_data):
    avi_estimator = setup_data['avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="polynomial", degree=2)
    predictor_list_g = avi_estimator.initialize_g_predictor(initial_g=initial_g)
    
    assert len(predictor_list_g) == len(initial_g)
    for i in range(len(initial_g)):
        assert len(predictor_list_g[i]) == initial_g[i].shape[1]

def test_update_g_predictor(setup_data):
    avi_estimator = setup_data['avi_estimator']
    equilibrium = setup_data['equilibrium']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="polynomial", degree=2)
    
    action_state = equilibrium.result.select(
        *[col for col in equilibrium.result.columns if col.startswith("action_value_")],
        *[col for col in equilibrium.result.columns if col.startswith("state_value_")],
    )
    
    ccp = avi_estimator.estimate_ccp(equilibrium=equilibrium, ccp_predictor_list=avi_estimator.ccp_predictor_list)
    e = avi_estimator.compute_conditional_expected_shock_from_result(equilibrium=equilibrium, ccp=ccp)
    
    current_selector, next_selector = avi_estimator.make_selector(equilibrium=equilibrium)
    predictor_list_g = avi_estimator.initialize_g_predictor(initial_g=initial_g)
    
    updated_predictor_list = avi_estimator.update_g_predictor(
        predictor_list_g=predictor_list_g,
        action_state=action_state,
        e=e,
        current_selector=current_selector,
        next_selector=next_selector,
        discount=equilibrium.discount,
    )
    
    assert len(updated_predictor_list) == len(predictor_list_g)

def test_estimate_g(setup_data):
    avi_estimator = setup_data['avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    initial_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="polynomial", degree=2)
    g, g_predictor_list = avi_estimator.estimate_g(initial_g=initial_g, num_iteration=2)
    
    assert len(g) == len(initial_g)
    assert len(g_predictor_list) == len(initial_g)
    for i in range(len(g)):
        assert g[i].shape == initial_g[i].shape

# Tests for oracle estimation
def test_oracle_h_estimation(setup_data):
    oracle_avi_estimator = setup_data['oracle_avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    oracle_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)
    h, _ = oracle_avi_estimator.estimate_h(initial_h=oracle_h, num_iteration=2)
    
    for i in range(len(h)):
        np.testing.assert_array_almost_equal(h[i], oracle_h[i], decimal=5)

def test_oracle_g_estimation(setup_data):
    oracle_avi_estimator = setup_data['oracle_avi_estimator']
    semi_gradient_estimator = setup_data['semi_gradient_estimator']
    
    oracle_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)
    g, _ = oracle_avi_estimator.estimate_g(initial_g=oracle_g, num_iteration=2)
    
    for i in range(len(g)):
        np.testing.assert_array_almost_equal(g[i], oracle_g[i], decimal=5)
