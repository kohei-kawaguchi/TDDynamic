import pytest
import numpy as np
import torch
import polars as pl
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleAVI,
    EstimatorForEquilibriumMultipleSemiGradient,
    MLEModel,
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
    
    predictor_list = [
        LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=0, i=i)
        for i in range(equilibrium.num["firm"])
    ]
    ccp_predictor_list = [
        CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
        for i in range(equilibrium.num["firm"])
    ]   
    avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=predictor_list, ccp_predictor_list=ccp_predictor_list)
    semi_gradient_estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
    
    action_set = avi_estimator.make_action_set(
        result=equilibrium.result, 
        num_action=equilibrium.num["action"], 
        num_firm=equilibrium.num["firm"]
    )
    
    h_semi, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)
    g_semi, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)
    h_all_semi, g_all_semi = semi_gradient_estimator.estimate_h_g_all()
    
    num_iteration = 1
    h, predictor_list_h = avi_estimator.estimate_h(initial_h=h_semi, num_iteration=num_iteration)
    g, predictor_list_g = avi_estimator.estimate_g(initial_g=g_semi, num_iteration=num_iteration)
    h_all, g_all = avi_estimator.estimate_h_g_all(predictor_list_h=predictor_list_h, predictor_list_g=predictor_list_g)
    
    x = np.array([
        avi_estimator.equilibrium.param["param_payoff"]["beta"],
        avi_estimator.equilibrium.param["param_payoff"]["alpha"], 
        avi_estimator.equilibrium.param["param_payoff"]["lambda"],
    ])
    
    mle_model = MLEModel(
        endog=np.random.normal(size=equilibrium.result.shape[0]),
        h=h,
        g=g,
        h_all=h_all,
        g_all=g_all,
        result=equilibrium.result,
        action_set=action_set
    )
    
    mle_model_semi = MLEModel(
        endog=np.random.normal(size=equilibrium.result.shape[0]),
        h=h_semi,
        g=g_semi,
        h_all=h_all_semi,
        g_all=g_all_semi,
        result=equilibrium.result,
        action_set=action_set
    )
    
    return {
        'equilibrium': equilibrium,
        'avi_estimator': avi_estimator,
        'semi_gradient_estimator': semi_gradient_estimator,
        'mle_model': mle_model,
        'mle_model_semi': mle_model_semi,
        'x': x,
        'h': h,
        'g': g,
        'h_semi': h_semi,
        'g_semi': g_semi
    }

def test_make_action_set(setup_data):
    avi_estimator = setup_data['avi_estimator']
    equilibrium = setup_data['equilibrium']
    
    action_set = avi_estimator.make_action_set(
        result=equilibrium.result,
        num_action=equilibrium.num["action"],
        num_firm=equilibrium.num["firm"]
    )
    assert isinstance(action_set, list)
    assert len(action_set) == equilibrium.num["firm"]
    assert all(isinstance(a, pl.DataFrame) for a in action_set)

def test_expand_result_action(setup_data):
    avi_estimator = setup_data['avi_estimator']
    equilibrium = setup_data['equilibrium']
    
    df = avi_estimator.expand_result_action(
        result=equilibrium.result,
        num_action=equilibrium.num["action"],
        num_firm=equilibrium.num["firm"]
    )
    assert isinstance(df, list)
    assert all(isinstance(a, pl.DataFrame) for a in df)

def test_compute_denominator(setup_data):
    denominator, benchmark = setup_data['mle_model'].compute_denominator(x=setup_data['x'])
    assert isinstance(denominator[0], pl.DataFrame)
    assert isinstance(benchmark, list)
    assert all(np.all(np.isfinite(d.to_numpy())) for d in denominator)
    assert all(np.isfinite(benchmark))

def test_compute_numerator(setup_data):
    denominator, benchmark = setup_data['mle_model'].compute_denominator(x=setup_data['x'])
    numerator = setup_data['mle_model'].compute_numerator(x=setup_data['x'], benchmark=benchmark)
    assert isinstance(numerator[0], pl.DataFrame)
    assert all(np.all(np.isfinite(n.to_numpy())) for n in numerator)

def test_compute_likelihood(setup_data):
    denominator, benchmark = setup_data['mle_model'].compute_denominator(x=setup_data['x'])
    numerator = setup_data['mle_model'].compute_numerator(x=setup_data['x'], benchmark=benchmark)
    likelihood = setup_data['mle_model'].compute_likelihood_individual(numerator=numerator, denominator=denominator)
    assert isinstance(likelihood[0], pl.DataFrame)
    assert all(np.all(np.isfinite(l.to_numpy())) for l in likelihood)

def test_compute_loglikelihood(setup_data):
    loglikelihood = setup_data['mle_model'].compute_loglikelihood(params=setup_data['x'])
    assert isinstance(loglikelihood, float)

def test_mle_loglikelihood_maximum(setup_data):
    true_params = setup_data['x']
    mle_model = setup_data['mle_model']
    
    # Evaluate loglikelihood at true parameters
    ll_true = mle_model.compute_loglikelihood(params=true_params)
    
    # For each parameter, evaluate at ±0.05 and verify lower loglikelihood
    for i in range(len(true_params)):
        params_minus = true_params.copy()
        params_plus = true_params.copy()
        
        params_minus[i] -= 0.05  # -0.05
        params_plus[i] += 0.05   # +0.05
        
        ll_minus = mle_model.compute_loglikelihood(params=params_minus)
        ll_plus = mle_model.compute_loglikelihood(params=params_plus)
        
        assert ll_minus < ll_true, f"Loglikelihood at -0.05 of param {i} should be lower"
        assert ll_plus < ll_true, f"Loglikelihood at +0.05 of param {i} should be lower"

def test_semi_gradient_loglikelihood_maximum(setup_data):
    true_params = setup_data['x']
    mle_model_semi = setup_data['mle_model_semi']
    
    # Evaluate loglikelihood at true parameters
    ll_true = mle_model_semi.compute_loglikelihood(params=true_params)
    
    # For each parameter, evaluate at ±0.05 and verify lower loglikelihood
    for i in range(len(true_params)):
        params_minus = true_params.copy()
        params_plus = true_params.copy()
        
        params_minus[i] -= 0.05  # -0.05
        params_plus[i] += 0.05   # +0.05
        
        ll_minus = mle_model_semi.compute_loglikelihood(params=params_minus)
        ll_plus = mle_model_semi.compute_loglikelihood(params=params_plus)
        
        assert ll_minus < ll_true, f"Loglikelihood at -0.05 of param {i} should be lower"
        assert ll_plus < ll_true, f"Loglikelihood at +0.05 of param {i} should be lower"

