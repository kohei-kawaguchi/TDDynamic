import pytest
import numpy as np
import polars as pl

from td_dynamic.karun.multiple.simulate_multiple import EquilibriumMultipleEntryExit

@pytest.fixture
def model():
    num = {
        "market": 1000, 
        "firm": 3, 
        "period": 10, 
        "state": 1, 
        "action": 2, 
        "tolerance": 1e-6
    }
    num["grid_size"] = np.full(num["state"], 5)

    param = {
        "param_payoff": {"beta": 0.7, "alpha": 0.5, "lambda": 0.5},
        "param_state": {
            "state_constant": np.concatenate((np.full(num["state"], 0.0),)),
            "state_trans": np.eye(num["state"]) * 0.1,
            "state_sd": np.eye(num["state"]),
        },
    }

    discount = 0.95

    model = EquilibriumMultipleEntryExit(discount=discount, num=num, param=param)
    return model

def test_discretize_state(model):
    model.discretize_state()
    
    assert model.markov.P is not None
    assert model.markov.state_values is not None
    assert np.allclose(model.markov.P.sum(axis=1), 1.0)

def test_state_mapping(model):
    model.discretize_state()
    model.compute_state_value()
    
    assert model.state_raw["state_value"] is not None
    assert model.state_raw["state_index"] is not None
    
    check = model.state_raw["state_index"].sort(
        model.state_raw["state_index"].columns[::-1]
    )
    assert check.equals(model.state_raw["state_index"])

def test_action_state_mapping(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    
    assert model.action_state["action_state_value"] is not None
    assert model.action_state["action_state_index"] is not None
    
    check = model.action_state["action_state_index"].sort(
        model.action_state["action_state_index"].columns[::-1]
    )
    assert check.equals(model.action_state["action_state_index"])

def test_payoff_computation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    
    assert model.covariate is not None
    assert model.theta is not None
    assert model.payoff["payoff_value"] is not None
    assert model.payoff["payoff_index"] is not None

def test_transition_computation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    
    assert model.transition["transition_probability"] is not None
    assert model.transition["transition_row_index"] is not None
    assert model.transition["transition_column_index"] is not None

def test_value_function_computation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    
    ccp_joint_value, ccp_joint_index = model.compute_ccp_joint(
        ccp_marginal_value=model.value["ccp_marginal_value"]
    )
    e_value, e_index = model.compute_conditional_expected_shock(
        ccp_marginal_value=model.value["ccp_marginal_value"]
    )
    
    assert ccp_joint_value is not None
    assert ccp_joint_index is not None
    assert e_value is not None
    assert e_index is not None

def test_choice_probability_computation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.compute_exante_value(ccp_marginal_value=model.value["ccp_marginal_value"])
    model.compute_choice_joint_value(exante_value=model.value["exante_value"])
    model.compute_choice_marginal_value(
        choice_joint_value=model.value["choice_joint_value"], 
        ccp_marginal_value=model.value["ccp_marginal_value"]
    )
    model.compute_ccp_marginal(choice_marginal_value=model.value["choice_marginal_value"])
    
    assert model.value["ccp_marginal_value"] is not None

def test_dynamic_problem_solution(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.solve_dynamic_problem()
    
    assert model.value["ccp_marginal_value"] is not None

def test_action_simulation(model):
    num_simulation = 1000
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.solve_dynamic_problem()
    
    np.random.seed(1)
    state_mt = 0
    
    ccp_joint_value, ccp_joint_index = model.compute_ccp_joint(
        ccp_marginal_value=model.value["ccp_marginal_value"]
    )
    num_action_profile = ccp_joint_index.select(pl.col("^action_.*$")).n_unique()
    action_count = np.zeros(num_action_profile)
    
    for _ in range(num_simulation):
        action_mt = model.simulate_action_mt(state_mt=state_mt)
        action_count[action_mt] += 1
        
    empirical_probability = action_count / num_simulation
    true_probability = np.array(ccp_joint_value[
        num_action_profile * state_mt : num_action_profile * (state_mt + 1)
    ]).flatten()
    
    assert np.allclose(empirical_probability, true_probability, atol=0.03)

def test_state_simulation(model):
    num_simulation = 1000
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.solve_dynamic_problem()
    
    np.random.seed(1)
    state_mt = 0
    action_mt = 1
    
    ccp_joint_value, ccp_joint_index = model.compute_ccp_joint(
        ccp_marginal_value=model.value["ccp_marginal_value"]
    )
    num_action_profile = ccp_joint_index.select(pl.col("^action_.*$")).n_unique()
    num_state = model.payoff["payoff_index"].select(pl.col("^state_.*$")).n_unique()
    state_count = np.zeros(num_state)
    
    for _ in range(num_simulation):
        state_mt = 0
        for _ in range(model.num["period"]):
            state_mt1 = model.simulate_state_mt(action_mt=action_mt, state_mt=state_mt)
        state_count[state_mt1] += 1
        
    empirical_probability = state_count / num_simulation
    true_probability = np.array(model.transition["transition_probability"][
        num_action_profile * state_mt : num_action_profile * (state_mt + 1), :
    ][action_mt, :]).flatten()
    
    assert np.allclose(empirical_probability, true_probability, atol=0.03)

def test_sub_simulation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state() 
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.solve_dynamic_problem()
    
    np.random.seed(3)
    state_m0 = 0
    m = 0
    
    result = model.simulate_sub(m=m, state_m0=state_m0, num_period=model.num["period"])
    
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == model.num["period"]
    assert all(col in result.columns for col in ["m", "t", "action_profile", "state_profile"])
    assert (result["m"] == m).all()
    assert (result["t"] == pl.Series(range(model.num["period"]))).all()

def test_full_simulation(model):
    model.discretize_state()
    model.compute_state_value()
    model.make_action_state()
    model.compute_payoff_covariate()
    model.compute_theta()
    model.compute_payoff()
    model.compute_transition()
    model.initialize_ccp_marginal()
    model.solve_dynamic_problem()
    
    np.random.seed(3)
    state_0 = model.initialize_state()
    model.simulate(state_0=state_0)
    model.add_state_value()
    
    assert model.result is not None

def test_equilibrium(model):
    model.solve_equilibrium()
    model.simulate_equilibrium()
    
    assert model.result is not None