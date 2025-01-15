# set up environment ----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.multiple.simulate_multiple import *

# set constants ----------------------------------------------------

num_simulation = 1000

num = {"market": 1000, "firm": 3, "period": 10, "state": 1, "action": 2, "tolerance": 1e-6}

num["grid_size"] = np.full(num["state"], 5)

# set parameters ----------------------------------------------------

param = {
    "param_payoff": {"beta": 0.7, "alpha": 0.5, "lambda": 0.5},
    "param_state": {
        "state_constant": np.concatenate((np.full(num["state"], 0.0),)),
        "state_trans": np.eye(num["state"]) * 0.1,
        "state_sd": np.eye(num["state"]),
    },
}

discount = 0.95

# solve the equilibrium -------------------------------------------

model = EquilibriumMultipleEntryExit(discount=discount, num=num, param=param)

## discretize the state -------------------------------------------

model.discretize_state()

model.markov.P
model.markov.state_values

model.markov.P.sum(axis=1)

## map to high-level parameters -----------------------------------

self = model

model.compute_state_value()

model.state_raw["state_value"]
model.state_raw["state_index"]

check = model.state_raw["state_index"].sort(
    model.state_raw["state_index"].columns[::-1]
)

(check == model.state_raw["state_index"]).min()

model.make_action_state()

model.action_state["action_state_value"]
model.action_state["action_state_index"]
model.action_state["action_state_index"].columns

check = model.action_state["action_state_index"].sort(
    model.action_state["action_state_index"].columns[::-1]
)

(check == model.action_state["action_state_index"]).min()

model.compute_payoff_covariate()

model.covariate

model.compute_theta()

model.theta

model.compute_payoff()

model.payoff["payoff_value"]
model.payoff["payoff_index"]

model.compute_transition()

model.transition["transition_probability"]
model.transition["transition_row_index"]
model.transition["transition_column_index"]

model.transition["transition_probability"].shape


## compute ex-ante value function ---------------------------------

model.initialize_ccp_marginal()

model.value["ccp_marginal_value"]
model.value["ccp_marginal_index"]

ccp_joint_value, ccp_joint_index = model.compute_ccp_joint(ccp_marginal_value=model.value["ccp_marginal_value"])

e_value, e_index = model.compute_conditional_expected_shock(ccp_marginal_value=model.value["ccp_marginal_value"])

e_value, e_index

ccp_marginal_diagonal_value, ccp_marginal_diagonal_row_index, ccp_marginal_diagonal_column_index = model.diagonalize_ccp_marginal(ccp_marginal_value=model.value["ccp_marginal_value"])

ccp_joint_diagonal_value, ccp_joint_diagonal_row_index, ccp_joint_diagonal_column_index = model.diagonalize_ccp_joint(ccp_joint_value=ccp_joint_value, ccp_joint_index=ccp_joint_index)

ccp_joint_diagonal_value, ccp_joint_diagonal_row_index, ccp_joint_diagonal_column_index

model.compute_exante_value(ccp_marginal_value=model.value["ccp_marginal_value"])

model.value["exante_value"]
model.value["exante_index"]

exante_value = model.value["exante_value"]

## compute condiitonal choice probability -----------------------

model.compute_choice_joint_value(exante_value=model.value["exante_value"])

model.value["choice_joint_value"]

model.compute_choice_marginal_value(choice_joint_value=model.value["choice_joint_value"], ccp_marginal_value=model.value["ccp_marginal_value"])

model.value["choice_marginal_value"]

choice_marginal_value = model.value["choice_marginal_value"]

model.compute_ccp_marginal(choice_marginal_value=model.value["choice_marginal_value"])

model.value["ccp_marginal_value"]

## solve dynamic problem ------------------------------------------

model.update_ccp_marginal(ccp_marginal_value=model.value["ccp_marginal_value"])

model.value["ccp_marginal_value"]

model.solve_dynamic_problem()

model.value["ccp_marginal_value"]

## simulate the model --------------------------------------------

### simulate an action -------------------------------------------

state_mt = 0

action_mt = model.simulate_action_mt(state_mt=state_mt)

np.random.seed(1)

ccp_joint_value, ccp_joint_index = model.compute_ccp_joint(ccp_marginal_value=model.value["ccp_marginal_value"])

check = ccp_joint_index.join(model.action_state["action_state_index"], on=[col for col in ccp_joint_index.columns if col.startswith("action_index_") or col.startswith("state_index_")], how="left")

num_action_profile = ccp_joint_index.select(pl.col("^action_.*$")).n_unique()
action_count = np.zeros(num_action_profile)

for _ in range(num_simulation):
    action_mt = model.simulate_action_mt(state_mt=state_mt)
    action_count[action_mt] += 1

empirical_probability = action_count / num_simulation

true_probability = ccp_joint_value[
    num_action_profile * state_mt : num_action_profile * (state_mt + 1)
]

true_probability = np.array(true_probability).flatten()

plt.scatter(empirical_probability, true_probability)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("True Probability")
plt.ylabel("Empirical Probability")
plt.show()

### simulate a state

state_mt = model.simulate_state_mt(action_mt=action_mt, state_mt=state_mt)

np.random.seed(1)

state_mt = 0
action_mt = 1

num_state = model.payoff["payoff_index"].select(pl.col("^state_.*$")).n_unique()
state_count = np.zeros(num_state)

for _ in range(num_simulation):
    state_mt = 0
    for _ in range(model.num["period"]):
        state_mt1 = model.simulate_state_mt(action_mt=action_mt, state_mt=state_mt)
    state_count[state_mt1] += 1   

empirical_probability = state_count / num_simulation
true_probability = model.transition["transition_probability"][
    num_action_profile * state_mt : num_action_profile * (state_mt + 1), :
][action_mt, :]

true_probability = np.array(true_probability).flatten()

plt.scatter(empirical_probability, true_probability)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("True Probability")
plt.ylabel("Empirical Probability")
plt.show()

### simulate state and action in a period ------------------------

state_mt = 0

action_mt, state_mt1 = model.simulate_t(state_mt=state_mt)

np.random.seed(3)

state_m0 = 0

action_m, state_m = model.simulate_m(state_m0=state_m0)

action_m
state_m
model.state_raw["state_value"][state_m, :]

np.random.seed(3)

state_0 = model.initialize_state()

model.simulate_sub(m=0, state_m0=state_m0, num_period=model.num["period"])

model.simulate(state_0=state_0)

model.add_state_value()

model.result

# wrap up ----------------------------------------------------------

model.solve_equilibrium()

model.simulate_equilibrium()