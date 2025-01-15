# set up environment ----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.single.simulate_single import *

# set constants ----------------------------------------------------

num_simulation = 1000

num = {"firm": 1000, "period": 10, "state": 1, "action": 2, "tolerance": 1e-6}

num["grid_size"] = np.full(num["state"], 5)

# set parameters ----------------------------------------------------

param = {
    "param_payoff": {"beta": 1, "lambda": 0.5},
    "param_state": {
        "state_constant": np.concatenate((np.full(num["state"], 0.0),)),
        "state_trans": np.eye(num["state"]) * 0.1,
        "state_sd": np.eye(num["state"]),
    },
}

discount = 0.95

# solve the equilibrium -------------------------------------------

model = EquilibriumSingleEntryExit(discount=discount, num=num, param=param)

## discretize the state -------------------------------------------

model.discretize_state()

model.markov.P
model.markov.state_values

model.markov.P.sum(axis=1)

## map to high-level parameters -----------------------------------

model.compute_state_value()

model.state_raw["state_value"]
model.state_raw["state_index"]

model.make_action_state()

model.action_state

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

## compute ex-ante value function ---------------------------------

model.initialize_ccp()

model.value["ccp_value"]
model.value["ccp_index"]

e_value, e_index = model.compute_conditional_expected_shock()

e_value, e_index

(
    ccp_diagonal_value,
    ccp_diagonal_row_index,
    ccp_diagonal_column_index,
) = model.diagonalize_ccp()

ccp_diagonal_value, ccp_diagonal_row_index, ccp_diagonal_column_index

model.compute_exante_value(ccp_value=model.value["ccp_value"])

model.value["exante_value"]
model.value["ccp_index"]

## compute condiitonal choice probability -----------------------

model.compute_choice_value(exante_value=model.value["exante_value"])

model.value["choice_value"]

model.compute_ccp(choice_value=model.value["choice_value"])

model.value["ccp_value"]

## solve dynamic problem ------------------------------------------

model.update_exante_value()

model.value["exante_value"]
model.value["ccp_index"]

model.solve_dynamic_problem()

model.value["exante_value"]
model.value["ccp_index"]

## simulate the model --------------------------------------------

### simulate an action -------------------------------------------

state_it = 0

action_it = model.simulate_action_it(state_it=state_it)

np.random.seed(1)

action_count = np.zeros(num["action"])

for _ in range(num_simulation):
    action_it = model.simulate_action_it(state_it=state_it)
    action_count[action_it] += 1

empirical_probability = action_count / num_simulation
true_probability = model.value["ccp_value"][num["action"] * state_it : num["action"] * (state_it + 1)]

true_probability = np.array(true_probability).flatten()

plt.scatter(empirical_probability, true_probability)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("True Probability")
plt.ylabel("Empirical Probability")
plt.show()

### simulate a state

state_it = model.simulate_state_it(action_it=action_it, state_it=state_it)

np.random.seed(1)

action_it = 1
state_it = 0

state_count = np.zeros(model.transition["transition_probability"].shape[1])

for _ in range(num_simulation):
    state_it = 0
    for _ in range(num["period"]):
        state_it1 = model.simulate_state_it(action_it=action_it, state_it=state_it)
    state_count[state_it1] += 1

empirical_probability = state_count / num_simulation
true_probability = model.transition["transition_probability"][
    num["action"] * state_it : num["action"] * (state_it + 1), :
][action_it, :]

plt.scatter(empirical_probability, true_probability)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("True Probability")
plt.ylabel("Empirical Probability")
plt.show()

### simulate state and action in a period ------------------------

action_it, state_it1 = model.simulate_it(state_it=state_it)

np.random.seed(3)

state_i0 = 0

action_i, state_i = model.simulate_i(state_i0=state_i0)

action_i
state_i
model.state_raw["state_value"][state_i, :]

np.random.seed(3)

state_0 = model.initialize_state()

model.simulate(state_0=state_0)

model.add_state_value()

model.result

# wrap up ----------------------------------------------------------

model.solve_equilibrium()

model.simulate_equilibrium()

model.result
