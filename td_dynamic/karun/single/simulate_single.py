import re

import numpy as np
import polars as pl
import quantecon as qe
from scipy.linalg import block_diag

from td_dynamic.karun.constant import EulerGamma


class EquilibriumSingle:
    def __init__(self, discount, num):
        self.action_state = {"action_state_value": None, "action_state_index": None}
        self.payoff = {"payoff_value": None, "payoff_index": None}
        self.transition = {
            "transition_probability": None,
            "transition_row_index": None,
            "transition_column_index": None,
        }
        self.value = {
            "ccp_value": None,
            "choice_value": None,
            "exante_value": None,
            "ccp_index": None,
        }
        self.result = None
        self.action_set = None
        self.discount = discount
        self.num = num

    def initialize_ccp(self):
        ccp_value = np.full(
            self.payoff["payoff_value"].shape[0],
            1 / self.payoff["payoff_index"]["action"].unique().len(),
        )

        ccp_value = ccp_value.reshape(-1, 1)
        self.value["ccp_value"] = ccp_value
        self.value["ccp_index"] = self.payoff["payoff_index"]

    def compute_conditional_expected_shock(self):
        e_value = EulerGamma - np.log(self.value["ccp_value"])
        e_index = self.value["ccp_index"]
        return e_value, e_index

    def diagonalize_ccp(self):
        ccp_diagonal_value = self.value["ccp_value"].reshape(-1, self.value["ccp_index"]["action"].unique().len())

        ccp_diagonal_value = [row for row in ccp_diagonal_value]
        ccp_diagonal_value = block_diag(*ccp_diagonal_value)
        ccp_diagonal_row_index = self.value["ccp_index"].drop("action").unique()

        ccp_diagonal_row_index = ccp_diagonal_row_index.sort(ccp_diagonal_row_index.columns[::-1])
        ccp_diagonal_column_index = self.value["ccp_index"].select("action")
        return ccp_diagonal_value, ccp_diagonal_row_index, ccp_diagonal_column_index

    def compute_exante_value(self, ccp_value):
        self.value["ccp_value"] = ccp_value
        e, *_ = self.compute_conditional_expected_shock()
        ccp_diagonal_value, *_ = self.diagonalize_ccp()
        term_1 = np.eye(ccp_diagonal_value.shape[0]) - self.discount * np.dot(
            ccp_diagonal_value, self.transition["transition_probability"]
        )
        term_2 = np.dot(ccp_diagonal_value, self.payoff["payoff_value"] + e)
        exante_value = np.linalg.solve(term_1, term_2)
        self.value["exante_value"] = exante_value
        self.value["ccp_index"] = self.payoff["payoff_index"]

    def compute_choice_value(self, exante_value):
        self.value["exante_value"] = exante_value
        choice_value = self.payoff["payoff_value"] + self.discount * np.dot(
            self.transition["transition_probability"], self.value["exante_value"]
        )
        self.value["choice_value"] = choice_value

    def compute_ccp(self, choice_value):
        self.value["choice_value"] = choice_value
        choice_value = np.exp(self.value["choice_value"])
        num_action = int(
            self.transition["transition_probability"].shape[0] / self.transition["transition_probability"].shape[1]
        )
        num_state = int(self.transition["transition_probability"].shape[0] / num_action)
        aggregator = np.full((num_action, num_action), 1)
        aggregator = block_diag(*[aggregator] * num_state)
        aggregator = np.dot(aggregator, choice_value)
        ccp = choice_value / aggregator
        self.value["ccp_value"] = ccp

    def update_exante_value(self, exante_value):
        self.compute_choice_value(exante_value=exante_value)
        self.compute_ccp(choice_value=self.value["choice_value"])
        self.compute_exante_value(ccp_value=self.value["ccp_value"])

    def solve_dynamic_problem(self):
        self.initialize_ccp()
        self.compute_exante_value(ccp_value=self.value["ccp_value"])
        distance = 1000
        while distance > self.num["tolerance"]:
            exante_value_old = self.value["exante_value"]
            self.update_exante_value(exante_value=exante_value_old)
            distance = np.max(np.abs(self.value["exante_value"] - exante_value_old))
            print(f"Distance: {distance}")
        self.compute_ccp(choice_value=self.value["choice_value"])

    def simulate_action_it(self, state_it):
        num_action = self.value["ccp_index"]["action"].unique().len()
        probability = self.value["ccp_value"][num_action * state_it : num_action * (state_it + 1)]
        probability = np.array(probability).flatten()
        action_it = np.random.choice(a=num_action, p=probability)
        return action_it

    def simulate_state_it(self, action_it, state_it):
        num_action = int(
            self.transition["transition_probability"].shape[0] / self.transition["transition_probability"].shape[1]
        )
        num_state = int(self.transition["transition_probability"].shape[0] / num_action)
        probability = self.transition["transition_probability"][num_action * state_it : num_action * (state_it + 1), :]
        probability = probability[action_it, :]
        state_it = np.random.choice(a=num_state, p=probability)

        return state_it

    def simulate_it(self, state_it):
        action_it = self.simulate_action_it(state_it=state_it)
        state_it1 = self.simulate_state_it(action_it=action_it, state_it=state_it)
        return action_it, state_it1

    def simulate_i(self, state_i0):
        state_it = state_i0
        state_i = np.full(self.num["period"], state_i0)
        action_i = np.full(self.num["period"], 0)
        for t in range(self.num["period"]):
            action_it, state_it = self.simulate_it(state_it=state_it)
            action_i[t] = action_it
            if t < self.num["period"] - 1:
                state_i[t + 1] = state_it
        return action_i, state_i

    def simulate(self, state_0):
        result = pl.DataFrame()
        for i in range(self.num["firm"]):
            action_i, state_i = self.simulate_i(state_i0=state_0[i])
            result_i = pl.DataFrame(
                {
                    "i": i,
                    "t": range(len(action_i)),
                    "action_index": action_i,
                    "state_index": state_i,
                }
            )
            result = result.vstack(result_i)
        self.result = result

    def add_state_value(self):
        index = self.state_raw["state_index"][self.result["state_index"]]
        value = self.state_raw["state_value"][self.result["state_index"]]
        value = pl.DataFrame(value)
        value.columns = [re.sub("column_", "state_value_", col) for col in value.columns]

        result = pl.concat([self.result, index, value], how="horizontal")

        self.result = result

    def compute_payoff(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def compute_transition(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def initialize_state(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def solve_equilibrium(self):
        raise NotImplementedError("This method must be implemented in the subclass")


class EquilibriumSingleLinear(EquilibriumSingle):
    def __init__(self, discount, num):
        super().__init__(discount=discount, num=num)
        self.covariate = None
        self.theta = None

    def compute_payoff(self):
        payoff_value = np.dot(self.covariate, self.theta)

        payoff_index = {"action": pl.Series(self.action_state["action_state_value"][:, 0]).rank("dense") - 1}
        payoff_index.update(
            {
                f"state_{i - 1}": pl.Series(self.action_state["action_state_value"][:, i]).rank("dense") - 1
                for i in range(1, self.action_state["action_state_value"].shape[1])
            }
        )
        payoff_index = pl.DataFrame(payoff_index)

        self.payoff["payoff_value"] = payoff_value
        self.payoff["payoff_index"] = payoff_index

    def compute_payoff_covariate(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def compute_theta(self):
        raise NotImplementedError("This method must be implemented in the subclass")


class EquilibriumSingleEntryExit(EquilibriumSingleLinear):
    def __init__(self, discount, num, param):
        super().__init__(discount=discount, num=num)
        self.param = param
        self.markov = None
        self.state_raw = {"state_value": None, "state_index": None}

    def make_action_set(self, result):
        action_set = result.select(["i", "t"])
        action_index = pl.DataFrame({"action_index": range(self.num["action"])})
        action_set = action_set.join(action_index, how="cross").sort(["i", "t", "action_index"])
        self.action_set = action_set
        return action_set

    def discretize_state(self):
        markov = qe.markov.approximation.discrete_var(
            A=self.param["param_state"]["state_trans"],
            C=self.param["param_state"]["state_sd"],
            grid_sizes=self.num["grid_size"],
        )
        state_mu = self.param["param_state"]["state_constant"] / (
            1 - np.diag(self.param["param_state"]["state_trans"])
        )
        state_mu = np.tile(state_mu, (markov.state_values.shape[0], 1))
        markov.state_values = markov.state_values + state_mu
        self.markov = markov

    def compute_state_value(self):
        past_action = np.arange(self.num["action"]).reshape(-1, 1)
        state = self.markov.state_values

        state_repeated = np.repeat(state, past_action.shape[0], axis=0)
        past_action_tiled = np.tile(past_action, (state.shape[0], 1))
        state_combined = np.hstack((past_action_tiled, state_repeated))

        state_index = pl.DataFrame(
            {f"state_{i}": pl.Series(state_combined[:, i]).rank("dense") - 1 for i in range(state_combined.shape[1])}
        )

        self.state_raw["state_value"] = state_combined
        self.state_raw["state_index"] = state_index

    def make_action_state(self):
        action = np.arange(self.num["action"]).reshape(-1, 1)
        self.compute_state_value()

        state_combined_repeated = np.repeat(self.state_raw["state_value"], action.shape[0], axis=0)
        action_tiled = np.tile(action, (self.state_raw["state_value"].shape[0], 1))
        action_state = np.hstack((action_tiled, state_combined_repeated))
        self.action_state["action_state_value"] = action_state

    def compute_payoff_covariate(self):
        covariate = np.column_stack(
            [
                self.action_state["action_state_value"][:, 2] * self.action_state["action_state_value"][:, 0],
                -(1 - self.action_state["action_state_value"][:, 1]) * self.action_state["action_state_value"][:, 0],
            ]
        )
        self.covariate = covariate

    def compute_theta(self):
        theta = np.array(
            [
                [self.param["param_payoff"]["beta"]],
                [self.param["param_payoff"]["lambda"]],
            ]
        )
        self.theta = theta

    def compute_transition(self):
        transition_probability = np.zeros(
            (
                self.num["action"] ** 2 * self.markov.P.shape[0],
                self.num["action"] * self.markov.P.shape[1],
            )
        )
        for k in range(self.num["action"]):
            for l in range(self.num["action"]):
                for i in range(self.markov.P.shape[0]):
                    for j in range(self.markov.P.shape[1]):
                        transition_probability[
                            self.num["action"] ** 2 * i + self.num["action"] * l + k,
                            self.num["action"] * j + k,
                        ] = self.markov.P[i, j]

        transition_row_index = self.payoff["payoff_index"]
        transition_column_index = self.state_raw["state_index"]

        self.transition["transition_probability"] = transition_probability
        self.transition["transition_row_index"] = transition_row_index
        self.transition["transition_column_index"] = transition_column_index

    def initialize_state(self):
        num_action = int(self.value["ccp_value"].shape[0] / (2 * self.markov.P.shape[0]))
        stationary = qe.MarkovChain(self.markov.P).stationary_distributions
        z_0 = np.random.choice(a=len(stationary[0]), size=self.num["firm"], p=stationary[0])
        a_0 = np.zeros(self.num["firm"])
        for i in range(self.num["firm"]):
            z_i0 = z_0[i]
            probability = self.value["ccp_value"][num_action * z_i0 : num_action * (z_i0 + 1)].flatten()
            a_0[i] = np.random.choice(a=num_action, p=probability)
        state_0 = num_action * z_0 + a_0
        state_0 = state_0.astype(int)
        return state_0

    def solve_equilibrium(self):
        ## discretize the state -------------------------------------------

        self.discretize_state()

        ## map to high-level parameters -----------------------------------

        self.compute_state_value()

        self.make_action_state()

        self.compute_payoff_covariate()

        self.compute_theta()

        self.compute_payoff()

        self.compute_transition()

        ## solve dynamic problem ------------------------------------------

        self.initialize_ccp()

        self.solve_dynamic_problem()

    def simulate_equilibrium(self):
        ## solve the model ------------------------------------------------

        self.solve_equilibrium()

        ## simulate the model --------------------------------------------

        state_0 = self.initialize_state()

        self.simulate(state_0=state_0)

        self.add_state_value()
        self.make_action_set(self.result)
