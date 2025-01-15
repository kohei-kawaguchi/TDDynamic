# set environmen --------------------------------------------------

EulerGamma = 0.577215664901532860606512090082402431042159336
import re

import numpy as np
import polars as pl
import quantecon as qe
from scipy.linalg import block_diag
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

class EquilibriumMultiple:
    def __init__(self, discount, num):
        self.action_state = {"action_state_value": None, "action_state_index": None}
        self.payoff = {"payoff_value": None, "payoff_index": None}
        self.transition = {
            "transition_probability": None,
            "transition_row_index": None,
            "transition_column_index": None
        }
        self.value = {
            "ccp_marginal_value": None,
            "ccp_marginal_index": None,
            "choice_joint_value": None,
            "choice_joint_index": None,
            "choice_marginal_value": None,
            "choice_marginal_index": None,
            "exante_value": None
        }
        self.result = None
        self.action_set = None
        self.discount = discount
        self.num = num

    def initialize_ccp_marginal(self):
        ccp_marginal_index = [
            self.payoff["payoff_index"].select(["action_index_" + str(action)] + [col for col in self.payoff["payoff_index"].columns if col.startswith("state_index_")]).unique()
            for action in range(len(self.payoff["payoff_value"]))
        ]

        ccp_marginal_index = [
            index.sort(
                index.columns[::-1]
            )
            for index in ccp_marginal_index
        ]

        ccp_marginal_value = [
            np.full(index.shape[0], 1 / index.select(pl.col("^action_index_.*$")).n_unique()).reshape(-1, 1)
            for index in ccp_marginal_index
        ]

        self.value["ccp_marginal_value"] = ccp_marginal_value
        self.value["ccp_marginal_index"] = ccp_marginal_index 
        
    def compute_ccp_joint(self, ccp_marginal_value):
        ccp_value = [
            self.value["ccp_marginal_index"][i].with_columns(pl.Series(f"ccp_{i}", ccp_marginal_value[i].flatten()))
            for i in range(len(ccp_marginal_value))
        ]

        ccp_joint_value = self.action_state["action_state_index"]
        for i in range(len(ccp_value)):
            ccp_joint_value = ccp_joint_value.join(ccp_value[i], on=ccp_value[i].columns[:-1], how="left")

        ccp_joint_value = ccp_joint_value.with_columns(
            pl.fold(
                1,
                lambda acc, x: acc * x,
                exprs=[pl.col(f"ccp_{i}") for i in range(len(ccp_value))]
            ).alias("ccp_joint")
        )

        ccp_joint_index = ccp_joint_value.select([col for col in ccp_joint_value.columns if col.startswith("action_index_") or col.startswith("state_index_")])
        ccp_joint_value = ccp_joint_value.select("ccp_joint").to_numpy()
        
        return ccp_joint_value, ccp_joint_index
        
    def compute_conditional_expected_shock(self, ccp_marginal_value):
        e_value = [
            EulerGamma - np.log(value)
            for value in ccp_marginal_value
        ]
        e_index = self.value["ccp_marginal_index"]
        return e_value, e_index
    
    def diagonalize_ccp_marginal(self, ccp_marginal_value):
        ccp_diagonal_value = [
            ccp_marginal_value[i].reshape(
                -1, self.value["ccp_marginal_index"][i][f"action_index_{i}"].unique().len()
            )
            for i in range(len(ccp_marginal_value))
        ]

        ccp_diagonal_value = [
            [row for row in value]
            for value in ccp_diagonal_value
        ]
        ccp_diagonal_value = [
            block_diag(*value)
            for value in ccp_diagonal_value
        ]
        ccp_diagonal_row_index = self.value["ccp_marginal_index"][0].select(pl.col("^state_index_.*$")).unique()
        ccp_diagonal_row_index = ccp_diagonal_row_index.sort(
            ccp_diagonal_row_index.columns[::-1]
        )
        ccp_diagonal_column_index = self.value["ccp_marginal_index"]

        return ccp_diagonal_value, ccp_diagonal_row_index, ccp_diagonal_column_index
    
    def diagonalize_ccp_joint(self, ccp_joint_value, ccp_joint_index):
        ccp_joint_diagonal_value = ccp_joint_value.reshape(
            -1, int(self.transition["transition_probability"].shape[0] / self.transition["transition_probability"].shape[1])
        )

        ccp_joint_diagonal_value = [row for row in ccp_joint_diagonal_value]
        ccp_joint_diagonal_value = block_diag(*ccp_joint_diagonal_value)

        ccp_joint_diagonal_row_index = ccp_joint_index.select(pl.col("^state_index_.*$")).unique()

        ccp_joint_diagonal_row_index = ccp_joint_diagonal_row_index.sort(
            ccp_joint_diagonal_row_index.columns[::-1]
        )
        ccp_joint_diagonal_column_index = ccp_joint_index
        
        return ccp_joint_diagonal_value, ccp_joint_diagonal_row_index, ccp_joint_diagonal_column_index

    def compute_exante_value(self, ccp_marginal_value):
        self.value["ccp_marginal_value"] = ccp_marginal_value

        ccp_marginal_diagonal_value, _, _ = self.diagonalize_ccp_marginal(ccp_marginal_value=ccp_marginal_value)

        ccp_joint_value, ccp_joint_index = self.compute_ccp_joint(ccp_marginal_value=ccp_marginal_value)

        ccp_joint_diagonal_value, _, _ = self.diagonalize_ccp_joint(ccp_joint_value=ccp_joint_value, ccp_joint_index=ccp_joint_index)

        e_value, _ = self.compute_conditional_expected_shock(ccp_marginal_value=ccp_marginal_value)

        d = [
            np.dot(ccp_marginal_diagonal_value[i], e_value[i])
            for i in range(len(ccp_marginal_diagonal_value))
        ]

        term_1 = np.eye(ccp_joint_diagonal_value.shape[0]) - self.discount * np.dot(
            ccp_joint_diagonal_value, self.transition["transition_probability"]
        )
        term_2 = [
            np.dot(ccp_joint_diagonal_value, self.payoff["payoff_value"][i]) + d[i]
            for i in range(len(self.payoff["payoff_value"]))
        ]
        exante_value = [
            np.linalg.solve(term_1, term_2[i])
            for i in range(len(term_2))
        ]
        self.value["exante_value"] = exante_value
        self.value["exante_index"] = self.payoff["payoff_index"]

    def compute_choice_joint_value(self, exante_value):
        self.value["exante_value"] = exante_value
        choice_joint_value = [
            self.payoff["payoff_value"][i] + self.discount * np.dot(
                self.transition["transition_probability"], exante_value[i]
            )
            for i in range(len(exante_value))
        ]
        self.value["choice_joint_value"] = choice_joint_value
        self.value["choice_joint_index"] = self.payoff["payoff_index"]
    
    def compute_choice_marginal_value(self, choice_joint_value, ccp_marginal_value):

        self.value["ccp_marginal_value"] = ccp_marginal_value
        self.value["choice_joint_value"] = choice_joint_value

        ccp_joint_value, ccp_joint_index = self.compute_ccp_joint(ccp_marginal_value=ccp_marginal_value)

        ccp_joint_df = pl.concat([ccp_joint_index, pl.DataFrame({"ccp_joint_value": ccp_joint_value.flatten()})], how="horizontal")

        choice_joint_df = [
            pl.concat([self.payoff["payoff_index"], pl.DataFrame({"choice_joint_value": choice_joint_value[i].flatten()})], how="horizontal")
            for i in range(len(choice_joint_value))
        ]

        df = [
            pl.concat([self.value["ccp_marginal_index"][i], pl.DataFrame({"ccp_marginal_value": self.value["ccp_marginal_value"][i].flatten()})], how="horizontal")
            for i in range(len(self.value["ccp_marginal_index"]))
        ]

        df = [
            ccp_joint_df.join(df[i], on=df[i].columns[:-1], how="left").with_columns(
                pl.col("ccp_joint_value") / pl.col("ccp_marginal_value")
            ).rename({"ccp_joint_value": "ccp_other_value"})
            for i in range(len(df))
        ]

        df = [
            df[i].join(choice_joint_df[i], on=choice_joint_df[i].select(pl.col("^action_index_.*$|^state_index_.*$")).columns, how="left").with_columns(
                pl.col("choice_joint_value") * pl.col("ccp_other_value")
            )
            for i in range(len(df))
        ]

        df = [
            df[i].group_by([col for col in df[i].columns if col.startswith("state_index_") or col == f"action_index_{i}"]).agg(
                pl.col("choice_joint_value").sum()
            )
            for i in range(len(df))
        ]

        df = [
            df[i].sort(df[i].columns[-2::-1])
            for i in range(len(df))
        ]

        choice_marginal_value = [
            df[i].select("choice_joint_value").to_numpy()
            for i in range(len(df))
        ]

        choice_marginal_index = [
            df[i].drop("choice_joint_value")
            for i in range(len(df))
        ]

        self.value["choice_marginal_value"] = choice_marginal_value
        self.value["choice_marginal_index"] = choice_marginal_index

    def compute_ccp_marginal(self, choice_marginal_value):
        self.value["choice_marginal_value"] = choice_marginal_value
        const = np.max(choice_marginal_value)
        choice_marginal_value = np.exp(choice_marginal_value - const)
        num_action = [
            int(index.select(pl.col("^action_index_.*$")).n_unique())
            for index in self.value["choice_marginal_index"]
        ]
        num_state = [
            int(index.select(pl.col("^state_index_.*$")).n_unique())
            for index in self.value["choice_marginal_index"]
        ]
        aggregator = [
            np.full((num_action[i], num_action[i]), 1)
            for i in range(len(num_action))
        ]
        aggregator = [
            block_diag(*[a] * num_state[i])
            for i, a in enumerate(aggregator)
        ]
        aggregator = [
            np.dot(a, choice_marginal_value[i])
            for i, a in enumerate(aggregator)
        ]
        ccp_marginal_value = [
            choice_marginal_value[i] / a
            for i, a in enumerate(aggregator)
        ]
        self.value["ccp_marginal_value"] = ccp_marginal_value

    def update_ccp_marginal(self, ccp_marginal_value):
        self.value["ccp_marginal_value"] = ccp_marginal_value
        self.compute_exante_value(ccp_marginal_value=self.value["ccp_marginal_value"])
        self.compute_choice_joint_value(exante_value=self.value["exante_value"])
        self.compute_choice_marginal_value(choice_joint_value=self.value["choice_joint_value"], ccp_marginal_value=self.value["ccp_marginal_value"])
        self.compute_ccp_marginal(choice_marginal_value=self.value["choice_marginal_value"])

    def solve_dynamic_problem(self):
        self.initialize_ccp_marginal()
        distance = 1000
        while distance > self.num["tolerance"]:
            ccp_marginal_value_old = self.value["ccp_marginal_value"]
            self.update_ccp_marginal(ccp_marginal_value=ccp_marginal_value_old)
            distance = np.max(np.abs(np.concatenate(self.value["ccp_marginal_value"]) - np.concatenate(ccp_marginal_value_old)))
            print(f"Distance: {distance}")

    def simulate_action_mt(self, state_mt):
        ccp_joint_value, ccp_joint_index = self.compute_ccp_joint(ccp_marginal_value=self.value["ccp_marginal_value"])

        num_action_profile = ccp_joint_index.select(pl.col("^action_index_.*$")).n_unique()
        probability = ccp_joint_value[
            num_action_profile * state_mt : num_action_profile * (state_mt + 1)
        ]
        probability = np.array(probability).flatten()
        action_mt = np.random.choice(a=num_action_profile, p=probability)
        
        return action_mt


    def simulate_state_mt(self, action_mt, state_mt):
        num_action_profile = self.payoff["payoff_index"].select(pl.col("^action_index_.*$")).n_unique()
        num_state = self.payoff["payoff_index"].select(pl.col("^state_index_.*$")).n_unique()
        probability = self.transition["transition_probability"][
            num_action_profile * state_mt : num_action_profile * (state_mt + 1), :
        ]
        probability = probability[action_mt, :]
        state_mt1 = np.random.choice(a=num_state, p=probability)

        return state_mt1

    def simulate_t(self, state_mt):
        action_mt = self.simulate_action_mt(state_mt=state_mt)
        state_mt1 = self.simulate_state_mt(action_mt=action_mt, state_mt=state_mt)
        return action_mt, state_mt1

    def simulate_m(self, state_m0):
        state_mt = state_m0
        state_m = np.full(self.num["period"], state_m0)
        action_m = np.full(self.num["period"], 0)
        for t in range(self.num["period"]):
            action_mt, state_mt1 = self.simulate_t(state_mt=state_mt)
            action_m[t] = action_mt
            if t < self.num["period"] - 1:
                state_m[t + 1] = state_mt1
            state_mt = state_mt1
        return action_m, state_m

    def simulate_sub(self, m, state_m0, num_period):
        action_m, state_m = self.simulate_m(state_m0=state_m0)
        return pl.DataFrame(
            {
                "m": m,
                "t": range(num_period),
                "action_profile": action_m,
                "state_profile": state_m,
            }
        )

    def simulate(self, state_0):
        # Set start method to 'spawn'
        multiprocessing.set_start_method('spawn', force=True)
        
        result = pl.DataFrame()
        with ProcessPoolExecutor() as executor:
            simulate_fn = partial(self.simulate_sub, num_period=self.num["period"])
            results = list(executor.map(simulate_fn, range(self.num["market"]), state_0))
            
        result = pl.concat(results)
        self.result = result

    def add_state_value(self):
        action_state_index = self.action_state["action_state_index"]
        action_state_value = self.action_state["action_state_value"]
        action_state_value = pl.DataFrame(action_state_value)
        action_state_value.columns = action_state_index.select(pl.col("^action_index_.*|state_index_.*$")).columns
        action_state_value.columns = [
            re.sub("action_index_", "action_value_", col) for col in action_state_value.columns
        ]
        action_state_value.columns = [
            re.sub("state_index_", "state_value_", col) for col in action_state_value.columns
        ]
        action_state_profile = pl.concat([action_state_index, action_state_value], how="horizontal")

        result = self.result.join(action_state_profile, on=["action_profile", "state_profile"], how="left")

        self.result = result

    def compute_payoff(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def compute_transition(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def initialize_state(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def solve_equilibrium(self):
        raise NotImplementedError("This method must be implemented in the subclass")


class EquilibriumMultipleLinear(EquilibriumMultiple):
    def __init__(self, discount, num):
        super().__init__(discount=discount, num=num)
        self.covariate = None
        self.theta = None

    def compute_payoff(self):
        payoff_value = [np.dot(cov, self.theta) for cov in self.covariate]

        self.payoff["payoff_value"] = payoff_value
        self.payoff["payoff_index"] = self.action_state["action_state_index"]

    def compute_payoff_covariate(self):
        raise NotImplementedError("This method must be implemented in the subclass")

    def compute_theta(self):
        raise NotImplementedError("This method must be implemented in the subclass")


class EquilibriumMultipleEntryExit(EquilibriumMultipleLinear):
    def __init__(self, discount, num, param):
        super().__init__(discount=discount, num=num)
        self.param = param
        self.markov = None
        self.state_raw = {"state_value": None, "state_index": None}

    # def make_action_set(self, result):
    #     action_set = result.select(["i", "t"])
    #     action_index = pl.DataFrame({"action_index": range(self.num["action"])})
    #     action_set = action_set.join(action_index, how="cross").sort(
    #         ["i", "t", "action_index"]
    #     )
    #     self.action_set = action_set
    #     return action_set

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

        state = self.markov.state_values
        action = [np.arange(self.num["action"]) for _ in range(self.num["firm"])]
        past_action = np.array(np.meshgrid(*action)).T.reshape(-1, self.num["firm"])

        state_repeated = np.repeat(state, past_action.shape[0], axis=0)
        past_action_tiled = np.tile(past_action, (state.shape[0], 1))
        state_combined = np.hstack((past_action_tiled, state_repeated))
        
        state_combined = pl.DataFrame(state_combined)
        state_combined = state_combined.sort(
            state_combined.columns[::-1]
        )
        state_combined = state_combined.to_numpy()

        state_index = pl.DataFrame(
            {
                f"state_index_{i}": pl.Series(state_combined[:, i]).rank("dense") - 1
                for i in range(state_combined.shape[1])
            }
        )

        self.state_raw["state_value"] = state_combined
        self.state_raw["state_index"] = state_index

    def make_action_state(self):
        action = [np.arange(self.num["action"]) for _ in range(self.num["firm"])]
        action = np.array(np.meshgrid(*action)).T.reshape(-1, self.num["firm"])
        self.compute_state_value()

        state_combined_repeated = np.repeat(
            self.state_raw["state_value"], action.shape[0], axis=0
        )
        action_tiled = np.tile(action, (self.state_raw["state_value"].shape[0], 1))
        action_state = np.hstack((action_tiled, state_combined_repeated))

        action_state = pl.DataFrame(action_state)
        action_state = action_state.sort(
            action_state.columns[::-1]
        )
        action_state = action_state.to_numpy()

        action_state_index = {"action_index_0": pl.Series(action_state[:, 0]).rank("dense") - 1}
        action_state_index.update(
            {
                f"action_index_{i}": pl.Series(action_state[:, i]).rank("dense") - 1
                for i in range(1, self.num["firm"])       
            }
        )
        action_state_index.update(
            {
                f"state_index_{i - self.num['firm']}": pl.Series(action_state[:, i]).rank("dense") - 1
                for i in range(self.num["firm"], action_state.shape[1])
            }
        )
        action_state_index = pl.DataFrame(action_state_index).select(pl.all().cast(pl.Int64))
        
        action_index = action_state_index.select(pl.col("^action_index_.*$")).unique()
        action_index = action_index.sort(action_index.columns[::-1]).with_columns(
            pl.lit(np.arange(action_index.shape[0])).alias("action_profile")
        )
        state_index = action_state_index.select(pl.col("^state_index_.*$")).unique()
        state_index = state_index.sort(state_index.columns[::-1]).with_columns(
            pl.lit(np.arange(state_index.shape[0])).alias("state_profile")
        )

        action_state_index = action_state_index.join(action_index, on=action_index.columns[:-1], how="left").join(state_index, on=state_index.columns[:-1], how="left")

        action_state_index = action_state_index.select(
            ["action_profile", "state_profile"] + [col for col in action_state_index.columns if col not in ["action_profile", "state_profile"]]
        )

        self.action_state["action_state_value"] = action_state
        self.action_state["action_state_index"] = action_state_index
        
    def compute_payoff_covariate(self):
        covariate = [
            np.column_stack(
                [
                    self.action_state["action_state_value"][:, -1] * self.action_state["action_state_value"][:, i],
                    (self.action_state["action_state_value"][:, i] - np.sum(self.action_state["action_state_value"][:, :self.num["firm"]], axis=1)) * self.action_state["action_state_value"][:, i],
                    -(1 - self.action_state["action_state_value"][:, self.num["firm"] + i]) * self.action_state["action_state_value"][:, i],
                ]
            )
            for i in range(self.num["firm"])
        ]

        self.covariate = covariate

    def compute_theta(self):
        theta = np.array(
            [
                [self.param["param_payoff"]["beta"]],
                [self.param["param_payoff"]["alpha"]],
                [self.param["param_payoff"]["lambda"]],
            ]
        )
        self.theta = theta

    def compute_transition(self):
        num_action_profile = self.num["action"] ** self.num["firm"]

        transition_probability = np.zeros(
            (
                num_action_profile ** 2 * self.markov.P.shape[0],
                num_action_profile * self.markov.P.shape[1],
            )
        )
        for k in range(num_action_profile):
            for l in range(num_action_profile):
                for i in range(self.markov.P.shape[0]):
                    for j in range(self.markov.P.shape[1]):
                        transition_probability[
                            num_action_profile ** 2 * i + num_action_profile * l + k,
                            num_action_profile * j + k,
                        ] = self.markov.P[i, j]

        self.transition["transition_probability"] = transition_probability
        self.transition["transition_row_index"] = self.payoff["payoff_index"]
        self.transition["transition_column_index"] = self.state_raw["state_index"]

    def initialize_state(self):
        num_action_profile = self.payoff["payoff_index"].select(pl.col("^action_.*$")).n_unique()
        stationary = qe.MarkovChain(self.markov.P).stationary_distributions.flatten()
        z_0 = np.random.choice(
            a=len(stationary), size = self.num["market"], p=stationary
        )
        a_0 = np.zeros(self.num["market"])
        ccp_joint_value, _ = self.compute_ccp_joint(ccp_marginal_value=self.value["ccp_marginal_value"])
        for m in range(self.num["market"]):
            probability = ccp_joint_value[
                num_action_profile * z_0[m] : num_action_profile * (z_0[m] + 1)
            ].flatten()
            a_0[m] = np.random.choice(a=num_action_profile, p=probability)
        state_0 = num_action_profile * z_0 + a_0
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

        self.initialize_ccp_marginal()

        self.solve_dynamic_problem()

    def simulate_equilibrium(self):
        ## solve the model ------------------------------------------------

        self.solve_equilibrium()

        ## simulate the model --------------------------------------------

        state_0 = self.initialize_state()

        self.simulate(state_0=state_0)

        self.add_state_value()