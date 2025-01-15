import logging
from copy import deepcopy

import numpy as np
import polars as pl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.base.model import GenericLikelihoodModel

from td_dynamic.constant import EulerGamma
from td_dynamic.karun.multiple.predictor_multiple import OracleActionState, PredictorBase
from td_dynamic.karun.multiple.simulate_multiple import EquilibriumMultiple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MLEModel(GenericLikelihoodModel):
    def __init__(self, endog, h, g, h_all, g_all, result, action_set, **kwargs):
        super().__init__(endog, **kwargs)
        self.h = h
        self.g = g
        self.h_all = h_all
        self.g_all = g_all
        self.result = result
        self.action_set = action_set

    def compute_denominator(self, x):
        denominator = [
            np.dot(self.h_all[i], x.reshape(-1, 1)) + self.g_all[i]
            for i in range(len(self.h_all))
        ]
        benchmark = [
            np.max(block)
            for block in denominator
        ]
        denominator = [
            np.exp(denominator[i] - benchmark[i])
            for i in range(len(self.h_all))
        ]
        denominator = [
            self.action_set[i].with_columns(pl.Series("denominator", denominator[i].flatten()))
            for i in range(len(self.h_all))
        ]
        denominator = [
            denominator[i].group_by(["m", "t"]).agg(pl.col("denominator").sum()).sort(["m", "t"])
            for i in range(len(self.h_all))
        ]
        return denominator, benchmark

    def compute_numerator(self, x, benchmark):
        numerator = [
            np.dot(self.h[i], x.reshape(-1, 1)) + self.g[i]
            for i in range(len(self.h))
        ]
        numerator = [
            np.exp(numerator[i] - benchmark[i])
            for i in range(len(self.h))
        ]
        numerator = [
            self.result.select(["m", "t"]).with_columns(pl.Series("numerator", numerator[i].flatten())).sort(["m", "t"])
            for i in range(len(self.h))
        ]
        return numerator

    def compute_likelihood_individual(self, numerator, denominator):
        likelihood = [
            numerator[i].join(denominator[i], on=["m", "t"], how="left").with_columns(
                likelihood=pl.col("numerator") / pl.col("denominator")
            )
            for i in range(len(numerator))
        ]
        return likelihood

    def compute_loglikelihood(self, params):
        denominator, benchmark = self.compute_denominator(x=params)
        numerator = self.compute_numerator(x=params, benchmark=benchmark)
        likelihood = self.compute_likelihood_individual(numerator=numerator, denominator=denominator)
        loglikelihood = [
            likelihood[i].select(pl.col("likelihood").log().sum())
            for i in range(len(likelihood))
        ]
        loglikelihood = sum(loglikelihood).item()
        return loglikelihood

    def loglike(self, params):
        return self.compute_loglikelihood(params)


class EstimatorForEquilibriumMultiple:
    def __init__(self, equilibrium: EquilibriumMultiple, ccp_predictor_list: list[PredictorBase], *args, **kwargs):
        self.equilibrium = equilibrium
        self.ccp_predictor_list = ccp_predictor_list
        self.h = None
        self.g = None
        self.h_all = None
        self.g_all = None

    @staticmethod
    def estimate_ccp_count(equilibrium):
        result = equilibrium.result
        ccp = [
            (   
                result.group_by([col for col in result.columns if col.startswith("state_index_")] + [f"action_index_{i}"])
                .agg(count=pl.col(f"action_index_{i}").count())
                .with_columns(
                    denominator=pl.col("count")
                    .sum()
                    .over([col for col in result.columns if col.startswith("state_index_")]),
                )
                .with_columns(ccp=pl.col("count") / pl.col("denominator"))
                .unique(subset=[col for col in result.columns if col.startswith("state_index_")] + [f"action_index_{i}"])
                .sort([col for col in result.columns if col.startswith("state_index_")][::-1] + [f"action_index_{i}"])
            )
            for i in range(equilibrium.num["firm"])
        ]
        return ccp

    @staticmethod
    def estimate_ccp(equilibrium, ccp_predictor_list):
        X = equilibrium.result.select(*[col for col in equilibrium.result.columns if col.startswith("state_value_")]).to_numpy()

        ccp = [
            ccp_predictor_list[i].fit(X, equilibrium.result.select(f"action_value_{i}").to_numpy().flatten())
            for i in range(len(ccp_predictor_list))
        ]

        return ccp
    
    @staticmethod
    def compute_conditional_expected_shock_from_result(equilibrium, ccp):
        result = equilibrium.result
        X = result.select(*[col for col in equilibrium.result.columns if col.startswith("state_value_")]).to_numpy()
        row_indices = np.arange(X.shape[0])

        e = []
        for i in range(len(ccp)):
            y = result.select(f"action_value_{i}").to_numpy().flatten()
            predicted = ccp[i].predict(X)
            predicted = predicted[row_indices, y.astype(int)]
            predicted = EulerGamma - np.log(predicted)
            predicted = predicted.reshape(-1, 1)
            e.append(predicted)

        e = [
            np.array(e_i)
            for e_i in e
        ]

        return e

    @staticmethod
    def make_selector(equilibrium):
        result = equilibrium.result
        boundary = result.group_by("m").agg([pl.col("t").max().alias("t_max"), pl.col("t").min().alias("t_min")])
        result_boundary = result.join(boundary, on="m")
        current_selector = (result_boundary["t"] != result_boundary["t_max"]).to_numpy()
        next_selector = (result_boundary["t"] != result_boundary["t_min"]).to_numpy()
        return current_selector, next_selector

    @staticmethod
    def compute_payoff_covariate(action_state, num_firm):
        covariate = [
            np.column_stack(
                [
                    action_state[:, -1] * action_state[:, i],
                    (action_state[:, i] - np.sum(action_state[:, :num_firm], axis=1)) * action_state[:, i],
                    -(1 - action_state[:, num_firm + i]) * action_state[:, i],
                ]
            )
            for i in range(num_firm)
        ]

        return covariate
    
    @staticmethod
    def make_action_set(result, num_action, num_firm):
        action_set = [
            result.select(["m", "t"]).join(
                pl.DataFrame({
                    f"action_index_{i}": range(num_action)
                }).with_columns(pl.lit(1).alias("_dummy")),
                how="cross"
            ).drop("_dummy")
            for i in range(num_firm)
        ]
        return action_set
    
    @staticmethod
    def expand_result_action(result, num_action, num_firm):
        action_set = EstimatorForEquilibriumMultiple.make_action_set(result=result, num_action=num_action, num_firm=num_firm)
        df = [
            action_set[i].join(
                result.select(
                    "m",
                    "t", 
                    pl.col("^state_value_.*$")
                ),
                on=["m", "t"],
                how="left"
            ).select(
                f"action_index_{i}",
                pl.col("^state_value_.*$")
            )
            for i in range(len(action_set))
        ]    
        return df
    
    def estimate_h(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_g(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_h_g_all(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_params(self, *args, **kwargs):
        raise NotImplementedError


class EstimatorForEquilibriumMultipleAVI(EstimatorForEquilibriumMultiple):
    def __init__(self, predictor_list: list[PredictorBase], ccp_predictor_list: list[PredictorBase], equilibrium: EquilibriumMultiple):
        super().__init__(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
        self.predictor_list = predictor_list

    def initialize_h_predictor(self, initial_h):
        # Prepare the data        
        action_state = self.equilibrium.result.select(
            *[col for col in self.equilibrium.result.columns if col.startswith("action_value_")],
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        num_firm = self.equilibrium.num["firm"]
        action_state = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]

        # Initialize a  list to store predictors
        predictor_list_h = [
            [deepcopy(self.predictor_list[i]) for _ in range(initial_h[i].shape[1])]
            for i in range(len(initial_h))
        ]
        predictor_list_h = self.update_predictor(predictor_list_h_or_g=predictor_list_h, X=action_state, Y=initial_h)
        
        return predictor_list_h

    def initialize_g_predictor(self, initial_g):
        # Prepare the data
        action_state = self.equilibrium.result.select(
            *[col for col in self.equilibrium.result.columns if col.startswith("action_value_")],
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        num_firm = self.equilibrium.num["firm"]
        action_state = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]

        # Initialize a list to store predictors
        predictor_list_g = [
            [deepcopy(self.predictor_list[i]) for _ in range(initial_g[i].shape[1])]
            for i in range(len(initial_g))
        ]
        predictor_list_g = self.update_predictor(predictor_list_h_or_g=predictor_list_g, X=action_state, Y=initial_g)
        return predictor_list_g

    def update_predictor(self, predictor_list_h_or_g, X, Y):
        # Loop through each column of Y
        for i in range(len(Y)): 
            for j in range(Y[i].shape[1]):
                y = np.array(Y[i][:, j])
                x = np.array(X[i])  
                # Train the predictor
                predictor = predictor_list_h_or_g[i][j]
                predictor.fit(x, y)
                # Calculate and print R-squared
                r_squared = predictor.score(x, y)
                print(f"predictor {j}: R-squared: {r_squared:.4f}")
                # Update the predictor in the list
                predictor_list_h_or_g[i][j] = predictor
        return predictor_list_h_or_g

    def update_h_predictor(
        self,
        predictor_list_h,
        action_state,
        payoff_covariate,
        current_selector,
        next_selector,
        discount,
    ):
        num_firm = len(predictor_list_h)
        
        action_state_list = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]
        
        payoff_covariate_current = [
            block[current_selector, :]
            for block in payoff_covariate
        ]

        action_state_current = [
            block[current_selector, :]
            for block in action_state_list
        ]
        action_state_next = [
            block[next_selector, :]
            for block in action_state_list
        ]

        y_pred_next = [
            [predictor.predict(action_state_next[i]) for predictor in predictor_list_h[i]]
            for i in range(len(predictor_list_h))
        ]
        y_pred_next = [
            np.column_stack(block)
            for block in y_pred_next
        ]
        Y = [
            payoff_covariate_current[i] + discount * y_pred_next[i]
            for i in range(len(payoff_covariate_current))
        ]
        X = action_state_current
        predictor_list_h = self.update_predictor(predictor_list_h_or_g=predictor_list_h, X=X, Y=Y)
        
        return predictor_list_h

    def update_g_predictor(self, predictor_list_g, action_state, e, current_selector, next_selector, discount):
        
        num_firm = len(predictor_list_g)
        
        action_state_list = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]

        e_next = [
            block[next_selector, :]
            for block in e
        ]

        action_state_current = [
            block[current_selector, :]
            for block in action_state_list
        ]
        action_state_next = [
            block[next_selector, :]
            for block in action_state_list
        ]

        y_pred_next = [
            [predictor.predict(action_state_next[i]) for predictor in predictor_list_g[i]]
            for i in range(len(predictor_list_g))
        ]
        y_pred_next = [
            np.column_stack(block)
            for block in y_pred_next
        ]
        Y = [
            discount * e_next[i] + discount * y_pred_next[i]
            for i in range(len(e_next))
        ]
        X = action_state_current
        predictor_list_g = self.update_predictor(predictor_list_h_or_g=predictor_list_g, X=X, Y=Y)

        return predictor_list_g

    def estimate_g(self, initial_g, num_iteration):
        action_state = self.equilibrium.result.select(
            *[col for col in self.equilibrium.result.columns if col.startswith("action_value_")],
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        num_firm = self.equilibrium.num["firm"]

        ccp = self.estimate_ccp(equilibrium=self.equilibrium, ccp_predictor_list=self.ccp_predictor_list)
        e = self.compute_conditional_expected_shock_from_result(equilibrium=self.equilibrium, ccp=ccp)

        current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

        predictor_list_g = self.initialize_g_predictor(initial_g=initial_g)
        
        for i in range(num_iteration):
            predictor_list_g = self.update_g_predictor(
                predictor_list_g=predictor_list_g,
                action_state=action_state,
                e=e,
                current_selector=current_selector,
                next_selector=next_selector,
                discount=self.equilibrium.discount,
            )

        action_state_list = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]

        g = [
            [predictor.predict(action_state_list[i]) for predictor in predictor_list_g[i]]
            for i in range(len(predictor_list_g))
        ]
        g = [
            np.column_stack(block)
            for block in g
        ]

        return g, predictor_list_g

    def estimate_h(self, initial_h, num_iteration):
        action_state = self.equilibrium.result.select(
            *[col for col in self.equilibrium.result.columns if col.startswith("action_value_")],
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        num_firm = self.equilibrium.num["firm"]

        payoff_covariate = self.compute_payoff_covariate(action_state=np.array(action_state), num_firm=num_firm)

        current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

        predictor_list_h = self.initialize_h_predictor(initial_h=initial_h)
        for k in range(num_iteration):
            predictor_list_h = self.update_h_predictor(
                predictor_list_h=predictor_list_h,
                action_state=action_state,
                payoff_covariate=payoff_covariate,
                current_selector=current_selector,
                next_selector=next_selector,
                discount=self.equilibrium.discount,
            )
            
        action_state_list = [
            np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
            for i in range(num_firm)
        ]

        h = [
            [predictor.predict(action_state_list[i]) for predictor in predictor_list_h[i]]
            for i in range(len(predictor_list_h))
        ]
        h = [
            np.column_stack(block)
            for block in h
        ]

        return h, predictor_list_h

    def estimate_h_g_all(self, predictor_list_h, predictor_list_g):
        df = self.expand_result_action(result=self.equilibrium.result, num_action=self.equilibrium.num["action"], num_firm=self.equilibrium.num["firm"])

        h_all = [
            [predictor.predict(np.array(df[i])) for predictor in predictor_list_h[i]]
            for i in range(len(predictor_list_h))
        ]
        h_all = [
            np.column_stack(block)
            for block in h_all
        ]
        
        g_all = [
            [predictor.predict(np.array(df[i])) for predictor in predictor_list_g[i]]
            for i in range(len(predictor_list_g))
        ]
        g_all = [
            np.column_stack(block)
            for block in g_all
        ]

        return h_all, g_all

    def estimate_params(self, initial_h, initial_g, num_iteration=10):
        start_params = np.array(
            [
                self.equilibrium.param["param_payoff"]["beta"],
                self.equilibrium.param["param_payoff"]["alpha"],
                self.equilibrium.param["param_payoff"]["lambda"],
            ]
        )
        action_set = self.make_action_set(result=self.equilibrium.result, num_action=self.equilibrium.num["action"], num_firm=self.equilibrium.num["firm"])
        
        print("Starting to estimate h")
        self.h, predictor_list_h = self.estimate_h(
            initial_h=initial_h,
            num_iteration=num_iteration,
        )
        print("Starting to estimate g")
        self.g, predictor_list_g = self.estimate_g(
            initial_g=initial_g,
            num_iteration=num_iteration,
        )
        print("Starting to estimate h_all and g_all")
        self.h_all, self.g_all = self.estimate_h_g_all(
            predictor_list_h=predictor_list_h,
            predictor_list_g=predictor_list_g,
        )

        print("Starting to estimate params")
        mle_model = MLEModel(
            endog=np.random.normal(size=self.equilibrium.result.shape[0]),
            h=self.h,
            g=self.g,
            h_all=self.h_all,
            g_all=self.g_all,
            result=self.equilibrium.result,
            action_set=action_set,
        )
        result = mle_model.fit(start_params=start_params)
        return result


class EstimatorForEquilibriumMultipleSemiGradient(EstimatorForEquilibriumMultiple):
    def __init__(self, equilibrium: EquilibriumMultiple, ccp_predictor_list: list[PredictorBase]):
        super().__init__(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
        self.omega = None
        self.xi = None
        self.basis = None

    @staticmethod
    def compute_state_value(equilibrium):
        past_action = np.arange(equilibrium.num["action"]).reshape(-1, 1)
        state = equilibrium.markov.state_values

        state_repeated = np.repeat(state, past_action.shape[0], axis=0)
        past_action_tiled = np.tile(past_action, (state.shape[0], 1))
        state_combined = np.hstack((past_action_tiled, state_repeated))

        state_index = pl.DataFrame(
            {f"state_{i}": pl.Series(state_combined[:, i]).rank("dense") - 1 for i in range(state_combined.shape[1])}
        )
        return state_combined, state_index

    @staticmethod
    def fit_basis_action_state(equilibrium, predictor_type="polynomial", degree=2):
        action_state = equilibrium.action_state["action_state_value"]
        num_firm = equilibrium.num["firm"]
        if predictor_type == "polynomial":
            basis = PolynomialFeatures(degree=degree)
            basis = [
                basis.fit(np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
                for i in range(num_firm)
            ]
            return basis
        elif predictor_type == "oracle":
            basis = [
                OracleActionState(action_state=np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
                for i in range(num_firm)
            ]
            return basis
        else:
            raise ValueError("Invalid basis predictor_type")

    @staticmethod
    def estimate_omega(
        payoff_covariate,
        basis_action_state,
        current_selector,
        next_selector,
        discount,
    ):
        basis_action_state_current = [
            block[current_selector, :]
            for block in basis_action_state
        ]

        basis_action_state_next = [
            block[next_selector, :]
            for block in basis_action_state
        ]

        payoff_covariate_current = [
            block[current_selector, :]
            for block in payoff_covariate
        ]

        term_1 = [
            np.dot(
                basis_action_state_current[i].T,
                basis_action_state_current[i] - discount * basis_action_state_next[i]
            )
            for i in range(len(basis_action_state_current))
        ]

        term_2 = [
            np.dot(basis_action_state_current[i].T, payoff_covariate_current[i])
            for i in range(len(basis_action_state_current))
        ]

        omega = [
            np.linalg.solve(term_1[i], term_2[i])
            for i in range(len(basis_action_state_current))
        ]

        return omega

    @staticmethod
    def estimate_xi(e, basis_action_state, current_selector, next_selector, discount):
        basis_action_state_current = [
        block[current_selector, :]
        for block in basis_action_state
        ]
        basis_action_state_next = [
        block[next_selector, :]
        for block in basis_action_state
        ]
        e_next = [
        block[next_selector]
        for block in e
        ]

        term_1 = [
            np.dot(
                basis_action_state_current[i].T,
                basis_action_state_current[i] - discount * basis_action_state_next[i],
            )
            for i in range(len(basis_action_state_current))
        ]
        term_2 = [
            discount * np.dot(basis_action_state_current[i].T, e_next[i])
            for i in range(len(basis_action_state_current))
        ]

        xi = [
            np.linalg.solve(term_1[i], term_2[i])
            for i in range(len(term_1))
        ]
        return xi

    @staticmethod
    def transform_basis_action_state(result, basis):
        action_state = result.select(
            *[col for col in result.columns if col.startswith("action_value_")],
            *[col for col in result.columns if col.startswith("state_value_")],
        )
        action_state = np.array(action_state)
        action_state = [
            action_state[:, [i, *range(len(basis), action_state.shape[1])]]
            for i in range(len(basis))
        ]
        basis_action_state = [
            basis[i].transform(action_state[i])
            for i in range(len(basis))
        ]
        return basis_action_state

    @staticmethod
    def compute_payoff_covariate_from_result(equilibrium):
        result = equilibrium.result
        action_state = result.select(
            *[col for col in result.columns if col.startswith("action_value_")],
            *[col for col in result.columns if col.startswith("state_value_")],
        )
        action_state = np.array(action_state)
        payoff_covariate = EstimatorForEquilibriumMultiple.compute_payoff_covariate(action_state=action_state, num_firm = equilibrium.num["firm"])
        return payoff_covariate

    def estimate_h(self, predictor_type="polynomial", degree=2):
        self.basis = self.fit_basis_action_state(
            equilibrium=self.equilibrium, predictor_type=predictor_type, degree=degree
        )
        basis_action_state = self.transform_basis_action_state(result=self.equilibrium.result, basis=self.basis)
        current_selector, next_selector = self.make_selector(self.equilibrium)

        payoff_covariate = self.compute_payoff_covariate_from_result(equilibrium=self.equilibrium)

        self.omega = self.estimate_omega(
            payoff_covariate=payoff_covariate,
            basis_action_state=basis_action_state,
            current_selector=current_selector,
            next_selector=next_selector,
            discount=self.equilibrium.discount,
        )

        self.h = [
            np.dot(basis_action_state[i], self.omega[i])
            for i in range(len(basis_action_state))
        ]

        return self.h, self.omega, self.basis

    def estimate_g(self, predictor_type="polynomial", degree=2):
        self.basis = self.fit_basis_action_state(
            equilibrium=self.equilibrium, predictor_type=predictor_type, degree=degree
        )
        basis_action_state = self.transform_basis_action_state(result=self.equilibrium.result, basis=self.basis)
        current_selector, next_selector = self.make_selector(self.equilibrium)

        ccp = self.estimate_ccp(equilibrium=self.equilibrium, ccp_predictor_list=self.ccp_predictor_list)
        e = self.compute_conditional_expected_shock_from_result(equilibrium=self.equilibrium, ccp=ccp)

        self.xi = self.estimate_xi(
            e=e,
            basis_action_state=basis_action_state,
            current_selector=current_selector,
            next_selector=next_selector,
            discount=self.equilibrium.discount,
        )

        self.g = [
            np.dot(basis_action_state[i], self.xi[i])
            for i in range(len(basis_action_state))
        ]
        return self.g, self.xi, self.basis

    def estimate_h_g_all(self):
        df = self.expand_result_action(result=self.equilibrium.result, num_action=self.equilibrium.num["action"], num_firm=self.equilibrium.num["firm"])

        basis_action_state = [
            self.basis[i].transform(np.array(df[i]))
            for i in range(len(self.basis))
        ]
        h_all = [
            np.dot(basis_action_state[i], self.omega[i])
            for i in range(len(basis_action_state))
        ]
        g_all = [
            np.dot(basis_action_state[i], self.xi[i])
            for i in range(len(basis_action_state))
        ]

        return h_all, g_all

    def estimate_params(self, predictor_type, degree):
        start_params = np.array(
            [
                self.equilibrium.param["param_payoff"]["beta"],
                self.equilibrium.param["param_payoff"]["alpha"],
                self.equilibrium.param["param_payoff"]["lambda"],
            ]
        )
        action_set = self.make_action_set(result=self.equilibrium.result, num_action=self.equilibrium.num["action"], num_firm=self.equilibrium.num["firm"])
        
        print("Starting to estimate h")
        self.h, _, _ = self.estimate_h(
            predictor_type=predictor_type,
            degree=degree,
        )
        print("Starting to estimate g")
        self.g, _, _ = self.estimate_g(
            predictor_type=predictor_type,
            degree=degree,
        )
        print("Starting to estimate h_all and g_all")
        h_all, g_all = self.estimate_h_g_all()

        print("Starting to estimate params")
        mle_model = MLEModel(
            endog=np.random.normal(size=self.equilibrium.result.shape[0]),
            h=self.h,
            g=self.g,
            h_all=h_all,
            g_all=g_all,
            result=self.equilibrium.result,
            action_set=action_set,
        )
        result = mle_model.fit(start_params=start_params)
        return result
