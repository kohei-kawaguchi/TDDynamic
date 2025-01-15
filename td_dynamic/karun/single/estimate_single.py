import logging
from copy import deepcopy

import numpy as np
import polars as pl
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.base.model import GenericLikelihoodModel

from td_dynamic.constant import EulerGamma
from td_dynamic.karun.single.predictor_single import OracleActionState, PredictorBase
from td_dynamic.karun.single.simulate_single import EquilibriumSingle

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
        denominator = np.dot(self.h_all, x.reshape(-1, 1)) + self.g_all
        benchmark = np.max(denominator)
        denominator = np.exp(denominator - benchmark)
        denominator = self.action_set.with_columns(pl.Series("denominator", denominator.flatten()))
        denominator = denominator.group_by(["i", "t"]).agg(pl.col("denominator").sum()).sort(["i", "t"])
        return denominator, benchmark

    def compute_numerator(self, x, benchmark):
        numerator = np.dot(self.h, x.reshape(-1, 1)) + self.g
        numerator = np.exp(numerator - benchmark)
        numerator = (
            self.result.select(["i", "t"]).with_columns(pl.Series("numerator", numerator.flatten())).sort(["i", "t"])
        )
        return numerator

    def compute_likelihood_individual(self, numerator, denominator):
        likelihood = numerator.join(denominator, on=["i", "t"], how="left").with_columns(
            likelihood=pl.col("numerator") / pl.col("denominator")
        )
        return likelihood

    def compute_loglikelihood(self, params):
        denominator, benchmark = self.compute_denominator(x=params)
        numerator = self.compute_numerator(x=params, benchmark=benchmark)
        likelihood = self.compute_likelihood_individual(numerator=numerator, denominator=denominator)
        loglikelihood = likelihood.select(pl.col("likelihood").log().sum())
        return loglikelihood.item()

    def loglike(self, params):
        return self.compute_loglikelihood(params)


class EstimatorForEquilibriumSingle:
    def __init__(self, equilibrium: EquilibriumSingle, *args, **kwargs):
        self.equilibrium = equilibrium
        self.h = None
        self.g = None
        self.h_all = None
        self.g_all = None

    @staticmethod
    def estimate_ccp_count(equilibrium):
        result = equilibrium.result
        ccp = (
            result.group_by([col for col in result.columns if col.startswith("state_value_")] + ["action_index"])
            .agg(count=pl.col("action_index").count())
            .with_columns(
                denominator=pl.col("count")
                .sum()
                .over([col for col in result.columns if col.startswith("state_value_")])
            )
            .with_columns(ccp=pl.col("count") / pl.col("denominator"))
            .sort([col for col in result.columns if col.startswith("state_value_")] + ["action_index"])
        )
        return ccp

    @staticmethod
    def compute_conditional_expected_shock_from_result(equilibrium, ccp):
        result = equilibrium.result
        e = (
            result.join(
                ccp,
                on=[col for col in result.columns if col.startswith("state_value_")] + ["action_index"],
                how="left",
            )
            .with_columns(e=EulerGamma - np.log(pl.col("ccp")))
            .select("e")
        )
        e = np.array(e)
        return e

    @staticmethod
    def make_selector(equilibrium):
        result = equilibrium.result
        boundary = result.group_by("i").agg([pl.col("t").max().alias("t_max"), pl.col("t").min().alias("t_min")])
        result_boundary = result.join(boundary, on="i")
        current_selector = (result_boundary["t"] != result_boundary["t_max"]).to_numpy()
        next_selector = (result_boundary["t"] != result_boundary["t_min"]).to_numpy()
        return current_selector, next_selector

    @staticmethod
    def compute_payoff_covariate(action_state):
        covariate = np.column_stack(
            [
                action_state[:, 2] * action_state[:, 0],
                -(1 - action_state[:, 1]) * action_state[:, 0],
            ]
        )
        return covariate

    def estimate_h(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_g(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_h_g_all(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_params(self, *args, **kwargs):
        raise NotImplementedError


class EstimatorForEquilibriumSingleAVI(EstimatorForEquilibriumSingle):
    def __init__(self, predictor: PredictorBase, equilibrium: EquilibriumSingle):
        super().__init__(equilibrium=equilibrium)
        self.predictor = predictor

    def initialize_h_predictor(self, initial_h):
        # Prepare the data
        action_state = self.equilibrium.result.select(
            "action_index",
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        action_state = np.array(action_state)
        # Initialize a list to store predictors
        predictor_list = [deepcopy(self.predictor) for _ in range(initial_h.shape[1])]
        predictor_list = self.update_predictor(predictor_list=predictor_list, X=action_state, Y=initial_h)
        return predictor_list

    def initialize_g_predictor(self, initial_g):
        # Prepare the data
        action_state = self.equilibrium.result.select(
            "action_index",
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        action_state = np.array(action_state)
        # Initialize a list to store predictors
        predictor_list = [deepcopy(self.predictor) for _ in range(initial_g.shape[1])]
        predictor_list = self.update_predictor(predictor_list=predictor_list, X=action_state, Y=initial_g)
        return predictor_list

    def update_predictor(self, predictor_list, X, Y):
        # Loop through each column of Y
        for i in range(Y.shape[1]):
            y = np.array(Y)[:, i]
            # Train the predictor
            predictor = predictor_list[i]
            predictor.fit(X, y)
            # Calculate and print R-squared
            r_squared = predictor.score(X, y)
            print(f"predictor {i}: R-squared: {r_squared:.4f}")
            # Update the predictor in the list
            predictor_list[i] = predictor
        return predictor_list

    def update_h_predictor(
        self,
        predictor_list,
        action_state,
        payoff_covariate,
        current_selector,
        next_selector,
        discount,
    ):
        payoff_covariate_current = payoff_covariate[current_selector, :]
        action_state_current = np.array(action_state)[current_selector, :]
        action_state_next = np.array(action_state)[next_selector, :]

        y_pred_next = [predictor.predict(action_state_next) for predictor in predictor_list]
        y_pred_next = np.array(y_pred_next).T

        Y = payoff_covariate_current + discount * y_pred_next
        X = action_state_current
        predictor_list = self.update_predictor(predictor_list=predictor_list, X=X, Y=Y)
        return predictor_list

    def update_g_predictor(self, predictor_list, action_state, e, current_selector, next_selector, discount):
        e_next = e[next_selector, :]
        action_state_current = np.array(action_state)[current_selector, :]
        action_state_next = np.array(action_state)[next_selector, :]

        y_pred_next = [predictor.predict(action_state_next) for predictor in predictor_list]
        y_pred_next = np.array(y_pred_next).T

        Y = discount * e_next + discount * y_pred_next
        X = action_state_current

        predictor_list = self.update_predictor(predictor_list=predictor_list, X=X, Y=Y)

        return predictor_list

    def estimate_g(self, initial_g, num_iteration):
        action_state = self.equilibrium.result.select(
            "action_index",
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )
        ccp = self.estimate_ccp_count(equilibrium=self.equilibrium)
        e = self.compute_conditional_expected_shock_from_result(equilibrium=self.equilibrium, ccp=ccp)
        current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

        predictor_list = self.initialize_g_predictor(initial_g=initial_g)
        for i in range(num_iteration):
            predictor_list = self.update_g_predictor(
                predictor_list=predictor_list,
                action_state=action_state,
                e=e,
                current_selector=current_selector,
                next_selector=next_selector,
                discount=self.equilibrium.discount,
            )

        g = [predictor.predict(np.array(action_state)) for predictor in predictor_list]
        g = np.array(g).T

        return g, predictor_list

    def estimate_h(self, initial_h, num_iteration):
        action_state = self.equilibrium.result.select(
            "action_index",
            *[col for col in self.equilibrium.result.columns if col.startswith("state_value_")],
        )

        payoff_covariate = self.compute_payoff_covariate(action_state=np.array(action_state))

        current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

        predictor_list = self.initialize_h_predictor(initial_h=initial_h)
        for i in range(num_iteration):
            predictor_list = self.update_h_predictor(
                predictor_list=predictor_list,
                action_state=action_state,
                payoff_covariate=payoff_covariate,
                current_selector=current_selector,
                next_selector=next_selector,
                discount=self.equilibrium.discount,
            )

        h = [predictor.predict(np.array(action_state)) for predictor in predictor_list]
        h = np.array(h).T

        return h, predictor_list

    def estimate_h_g_all(self, h_predictor_list, g_predictor_list):
        df = self.equilibrium.action_set.join(
            self.equilibrium.result.select(
                "i",
                "t",
                pl.col("action_index").alias("action_chosen"),
                pl.col("^state_value_.*$"),
            ),
            on=["i", "t"],
            how="left",
        )

        action_state_all = df.select(
            "action_index",
            *[col for col in df.columns if col.startswith("state_value_")],
        )

        h_all = [predictor.predict(np.array(action_state_all)) for predictor in h_predictor_list]
        h_all = np.array(h_all).T

        g_all = [predictor.predict(np.array(action_state_all)) for predictor in g_predictor_list]
        g_all = np.array(g_all).T

        return h_all, g_all

    def estimate_params(self, initial_h, initial_g, num_iteration=10):
        start_params = np.array(
            [
                self.equilibrium.param["param_payoff"]["beta"],
                self.equilibrium.param["param_payoff"]["lambda"],
            ]
        )
        print("Starting to estimate h")
        self.h, h_predictor_list = self.estimate_h(
            initial_h=initial_h,
            num_iteration=num_iteration,
        )
        print("Starting to estimate g")
        self.g, g_predictor_list = self.estimate_g(
            initial_g=initial_g,
            num_iteration=num_iteration,
        )
        print("Starting to estimate h_all and g_all")
        self.h_all, self.g_all = self.estimate_h_g_all(
            h_predictor_list=h_predictor_list,
            g_predictor_list=g_predictor_list,
        )

        print("Starting to estimate params")
        mle_model = MLEModel(
            endog=np.random.normal(size=self.equilibrium.result.shape[0]),
            h=self.h,
            g=self.g,
            h_all=self.h_all,
            g_all=self.g_all,
            result=self.equilibrium.result,
            action_set=self.equilibrium.action_set,
        )
        result = mle_model.fit(start_params=start_params)
        return result


class EstimatorForEquilibriumSingleSemiGradient(EstimatorForEquilibriumSingle):
    def __init__(self, equilibrium: EquilibriumSingle):
        super().__init__(equilibrium=equilibrium)
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
        if predictor_type == "polynomial":
            basis = PolynomialFeatures(degree=degree)
            basis = basis.fit(equilibrium.action_state["action_state_value"])
            return basis
        elif predictor_type == "oracle":
            basis = OracleActionState(action_state=np.array(equilibrium.action_state["action_state_value"]))
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
        basis_action_state_current = basis_action_state[current_selector, :]
        basis_action_state_next = basis_action_state[next_selector, :]
        payoff_covariate_current = payoff_covariate[current_selector, :]
        term_1 = np.dot(
            basis_action_state_current.T,
            basis_action_state_current - discount * basis_action_state_next,
        )
        term_2 = np.dot(basis_action_state_current.T, payoff_covariate_current)
        omega = np.linalg.solve(term_1, term_2)
        return omega

    @staticmethod
    def estimate_xi(e, basis_action_state, current_selector, next_selector, discount):
        basis_action_state_current = basis_action_state[current_selector, :]
        basis_action_state_next = basis_action_state[next_selector, :]
        e_next = e[next_selector]

        term_1 = np.dot(
            basis_action_state_current.T,
            basis_action_state_current - discount * basis_action_state_next,
        )
        term_2 = discount * np.dot(basis_action_state_current.T, e_next)
        xi = np.linalg.solve(term_1, term_2)
        return xi

    @staticmethod
    def transform_basis_action_state(result, basis):
        action_state = result.select(
            "action_index",
            *[col for col in result.columns if col.startswith("state_value_")],
        )
        basis_action_state = basis.transform(np.array(action_state))
        return basis_action_state

    @staticmethod
    def compute_payoff_covariate_from_result(equilibrium):
        result = equilibrium.result
        action_state = result.select(
            "action_index",
            *[col for col in result.columns if col.startswith("state_value_")],
        )
        action_state = np.array(action_state)
        payoff_covariate = EstimatorForEquilibriumSingle.compute_payoff_covariate(action_state=action_state)
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

        self.h = np.dot(basis_action_state, self.omega)
        return self.h, self.omega, self.basis

    def estimate_g(self, predictor_type="polynomial", degree=2):
        self.basis = self.fit_basis_action_state(
            equilibrium=self.equilibrium, predictor_type=predictor_type, degree=degree
        )
        basis_action_state = self.transform_basis_action_state(result=self.equilibrium.result, basis=self.basis)
        current_selector, next_selector = self.make_selector(self.equilibrium)

        ccp = self.estimate_ccp_count(self.equilibrium)
        e = self.compute_conditional_expected_shock_from_result(self.equilibrium, ccp=ccp)

        self.xi = self.estimate_xi(
            e=e,
            basis_action_state=basis_action_state,
            current_selector=current_selector,
            next_selector=next_selector,
            discount=self.equilibrium.discount,
        )

        self.g = np.dot(basis_action_state, self.xi)
        return self.g, self.xi, self.basis

    def estimate_h_g_all(self):
        df = self.equilibrium.action_set.join(
            self.equilibrium.result.select(
                "i",
                "t",
                pl.col("action_index").alias("action_chosen"),
                pl.col("^state_value_.*$"),
            ),
            on=["i", "t"],
            how="left",
        )

        action_state_all = df.select(
            "action_index",
            *[col for col in df.columns if col.startswith("state_value_")],
        )
        basis_action_state_all = self.transform_basis_action_state(result=action_state_all, basis=self.basis)
        self.h_all = np.dot(basis_action_state_all, self.omega)
        self.g_all = np.dot(basis_action_state_all, self.xi)

        return self.h_all, self.g_all

    def estimate_params(self, predictor_type, degree):
        self.estimate_h(predictor_type=predictor_type, degree=degree)
        self.estimate_g(predictor_type=predictor_type, degree=degree)
        self.estimate_h_g_all()

        # MLE
        mle_model = MLEModel(
            endog=np.random.normal(size=self.equilibrium.result.shape[0]),
            h=self.h,
            g=self.g,
            h_all=self.h_all,
            g_all=self.g_all,
            result=self.equilibrium.result,
            action_set=self.equilibrium.action_set,
        )
        start_params = np.array(
            [
                self.equilibrium.param["param_payoff"]["beta"],
                self.equilibrium.param["param_payoff"]["lambda"],
            ]
        )
        result = mle_model.fit(start_params=start_params)
        return result
