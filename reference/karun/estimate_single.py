# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from td_dynamic.karun.single.estimate_single import (
    EstimatorForEquilibriumSingleAVI,
    EstimatorForEquilibriumSingleSemiGradient,
    MLEModel,
)
import polars as pl
from td_dynamic.karun.single.predictor_single import LinearRegressionPredictor
from td_dynamic.karun.utils import read_pickle_from_s3
# set constants ----------------------------------------------------

EulerGamma = 0.577215664901532860606512090082402431042159336
degree = 0
type = "oracle"
num_iteration = 10

prefix = "output/estimate_single_objective/"
bucket_name = "football-markov"

# load data --------------------------------------------------------

equilibrium = read_pickle_from_s3(
    bucket=bucket_name, prefix="output/simulate_single/", file_name="equilibrium.pkl"
)

# initialize predictor and estimators ----------------------------

predictor = LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=2)
avi_estimator = EstimatorForEquilibriumSingleAVI(equilibrium=equilibrium, predictor=predictor)
semi_gradient_estimator = EstimatorForEquilibriumSingleSemiGradient(equilibrium=equilibrium)
result_semi_gradient = semi_gradient_estimator.estimate_params(predictor_type="polynomial", degree=2)

# estimate h and g --------------------------------------------------

h, h_model_list = avi_estimator.estimate_h(
    initial_h=semi_gradient_estimator.h,
    num_iteration=num_iteration,
)
g, g_model_list = avi_estimator.estimate_g(
    initial_g=semi_gradient_estimator.g,
    num_iteration=num_iteration,
)

h_all, g_all = avi_estimator.estimate_h_g_all(
    h_predictor_list=h_model_list,
    g_predictor_list=g_model_list,
)

# estimate the parameter --------------------------------------------

x = np.array(
    [
        equilibrium.param["param_payoff"]["beta"],
        equilibrium.param["param_payoff"]["lambda"],
    ]
)

mle_model = MLEModel(
    endog=np.random.normal(size=equilibrium.result.shape[0]),
    h=h,
    g=g,
    h_all=h_all,
    g_all=g_all,
    result=equilibrium.result,
    action_set=equilibrium.action_set,
)

result_mle = mle_model.fit(start_params=x)
result_mle.summary()
