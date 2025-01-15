# set up environment -----------------------------------------------

from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultipleAVI,
    EstimatorForEquilibriumMultipleSemiGradient,
)
from td_dynamic.karun.multiple.predictor_multiple import (
  LinearRegressionPredictor,
  CCPLogisticRegressionPredictor
)
from td_dynamic.karun.utils import read_pickle_from_local

# set constants ----------------------------------------------------

degree = 0
predictor_type = "oracle"
num_iteration = 1

# load data --------------------------------------------------------

equilibrium = read_pickle_from_local("output/simulate_multiple/equilibrium.pkl")

# estimate by semi-gradient ----------------------------------------
ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type=predictor_type, degree=degree)
    for i in range(equilibrium.num["firm"])
]
semi_gradient_estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)
result_semi = semi_gradient_estimator.estimate_params(predictor_type=predictor_type, degree=degree)

# estimate by avi --------------------------------------------------

predictor_list = [
    LinearRegressionPredictor(equilibrium=equilibrium, predictor_type=predictor_type, degree=degree, i=i)
  for i in range(equilibrium.num["firm"])  
]
avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=predictor_list, ccp_predictor_list=ccp_predictor_list)
result_avi = avi_estimator.estimate_params(initial_h=semi_gradient_estimator.h, initial_g=semi_gradient_estimator.g, num_iteration=num_iteration)

