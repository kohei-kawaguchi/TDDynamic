# set up environment -----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from td_dynamic.karun.multiple.estimate_multiple import (
    EstimatorForEquilibriumMultiple,
    EstimatorForEquilibriumMultipleAVI,
    EstimatorForEquilibriumMultipleSemiGradient,
)
from td_dynamic.karun.multiple.predictor_multiple import (
    MLPPredictor,
    LinearRegressionPredictor,
    CCPLogisticRegressionPredictor,
)
from td_dynamic.karun.utils import read_pickle_from_local
import torch
from copy import deepcopy

# set constants ----------------------------------------------------

predictor_type = "polynomial"
degree = 1
num_iteration = 2

# load data --------------------------------------------------------

equilibrium = read_pickle_from_local("output/simulate_multiple/equilibrium.pkl")
predictor_list = [
  LinearRegressionPredictor(equilibrium=equilibrium, predictor_type=predictor_type, degree=degree, i=i)
  for i in range(equilibrium.num["firm"])
]
# predictor_list = [
#     MLPPredictor(
#         equilibrium=equilibrium,
#         predictor_type="oracle",
#         degree=2,
#         hidden_layer_sizes=(16,),
#         learning_rate=0.001,
#         batch_size=256,
#         num_epochs=2,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         i=i,
#     )
#   for i in range(equilibrium.num["firm"])
# ]
ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type=predictor_type, degree=1)
    for i in range(equilibrium.num["firm"])
]
avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=predictor_list, ccp_predictor_list=ccp_predictor_list)

oracle_predictor_list = [
    LinearRegressionPredictor(equilibrium=equilibrium, predictor_type="oracle", degree=0, i=i)
  for i in range(equilibrium.num["firm"])  
]
oracle_ccp_predictor_list = [
    CCPLogisticRegressionPredictor(equilibrium=equilibrium, ccp_predictor_type="oracle", degree=0)
    for i in range(equilibrium.num["firm"])
]
oracle_avi_estimator = EstimatorForEquilibriumMultipleAVI(equilibrium=equilibrium, predictor_list=oracle_predictor_list, ccp_predictor_list=oracle_ccp_predictor_list)

# estimate using semi gradient method ------------------------------

semi_gradient_estimator = EstimatorForEquilibriumMultipleSemiGradient(equilibrium=equilibrium, ccp_predictor_list=ccp_predictor_list)

initial_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type=predictor_type, degree=degree)
initial_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type=predictor_type, degree=degree)

# estimate h with avi -------------------------------------------------

self = avi_estimator
result = self.equilibrium.result
num_firm = self.equilibrium.num["firm"]

action_state = result.select(
    *[col for col in result.columns if col.startswith("action_value_")],
    *[col for col in result.columns if col.startswith("state_value_")],
)
action_state = [
    np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
    for i in range(num_firm)
]

# Initialize a list to store predictors
predictor_list_h = [
  [deepcopy(self.predictor_list[i]) for _ in range(initial_h[i].shape[1])]
  for i in range(len(initial_h))
]

# update_predictor

X = action_state
Y = initial_h

i = 0
j = 0
y = np.array(Y[i][:, j])
x = np.array(X[i])  
# Train the predictor
predictor = predictor_list_h[i][j]
predictor.fit(x, y)
# Calculate and print R-squared
r_squared = predictor.score(x, y)
print(f"predictor {j}: R-squared: {r_squared:.4f}")
# Update the predictor in the list
predictor_list_h[i][j] = predictor

# Loop through each column of Y
predictor_list_h = avi_estimator.update_predictor(predictor_list_h_or_g=predictor_list_h, X=X, Y=Y)

# initialize_h_predictor
predictor_list_h = avi_estimator.initialize_h_predictor(initial_h=initial_h)

# update_h_predictor

self = avi_estimator
num_firm = self.equilibrium.num["firm"]

action_state = equilibrium.result.select(
    *[col for col in equilibrium.result.columns if col.startswith("action_value_")],
    *[col for col in equilibrium.result.columns if col.startswith("state_value_")],
)

payoff_covariate = self.compute_payoff_covariate(action_state=np.array(action_state), num_firm=num_firm)

current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

predictor_list_h = self.initialize_h_predictor(initial_h=initial_h)

predictor_list_h = self.update_h_predictor(
    predictor_list_h=predictor_list_h,
    action_state=action_state,
    payoff_covariate=payoff_covariate,
    current_selector=current_selector,
    next_selector=next_selector,
    discount=self.equilibrium.discount,
)

# estimate_h

h, h_predictor_list = avi_estimator.estimate_h(initial_h=initial_h, num_iteration=num_iteration)

# compare linear regression h with output from semi gradient method 

oracle_h, _, _ = semi_gradient_estimator.estimate_h(predictor_type="oracle", degree=0)

# Flatten h and oracle_h for scatter plot
# they may not perfectly match

for i in range(len(h)):
    h_flat = h[i].flatten()
    oracle_h_flat = oracle_h[i].flatten()

    plt.figure()
    plt.scatter(oracle_h_flat, h_flat)
    plt.xlabel("OracleActionState h Values")
    plt.ylabel("Estimated h Values")
    plt.title("Scatter Plot between OracleActionState h and Estimated h Values")
    plt.plot(
        [oracle_h_flat.min(), oracle_h_flat.max()],
        [oracle_h_flat.min(), oracle_h_flat.max()],
        "k--",
        lw=1,
    )
    plt.show()

# compare oracle h with output from semi gradient method 

h, predictor_list = oracle_avi_estimator.estimate_h(initial_h=oracle_h, num_iteration=num_iteration)

# Flatten h and oracle_h for scatter plot
# perfect match is expected

for i in range(len(h)):
    h_flat = h[i].flatten()
    oracle_h_flat = oracle_h[i].flatten()

    plt.figure()
    plt.scatter(oracle_h_flat, h_flat)
    plt.xlabel("OracleActionState h Values")
    plt.ylabel("Estimated h Values")
    plt.title("Scatter Plot between OracleActionState h and Estimated h Values")
    plt.plot(
        [oracle_h_flat.min(), oracle_h_flat.max()],
        [oracle_h_flat.min(), oracle_h_flat.max()],
        "k--",
        lw=1,
    )
    plt.show()


# estimate g with avi -------------------------------------------------

self = avi_estimator
result = self.equilibrium.result
num_firm = self.equilibrium.num["firm"]

action_state = result.select(
    *[col for col in result.columns if col.startswith("action_value_")],
    *[col for col in result.columns if col.startswith("state_value_")],
)
action_state = [
    np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
    for i in range(num_firm)
]

# Initialize a list to store predictors
predictor_list_g = [
  [deepcopy(self.predictor_list[i]) for _ in range(initial_g[i].shape[1])]
  for i in range(len(initial_g))
]

# update_predictor

X = action_state
Y = initial_g

i = 0
j = 0
y = np.array(Y[i][:, j])
x = np.array(X[i])  
# Train the predictor
predictor = predictor_list_g[i][j]
predictor.fit(x, y)
# Calculate and print R-squared
r_squared = predictor.score(x, y)
print(f"predictor {j}: R-squared: {r_squared:.4f}")
# Update the predictor in the list
predictor_list_g[i][j] = predictor

# Loop through each column of Y
predictor_list_g = avi_estimator.update_predictor(predictor_list_h_or_g=predictor_list_g, X=X, Y=Y)

# initialize_g_predictor
predictor_list_g = avi_estimator.initialize_g_predictor(initial_g=initial_g)

# update_g_predictor

self = avi_estimator
num_firm = self.equilibrium.num["firm"]
discount = self.equilibrium.discount

action_state = equilibrium.result.select(
    *[col for col in equilibrium.result.columns if col.startswith("action_value_")],
    *[col for col in equilibrium.result.columns if col.startswith("state_value_")],
)

action_state_list = [
    np.array(action_state[:, [i, *range(num_firm, action_state.shape[1])]])
    for i in range(num_firm)
]

ccp = self.estimate_ccp(equilibrium=self.equilibrium, ccp_predictor_list=self.ccp_predictor_list)
e = self.compute_conditional_expected_shock_from_result(equilibrium=self.equilibrium, ccp=ccp)

current_selector, next_selector = self.make_selector(equilibrium=self.equilibrium)

predictor_list_g = self.initialize_g_predictor(initial_g=initial_g)

predictor_list_g = avi_estimator.update_g_predictor(predictor_list_g=predictor_list_g, action_state=action_state, e=e, current_selector=current_selector, next_selector=next_selector, discount=avi_estimator.equilibrium.discount)

# estimate_g

g, g_predictor_list = avi_estimator.estimate_g(initial_g=initial_g, num_iteration=num_iteration)

# Assuming oracle_g is already defined or computed elsewhere in the code
oracle_g, _, _ = semi_gradient_estimator.estimate_g(predictor_type="oracle", degree=0)

# Flatten g and oracle_g for scatter plot
# they may not perfectly match

for i in range(len(g)):
    g_flat = g[i].flatten()
    oracle_g_flat = oracle_g[i].flatten()

    # 
    plt.figure()
    plt.scatter(oracle_g_flat, g_flat)
    plt.xlabel("OracleActionState g Values")
    plt.ylabel("Estimated g Values")
    plt.title(f"Scatter Plot between OracleActionState g and Estimated g Values for Firm {i}")
    plt.plot(
        [oracle_g_flat.min(), oracle_g_flat.max()],
        [oracle_g_flat.min(), oracle_g_flat.max()],
        "k--",
        lw=1,
    )
    plt.show()
    
# compare oracle h with output from semi gradient method 

g, predictor_list = oracle_avi_estimator.estimate_g(initial_g=oracle_g, num_iteration=num_iteration)

# Flatten g and oracle_g for scatter plot
# perfect match is expected

for i in range(len(g)):
    g_flat = g[i].flatten()
    oracle_g_flat = oracle_g[i].flatten()

    plt.figure()
    plt.scatter(oracle_g_flat, g_flat)
    plt.xlabel("OracleActionState g Values")
    plt.ylabel("Estimated g Values")
    plt.title("Scatter Plot between OracleActionState g and Estimated g Values")
    plt.plot(
        [oracle_g_flat.min(), oracle_g_flat.max()],
        [oracle_g_flat.min(), oracle_g_flat.max()],
        "k--",
        lw=1,
    )
    plt.show()
