# set up environment ----------------------------------------------

import numpy as np
from td_dynamic.karun.single.simulate_single import EquilibriumSingleEntryExit
from td_dynamic.karun.utils import create_s3_folder_if_not_exists, write_pickle_to_s3

# set constants ----------------------------------------------------

prefix = "output/simulate_single/"
bucket_name = "football-markov"

create_s3_folder_if_not_exists(bucket=bucket_name, folder=prefix)

num = {"firm": 10000, "period": 10, "state": 1, "action": 2, "tolerance": 1e-6}

num["grid_size"] = np.full(num["state"], 3)

# set parameters ----------------------------------------------------

param = {
    "param_payoff": {"beta": 1, "lambda": 5},
    "param_state": {
        "state_constant": np.concatenate((np.full(num["state"], 0.0),)),
        "state_trans": np.eye(num["state"]) * 0.1,
        "state_sd": np.eye(num["state"]),
    },
}

discount = 0.95

# solve the equilibrium -------------------------------------------

model = EquilibriumSingleEntryExit(discount=discount, num=num, param=param)

model.simulate_equilibrium()

write_pickle_to_s3(obj=model, bucket=bucket_name, prefix=prefix, file_name="equilibrium.pkl")
