# set up environment ----------------------------------------------

import numpy as np
from td_dynamic.karun.multiple.simulate_multiple import EquilibriumMultipleEntryExit
from td_dynamic.karun.utils import create_s3_folder_if_not_exists, write_pickle_to_s3
import multiprocessing

# set constants ----------------------------------------------------

prefix = "output/simulate_multiple/"
bucket_name = "football-markov"

create_s3_folder_if_not_exists(bucket=bucket_name, folder=prefix)

num = {"market": 1000, "firm": 3, "period": 100, "state": 1, "action": 2, "tolerance": 1e-6}

num["grid_size"] = np.full(num["state"], 5)

# set parameters ----------------------------------------------------

param = {
    "param_payoff": {"beta": 0.7, "alpha": 0.1, "lambda": 0.1},
    "param_state": {
        "state_constant": np.concatenate((np.full(num["state"], 0.0),)),
        "state_trans": np.eye(num["state"]) * 0.3,
        "state_sd": np.eye(num["state"]),
    },
}

discount = 0.95

# solve the equilibrium -------------------------------------------

def main():
    model = EquilibriumMultipleEntryExit(discount=discount, num=num, param=param)
    model.simulate_equilibrium()
    write_pickle_to_s3(obj=model, bucket=bucket_name, prefix=prefix, file_name="equilibrium.pkl")

if __name__ == '__main__':
    # This is required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
