import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt

import bandits_lab.algorithms as algs

from copy import deepcopy
from joblib import Parallel, delayed

# running the simuations from a data_dict:
# data_dict={
#         'name':'Long Name',
#         'short_name':'short_name',
#         'T':T,
#         'N_tests':N_tests,
#         'band_list':band_list,
#         'alg_list':alg_list,
#         'results':None,
#     }


# Playing algorithms


def one_regret(a, b, T):
    alg = deepcopy(a)
    bandit = deepcopy(b)
    converged = True
    try:
        alg.play_T_times(bandit, T)
        return bandit.cum_regret, converged
    except algs.ConvergenceError:
        converged = False
        return None, converged


def n_regret(alg, bandit, T, N_test=100, verb=False, n_jobs=1):
    reg_list = Parallel(n_jobs=n_jobs)(
        delayed(one_regret)(alg, bandit, T) for _ in range(N_test)
    )
    n_regret_list = []
    all_converged = True
    for regret, converged in reg_list:
        if not (converged):
            all_converged = False
            continue
        else:
            n_regret_list.append(regret)
    if all_converged:
        return np.array(n_regret_list), True
    else:
        return None, False


# Saving utilities:
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path


def save_data_dict(data_dict, uniquify=False):
    folder = data_dict["folder"]
    if uniquify:
        path = uniquify(folder + data_dict["short_name"] + ".pkl")
    else:
        path = folder + data_dict["short_name"] + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)


def load_data_dict(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def launch(data_dict, verb=False, n_jobs=1, checkpoints=True):
    if "seed" in data_dict.keys():
        np.random.seed(data_dict["seed"])
    T, band_list = data_dict["T"], data_dict["band_list"]
    alg_list, N_tests = data_dict["alg_list"], data_dict["N_tests"]
    results = []
    time_comp = []
    ended = []
    for i, band in enumerate(band_list):
        results.append([])
        time_comp.append([])
        ended.append([])
        N_test = N_tests[i]
        for j, alg in enumerate(alg_list):
            t0 = time.time()
            n_regret_array, all_converged = n_regret(
                alg, band, T, N_test=N_test, verb=verb, n_jobs=n_jobs
            )
            if all_converged:
                time_taken = time.time() - t0
                time_comp[-1].append(time_taken)
                ended[-1].append(True)
                print(
                    "{} took {:0.2f}s total, i.e.,".format(alg.label, time_taken)
                    + " {:0.2f}s per run".format(time_taken / N_test),
                )
                mean_reg = np.mean(n_regret_array, axis=0)
                var_reg = np.var(n_regret_array, axis=0)
                results[-1].append((mean_reg, var_reg))
                data_dict["time_comp"] = time_comp
                data_dict["results"] = results
                data_dict["ended"] = ended
                if checkpoints:
                    print("saved")
                    save_data_dict(data_dict, uniquify=False)
            else:
                time_comp[-1].append(None)
                results[-1].append((None, None))
                ended[-1].append(False)
                print(alg.label, " failed to converge")
                if checkpoints:
                    print("saved")
                    save_data_dict(data_dict, uniquify=False)
                continue
