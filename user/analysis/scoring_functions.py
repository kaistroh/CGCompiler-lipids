import numpy as np
from pyemd import emd
from scipy.spatial.distance import cdist

def scaled_absolute_error(y_target, y_observed):
    return np.abs( (y_target - y_observed) / y_target )

def range_SAE(y_target, y_observed, Etol, cap):
    sae = scaled_absolute_error(y_target, y_observed)

    return np.min([ np.max([0, sae - Etol]), cap ])

def dict_SAE(target_dict, observed_dict):
    score = 0
    for key in observed_dict:
        score += scaled_absolute_error(target_dict[key], observed_dict[key])

    return score / len(observed_dict)

def dict_range_SAE(target_dict, observed_dict, Etol, cap):
    score = 0
    for key in observed_dict:
        score += range_SAE(target_dict[key], observed_dict[key], Etol, cap)

    return score / len(observed_dict)

def emd_score(target_dict, observed_dict, weights_dict=None):
    """
    Calculates the earth mover's distance, observed_dict contains the bonds (angles) of interest
    """
    score = 0
    for key in observed_dict: # keys are bonds or angles to optimize
        hist_target, bins_target = target_dict[key]
        hist_observed, bins_observed = observed_dict[key]

        hist_target = np.array(hist_target)
        bins_target = np.array(bins_target)

        hist_observed = np.array(hist_observed)
        bins_observed = np.array(bins_observed)

        dist_mat = cdist(bins_target.reshape(-1,1), bins_observed.reshape(-1,1)) / (bins_target[-1] - bins_target[0])
        if weights_dict is not None:
            score += emd(hist_target, hist_observed, dist_mat) * weights_dict[key]
        else:
            score += emd(hist_target, hist_observed, dist_mat)

    return score / len(observed_dict)


def get_normalizations(obs_dict):
    import user.usersettings as UserSettings

    observables_in_use = {}
    for molkey in obs_dict:
        for tr_system in obs_dict[molkey]:
            if tr_system in UserSettings.training_systems[molkey]:
                for observable, weight in obs_dict[molkey][tr_system].items():
                    if observable in observables_in_use.keys():
                        observables_in_use[observable] += weight
                    else:
                        observables_in_use[observable] = weight

    return observables_in_use

