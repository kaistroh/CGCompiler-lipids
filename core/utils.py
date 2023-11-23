from collections import defaultdict
from copy import deepcopy
import sys
import gc
import json
import numpy as np
from mergedeep import merge, Strategy

def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size


def merge_dicts(dict0, dict1):
    merged = defaultdict(dict)
    merged.update(dict0)
    for key, nested_dict in dict1.items():
        merged[key].update(nested_dict)
        
    return dict(merged)

def merge_dicts_multi_level(dict0, dict1):
    merged = defaultdict(dict)
    merged.update(dict0)
    print(merged)
    for key, nested_dict in dict1.items():
        print(key)
        print(nested_dict)
        merged[key].update(nested_dict)
        
    return dict(merged)

def merge_dicts_deep(dict0, dict1):
    merged = deepcopy(dict0)
    merge(merged, dict1, strategy=Strategy.ADDITIVE)

    return merged


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def my_cauchy(size, x0, gamma, RNG):
    out = np.zeros(size)
    
    for i in range(size):
        rand = RNG.uniform(0,1)
        out[i] = np.tan(np.pi * (rand - 0.5))
        
    return out * gamma + x0

def check_observables_calc(weights_dict, calc_dict):
    for molkey in weights_dict:
        for tr_system in weights_dict[molkey]:
            for observable in weights_dict[molkey][tr_system]:
                if calc_dict[tr_system][observable] is not True:
                    raise Exception('observable %s for %s is used in scoring but not calculated' %(observable, tr_system))