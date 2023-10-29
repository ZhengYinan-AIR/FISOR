import copy
from typing import Dict, List, Optional, Sequence

import numpy as np


def set_hyperparameters(
    hyperparameter: dict,
    variants: List[dict],
    group_keys: List = None,
):

    if len(hyperparameter.keys()) == 0:
        for variant in variants:
            if group_keys is not None:
                group_name = ''
                for key in group_keys:
                    group_name += str(variant[key])

                variant['group'] = group_name
                variant['seed'] = np.random.randint(1000)
            

        return variants
    else:
        new_variants = []
        k = list(hyperparameter.keys())[0]
        for item in hyperparameter[k]:
            for variant in variants:
                new_variant = copy.deepcopy(variant)
                new_variant[k] = item
                new_variants.append(new_variant)
                del variant

        hyperparameter.pop(k, None)
        return set_hyperparameters(hyperparameter, new_variants, group_keys)
