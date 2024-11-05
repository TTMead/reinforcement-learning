'''
Functions for normalizing actions and observations
'''

import numpy as np
import warnings


def normalize_single_obs(obs, ranges):
    # Tile the ranges to account for stacked observations
    stack_size = len(obs) // len(ranges)
    stacked_ranges = np.tile(ranges, (stack_size, 1))

    mins = stacked_ranges[:,0]
    maxs = stacked_ranges[:,1]

    # Ensure that the observation vector is within the min/max bounds before we normalize
    for i in range(len(obs)):
        if (obs[i] < mins[i]) or (obs[i] > maxs[i]):
            warnings.warn("Observation value (" + str(obs[i]) + ") is outside of valid range (" + str(mins[i]) + ", " + str(maxs[i]) + "). Clamping value.")
            obs[i] = np.clip(obs[i], mins[i], maxs[i])

    # Normalize
    return (obs - mins) / (maxs - mins)


def normalize_obs(obs, ranges):
    for i in range(obs.shape[0]):
        obs[i] = normalize_single_obs(obs[i], ranges)

    return obs
