"""Extends the normalize wrappers with saving the mean/std values to the environment metadata.
Branched from https://gymnasium.farama.org/_modules/gymnasium/wrappers/normalize/"""

"""Set of wrappers for normalizing actions and observations."""
import numpy as np

import gymnasium as gym
import warnings

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), init_mean=None, init_var=None):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

        if (init_mean is not None):
            assert (init_mean.shape == shape), "Desired initial mean has incorrect shape"
            self.mean = init_mean
        
        if (init_var is not None):
            assert (init_var.shape == shape), "Desired initial var has incorrect shape"
            self.var = init_var

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, mean = None, var = None):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape, init_mean=mean, init_var=var)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape, init_mean=mean, init_var=var)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)

        # Update the mean/variance metadata
        self.metadata["observation_mean"] = self.obs_rms.mean
        self.metadata["observation_var"] = self.obs_rms.var

        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class StaticNormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance. Unlike the NormalizeObservation wrapper,
    this wrapper does NOT use a running mean/var and instead utilises a constant mean/var provided at construction."""

    def __init__(self, env: gym.Env, mean, var, epsilon: float = 1e-8, ):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            assert (self.single_observation_space.shape == mean.shape)
            assert (self.single_observation_space.shape == var.shape)
        else:
            assert (self.observation_space.shape == mean.shape)
            assert (self.observation_space.shape == var.shape)
        
        self.mean = mean
        self.var = var
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)

class MinMaxNormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations between a set of minimum and maximum values into the range of 0 to 1"""

    def __init__(self, env: gym.Env, ranges: np.ndarray, epsilon: float = 1e-8, ):
        """
        Args:
            env (Env): The environment to apply the wrapper
            ranges: The minimum and maximum values that each observation will have.
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            assert ((self.single_observation_space.shape[0] % ranges.shape[0]) == 0), ("Normalization range shape (" + str(ranges.shape) + ") is not a multiple of the observation shape (" + str((self.single_observation_space.shape[0], 2)) + ")")
            self.stack_size = int(self.single_observation_space.shape[0] / ranges.shape[0])
        else:
            assert ((self.observation_space.shape[0] % ranges.shape[0]) == 0), ("Normalization range shape (" + str(ranges.shape) + ") is not a multiple of the observation shape (" + str((self.observation_space.shape[0], 2)) + ")")
            self.stack_size = int(self.observation_space.shape[0] / ranges.shape[0])
        
        for range in ranges:
            assert (range[0] != range[1]), "Normalization range has the same min and max value."
        
        self.ranges = ranges
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation between the min and max values to the range of 0 to 1."""
        mins = self.ranges[:,0]
        maxs = self.ranges[:,1]
        obs_stacked_vec = obs[0]

        for stack in range(self.stack_size):
            stack_index = (stack * self.ranges.shape[0])
            obs_vec = obs_stacked_vec[stack_index:(stack_index + self.ranges.shape[0])]

            # Ensure that the observation vector is within the min/max bounds before we normalize
            for i in range(len(obs_vec)):
                if (obs_vec[i] <= mins[i]) or (obs_vec[i] >= maxs[i]):
                    warnings.warn("Observation value (" + str(obs_vec[i]) + ") is outside of valid range (" + str(mins[i]) + ", " + str(maxs[i]) + "). Clamping value.")
                    obs_vec[i] = np.clip(obs_vec[i], mins[i], maxs[i])
            
            # Normalize
            obs_vec = (obs_vec - mins) / (maxs - mins)
            obs_stacked_vec[stack_index:(stack_index + self.ranges.shape[0])] = obs_vec
        
        obs[0] = obs_stacked_vec
        return obs
