# QDSA RL Repository
Repository containing code generated under the 2023 QDSA Research Grant.

The `notebooks/` directory contains Python notebooks created for random
testing or under study materials.

The `scripts/` directory contains Python RL implementations developed during the
project.


## RL Implementations
The PPO training algorithm is branched from the
[CleanRL](https://github.com/vwxyzjn/cleanrl) library. The trainer accepts PPO
agents with custom network topologies, defined in the `scripts/agents/` directory.
To use the trainer, install the `requirements.txt` into a local Python `venv` and run,
```bash
python scripts/ppo.py
```

This repository also contains a `UnityToGymWrapper` that wraps a Unity3D 
[ml-agents](https://github.com/Unity-Technologies/ml-agents) environment into 
the [gymnasium](https://github.com/Farama-Foundation/Gymnasium) RL API standard,
so that any Unity3D ML scene can be used with the CleanRL algorithms and any 
other modern RL libraries (e.g. [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)).
