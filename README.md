# QDSA RL Repository
Repository containing code generated under the 2023 QDSA Research Grant.

## RL Implementations
The PPO training algorithms are branched from the
[CleanRL](https://github.com/vwxyzjn/cleanrl) library. The trainers accept PPO
agents with custom network topologies, defined in the `scripts/agents/` directory.
To use a trainer, install the `requirements.txt` into a local Python `venv` and run,
```bash
python scripts/training/[training_script].py [args]
```

This repository also contains a `UnityToGymWrapper` that wraps a Unity3D 
[ml-agents](https://github.com/Unity-Technologies/ml-agents) environment into 
the [gymnasium](https://github.com/Farama-Foundation/Gymnasium) RL API standard,
so that any Unity3D ML scene can be used with the CleanRL algorithms and any 
other modern RL libraries (e.g. [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)).
