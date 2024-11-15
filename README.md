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
