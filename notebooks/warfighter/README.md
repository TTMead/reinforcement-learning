# Warfighter

## ML Learn CLI
To train the model using the config file use,
```bash
mlagents-learn config/warfighter_config.yaml --run-id=Warfighter
```
- If resuming and existing training session the `--resume` flag can be used.
- A specific timescale can be specified with the `--time-scale=1.0` flag.
