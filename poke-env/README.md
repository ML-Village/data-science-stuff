# poke-env

## Purpose

This directory extends the poke-env package to other RL frameworks beyond `keras-rl`

## Getting Started

1. Host a local pokemon showdown server. Refer [here](https://poke-env.readthedocs.io/en/stable/getting_started.html#configuring-a-showdown-server) for instructions.

2. Install python packages.

```bash
python -m venv venv
. venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

3. Start training

```bash
python train.py
```

## Project Directory

    .
    ├── src                     
    │   ├── envs            # Extensions from poke-env's EnvPlayer class
    │   ├── players         # Extensions from poke-env's Player class
    │   └── trainers        # Helpers to train RL model using different frameworks
    │       └── torch       
    ├── requirements.txt    # Python packages required
    ├── train.py            # Script to train RL model
    └── README.md