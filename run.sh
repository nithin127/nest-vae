#!/usr/bin/env bash

# Source bashrc
source $HOME/.bashrc

# Activate the environment
# source activate deleutri
rl

# Run the script
python divergence_vae.py $@
