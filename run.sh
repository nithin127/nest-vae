#!/usr/bin/env bash

# Source bashrc
source $HOME/.bashrc

# Activate the environment
source activate deleutri

# Run the script
# python tristan/pytorch_tutorial_vae/main.py $@
python tristan/vae.py $@
