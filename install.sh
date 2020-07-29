#!/bin/bash
#######################################

# Update packages
sudo apt update -y

# Install required packages
sudo apt install npm nodejs python3-venv python3-pip ffmpeg -y

# Create virtual environment
python3 -m venv env

#Activate virtual environment
source ./env/bin/activate

# Install required python packages using pip
make install

# Install ipywidgets extension for jupyter lab (this is required for progress bars to work correctly)
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Deactivate virtual environment
deactivate