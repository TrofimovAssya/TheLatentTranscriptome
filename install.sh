#!/bin/bash
#Install script. setup the virtual environment and everything

python -m venv .env
. .env/bin/activate # Need to run this everytime we want to run the project. 
sudo pip install -e .
sudo pip install -r requirements.txt

