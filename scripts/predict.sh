#!/bin/bash
source activate ein
set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'

python seld/main.py -c $CONFIG_FILE infer --num_workers=4
