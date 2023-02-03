#!/bin/bash

IP=$(./raspberryip.sh)

ssh -t continualist@$IP 'cd democontinualist/auto-adaptive-AI-in-the-box-DEMO-v1; export INFLUX_TOKEN='$INFLUX_TOKEN'; bash'
