#!/bin/bash

IP=192.168.2.1
#IP=$(./raspberryip.sh)

ssh -t continualist@$IP "sudo date -s '$(date)'; cd democontinualist/auto-adaptive-AI-in-the-box-DEMO-v1; export INFLUX_TOKEN='$INFLUX_TOKEN'; bash"
