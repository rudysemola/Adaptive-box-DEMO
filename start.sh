#!/bin/bash

IP=$(echo $SSH_CLIENT | awk '{ print $1}')

docker kill demo
docker rm demo
docker run -dit -v $(pwd):/container/demo -e INFLUX_TOKEN=$INFLUX_TOKEN --name demo --privileged -it adaptivets-demo
docker exec -t demo bash -c "cd /container/demo; python3 ./main.py $IP"
