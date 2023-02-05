#!/bin/bash

influxd --http-idle-timeout 0 --session-length 360 --http-read-header-timeout 0 --http-write-timeout 0 --http-read-timeout 0
