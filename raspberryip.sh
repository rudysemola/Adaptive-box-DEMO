#!/bin/bash

MAC=b8:27:eb:12:1d:46

arp -a | grep $MAC | awk '{print $2;}' | rev | cut -c 2- | rev | cut -c 2-
