#!/bin/bash
cd ./LoadBalancer/
python server.py > /dev/null &

sleep 3

cd ../Training/scripts/
python train.py ../Configs/pgd.json &

sleep 5

cd ../../GeneratorClient/
python client.py > /dev/null & 
python client.py > /dev/null & 
python client.py > /dev/null &  