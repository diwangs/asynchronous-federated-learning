#!/bin/sh
# ./train.sh {centralized/distributed} {mnist/cifar} <n_servers> <stdev> <staleness>
python src/servers.py 2 0 &
sudo tc qdisc add dev lo root handle 1:0 netem delay 2ms loss random 10%
while [ -z "$(pgrep -P $!)" ]; do sleep 1; done # Wait till the servers fork
trap "kill $(echo $(pgrep -P $!) | tr '\n' ' ') $!; sudo tc qdisc del dev lo root" INT EXIT
python src/client.py 2
