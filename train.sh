#!/bin/sh

N_SERVER=2
STDEV=0
LOSS=10%
STALENESS_THRESHOLD=0

cleanup()
{
    echo "Restoring network..."
    sudo tc qdisc del dev lo root
    echo "Gracefully (not really) shutting servers down..."
    kill -9 $(echo $(pgrep -P $SERVERS_PID) | tr '\n' ' ')$SERVERS_PID
}
trap cleanup SIGINT SIGTERM

python src/servers.py $N_SERVER $STDEV &
SERVERS_PID=$!
echo "Modifying network..."
sudo tc qdisc add dev lo root handle 1:0 netem loss random $LOSS
python src/client.py $N_SERVER $STALENESS_THRESHOLD
