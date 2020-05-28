#!/bin/sh

N_SERVER=$1
STDEV=$2
LOSS=$3
STALENESS_THRESHOLD=$4

cleanup()
{
    echo "Restoring network..."
    sudo tc qdisc del dev lo root
    echo "Forcefully shutting servers down..."
    # TODO: wait for servers to fork
    kill -9 $(echo $(pgrep -P $SERVERS_PID) | tr '\n' ' ')$SERVERS_PID
}
trap cleanup EXIT

LOG_DIR="out/$STDEV-$LOSS-$STALENESS_THRESHOLD-$(date +%s)"
mkdir -p $LOG_DIR
python src/servers.py $N_SERVER $STDEV &
SERVERS_PID=$!
echo "Modifying network..."
sudo tc qdisc add dev lo root handle 1:0 netem loss random $LOSS
python src/client.py $N_SERVER $STALENESS_THRESHOLD "$LOG_DIR/f1.csv" "$LOG_DIR/yappi.csv"
