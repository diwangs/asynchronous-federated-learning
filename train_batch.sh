#!/bin/sh

N_SERVER=4

mkdir -p logs
for i in {1..3}; do # Redo 3 times
	for LOSS in 0% 10% 20%; do
		for STDEV in 0 10 1000; do
			for STALENESS_THRESHOLD in 0 6 12 9999; do
				echo ""
				echo " ---- STARTING $LOSS $STDEV $STALENESS_THRESHOLD $i ---- "
				echo ""
				./train.sh $N_SERVER $LOSS $STDEV $STALENESS_THRESHOLD 2>&1 | tee logs/$LOSS-$STDEV-$STALENESS_THRESHOLD-$i.log
			done
		done
	done
done