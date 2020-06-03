#!/bin/sh

N_SERVER=4

mkdir -p logs
for i in 1; do # Redo 3 times
	for STDEV in 1000; do # 0, 10, 1000
		for LOSS in 20%; do # 0%, 20%?
			for STALENESS_THRESHOLD in 9999; do
				echo ""
				echo " ---- STARTING $STDEV $LOSS $STALENESS_THRESHOLD $i ---- "
				echo ""
				./train.sh $N_SERVER $STDEV $LOSS $STALENESS_THRESHOLD 2>&1 | tee logs/$STDEV-$LOSS-$STALENESS_THRESHOLD-$i.log
			done
		done
	done
done