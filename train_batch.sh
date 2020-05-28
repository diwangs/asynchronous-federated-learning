#!/bin/sh

N_SERVER=16

for STDEV in 0 10 1000; do
    for LOSS in 0% 10% 25%; do
        for STALENESS_THRESHOLD in 0 5 10; do
            for i in {1..3}; do # Redo 3 times
                # echo $STDEV $LOSS $STALENESS_THRESHOLD $i
                ./train.sh $N_SERVER $STDEV $LOSS $STALENESS_THRESHOLD
            done
        done
    done
done