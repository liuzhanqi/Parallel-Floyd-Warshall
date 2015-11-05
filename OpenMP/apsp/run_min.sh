#!/bin/bash

RESULT_FILE="result_min"
APPROACH="0 1"
GRAPH_SIZE="360 720 1440 2880 4320"
MAX_THREAD="16"

for APP in $APPROACH
do
	echo "APP = $APP"
	icc -openmp -O3 -std=c99 -D APPROACH=$APP -o apsp *.c

	for SIZE in $GRAPH_SIZE
	do
		echo "SIZE = $SIZE"
		./apsp $SIZE $MAX_THREAD | tee -a $RESULT_FILE
	done
done