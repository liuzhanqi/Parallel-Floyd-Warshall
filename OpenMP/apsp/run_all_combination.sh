#!/bin/bash

RESULT_FILE="result"
APPROACH="0 1 2 3"
GRAPH_SIZE="360 720 1440 2880 5760"
MAX_THREAD="12"

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