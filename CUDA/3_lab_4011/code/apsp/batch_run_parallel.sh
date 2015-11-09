#!/bin/bash

result=result_parallel.txt
touch $result
for size in {512..4096..512}
do
	./template $size >> $result
	echo "done $size."
done 
