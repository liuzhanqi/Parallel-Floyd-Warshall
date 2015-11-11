#!/bin/bash

result=batch_result.txt
touch $result

make clean && make

rc=$?; 
if [[ $rc != 0 ]]; 
then 
	exit $rc; 
fi

for size in {512..8192..512}
do
	./template $size >> $result
	echo "done $size."
done

printf "\n" >> $result 
