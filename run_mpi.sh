#!/bin/bash

out_name="apsp"
in_file="./*.c"
number_of_core=2

mpicc -O3 -o out_name in_file

compile_result=$?

echo "Compilation finished with $compile_result..."
if [ compile_result -ne 0 ]; then
	echo "abort."
	exit
else
	echo "running..."
	mpirun -np number_of_core out_name
fi


# mpicc -O3 -o apsp ./*.c
# mpirun -np 2 apsp