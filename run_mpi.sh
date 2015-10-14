#!/bin/bash

while getopts ":l:i:o:c:m:" opt; do
  case $opt in
  	l) running_locally="$OPTARG"
	;;
    i) in_files="$OPTARG"
    ;;
    o) out_file="$OPTARG"
    ;;
    c) num_cores="$OPTARG"
    ;;
   	m) mat_size="$OPTARG"
    ;;
    \?)
		echo "Invalid option -$OPTARG" >&2
		exit 1
    ;;
  esac
done

if ([ $running_locally -ne 0 ] && [ $running_locally -ne 1 ]) || [ ! "$in_files" ] || [ ! "$out_file" ] || [ ! "$num_cores" ] || [ ! "$mat_size" ]; then
	echo "usage:"
	echo "-l [0/1 running locally]"
	echo "-i [array of input files to compile]"
	echo "-o [one output file for binary]"
	echo "-c [array of optional cores]"
	echo "-m [array of number of vertex]"
	exit 1
fi

printf "Argument running_locally is %s\n" "$running_locally"
printf "Argument in_files is %s\n" "$in_files"
printf "Argument out_file is %s\n" "$out_file"
printf "Argument num_cores is %s\n" "$num_cores"
printf "Argument mat_size is %s\n" "$mat_size"

echo "mpicc -O3 -std=c99 -o $out_file $in_files"
mpicc -O3 -std=c99 -o $out_file $in_files

compile_result=$?

echo "Compilation finished with $compile_result..."
if [ $compile_result -ne 0 ]; then
	echo "abort."
	exit
else
	for core in $num_cores
	do
		echo "running with $core cores..."
		if [ $running_locally -eq 0 ]; then
			echo "qsub -pe mpich $core -v mat_size=\"$mat_size\" mysge.sh"
			qsub -pe mpich $core -v mat_size="$mat_size" -v out_file="$out_file" submit.sh
		else
			for size in $mat_size
			do
				echo "running with mat size: $size"
				echo "mpirun -np $core $out_file $size"
				mpirun -np $core $out_file $size
			done
		fi
	done
fi

echo "--- all done ---"

# mpicc -O3 -o apsp ./*.c
# mpirun -np 2 apsp