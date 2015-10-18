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

printf "... running_locally = %s\n" "$running_locally"
printf "... in_files  = %s\n" "$in_files"
printf "... out_file  = %s\n" "$out_file"
printf "... num_cores = %s\n" "$num_cores"
printf "... mat_size  = %s\n" "$mat_size"

if type "icc" > /dev/null; then
	echo "icc –openmp –O3 –std=c99 –o $out_file $in_files"
	icc -openmp -O3 -std=c99 -o $out_file $in_files
elif type "gcc-5" > /dev/null; then
	echo "gcc-5 -openmp -O3 -I/usr/include -L/usr/lib -o $out_file $in_files"
	gcc-5 -fopenmp -O3 -I/usr/include -L/usr/lib -o $out_file $in_files
else
	echo "compiler icc/gcc-5 not found."
	exit 1
fi

compile_result=$?
echo "Compilation finished: $compile_result"
if [ $compile_result -ne 0 ]; then
	echo "Compilation Error."
	exit $compile_result
else
	for core in $num_cores
	do
		echo "=== === === === === === === === === ==="
		export OMP_NUM_THREADS="$core"
		if [ $running_locally -eq 0 ]; then
			echo "do nothing..."
			# echo "qsub -pe mpich $core -v mat_size=\"$mat_size\" mysge.sh"
			# qsub -pe mpich $core -v mat_size="$mat_size" -v out_file="$out_file" submit.sh
		else
			for size in $mat_size
			do
				echo "running: $core cores $size vertex"
				./$out_file $size
				echo "--- --- --- --- --- --- --- --- --- ---"

				sleep 1
			done
		fi
	done
fi


