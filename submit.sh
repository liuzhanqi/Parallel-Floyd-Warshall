#!/bin/bash
#
# Put your Job commands here.
#
#------------------------------------------------

RunMPI () {
	# out_file=$1
	echo "arg 1: $1"
	for mat_size in ${@:2:$#}
	do
		echo "running with mat size: $mat_size..."
		echo "/opt/openmpi/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines -mca btl tcp,self,sm /home/team12/lab1/$out_file $mat_size"
		/opt/openmpi/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines -mca btl tcp,self,sm /home/team12/lab1/$out_file $mat_size
	done
}

RunMPI $out_file $mat_size

#------------------------------------------------
