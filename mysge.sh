#!/bin/bash
#
# Put your Job commands here.
#
#------------------------------------------------

/opt/openmpi/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines -mca btl tcp,self,sm \
/home/team12/lab1

#------------------------------------------------
