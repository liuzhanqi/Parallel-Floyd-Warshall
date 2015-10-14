ssh team12@172.21.148.156
scp -r ./* team12@172.21.148.156:/home/team12/lab1
- -
mpicc -O3 -std=c99 -o apsp APSPtest.c MatUtil.c
mpirun -np 6 apsp 1200

gcc -O3 -std=c99 -o apspseq APSPsequential.c MatUtil.c
apspseq 1200

#submit my job
cd /home/team12/lab1
qsub -pe mpich 6 -v mat_size="600 1200 2400 4800" mysge.sh

# or batch submit
# check first
./run_mpi.sh
./run_mpi.sh -l 0 -i "APSPtest.c MatUtil.c" -o "apsp" -c "2 4 6 8" -m "600 1200 2400 4800"
./run_mpi.sh -l 0 -i "APSPtest.c MatUtil.c" -o "apsp" -c "10" -m "1200"

./run_mpi.sh -l 0 -i "APSPtest_divisible.c MatUtil.c" -o "apsp" -c "2 4 6 8" -m "600 1200 2400 4800"


#check out the result or error log
cat /home/team12/mysge.sh.o5981
cat /home/team12/mysge.sh.e5981

ip
welcome
