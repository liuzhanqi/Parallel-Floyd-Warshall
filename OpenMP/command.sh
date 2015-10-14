# 
# ssh chen0818@17221148161
# 
# scp -r /Users/cly/Dropbox/code/Parallel-Floyd-Warshall/OpenMP/* chen0818@17221148161:~/4011_lab2
# 
# icc –O3 –std=c99 –o APSPtest APSPtest.c MatUtil.c
# ./APSPtest N
# 
# icc –openmp –O3 –std=c99 –o APSPtest APSPtest.c MatUtil.c
# ./APSPtest N
# 
# By default 12 threads will be used (6 processing cores with hyperthreading).
# To set the number of threads to p before execution, type:
# export OMP_NUM_THREADS=p
# 
# 
# 
# 
# 
# gcc -O3 -std=c99 -o apspseq APSPsequential.c MatUtil.c
# apspseq 1200
# 
# #submit my job
# cd /home/team12/lab1
# qsub -pe mpich 6 -v mat_size="600 1200 2400 4800" mysge.sh
# 
# 
# 