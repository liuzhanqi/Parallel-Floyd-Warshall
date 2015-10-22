# 
# ssh chen0818@172.21.148.162
# 
# scp -r /Users/cly/Dropbox/code/Parallel-Floyd-Warshall/OpenMP/* chen0818@172.21.148.162:~/4011_lab2
# 
# icc –openmp –O3 –std=c99 –o apsp *.c
# ./apsp N
# 
# By default 12 threads will be used (6 processing cores with hyperthreading).
# To set the number of threads to p before execution, type:
# export OMP_NUM_THREADS=p
# 
# ./run_mp.sh -l 1 -i "APSPtest.c MatUtil.c" -o "apsp" -c "2 4 6 8" -m "720 1440"
# 
# 
# 
# 
# 
