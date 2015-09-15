ssh team12@172.21.148.156
welcome
scp /Users/cly/Desktop/Lab1/* team12@172.21.148.156:/home/team12/lab1
mpicc -O3 -std=c99 -o apsp APSPtest.c MatUtil.c
mpirun -np 6 apsp 1200

gcc -O3 -std=c99 -o apspseq APSPsequential.c MatUtil.c
apspseq 1200