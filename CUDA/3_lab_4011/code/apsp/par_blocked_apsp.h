#ifndef __PAR_BLOCKED_APSP_H__
#define __PAR_BLOCKED_APSP_H__

#define BLOCK_SIZE 32
#define THREAD_SIZE 2

void par_blocked_apsp(int N, int *mat);

#endif // __PAR_BLOCKED_APSP_H__