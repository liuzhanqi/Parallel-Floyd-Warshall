#ifndef __APSP_H__
#define __APSP_H__

static int ref_N;
static int* ref;

void gen_apsp(int N, int* mat);

void seq_apsp();

bool check_apsp(int* mat);

#endif // __APSP_H__