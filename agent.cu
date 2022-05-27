/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
/*************************************************************************
/* ECE 277: GPU Programmming 2020 FALL
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

float epsilon;
short *d_action;
float *qtable;
float *e;
int* trace;
short *active;
short* clear;
int k;
int *p;
float* dOut;
#define MAX(a,b) (((a)>(b))?(a):(b));

__global__ void cuda_init(float *qtable, int* p, short* active, float* out, curandState* states, float* e, int* trace) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid == 0) {
		*p = 0;
	}
	for (int i = 0; i < 32; i++) {
		*(qtable + i + tid*32) = 0;
		*(e + i + tid*32) = 0;
	}
	*(active + tid) = 1;
	curandState* state = states + tid;
	curand_init(100, tid, 0, state);
	int nthreads = gridDim.x * blockDim.x;
	for (int i = tid; i < 2000 * 128; i += nthreads)
	{
		float random = curand_uniform(state);
		out[i] = random;
	}
}

void agent_init()
{
	cudaMalloc((short**)&d_action, 128 * sizeof(short));
	cudaMalloc((float**)&qtable, 32 * 32 * 4 * sizeof(float));
	cudaMalloc((float**)&e, 32 * 32 * 4 * sizeof(float));
	cudaMalloc((int**)&p, sizeof(int));
	cudaMalloc((short**)&active, 128 * sizeof(short));
	cudaMalloc((void **)&dOut, sizeof(float) * 2000 * 128);
	cudaMalloc((int**)&trace, sizeof(int) * 64 * 3 * 128);
	static curandState *state = NULL;
	cudaMalloc((void **)&state, sizeof(curandState));
	cudaMalloc((short**)&clear, sizeof(short) * 128);
	epsilon = 1;
	k = 0;
	cuda_init << <2, 64 >> > (qtable, p, active, dOut, state, e, trace);
}

__global__ void cuda_agent_init(short* active, float* e, int* trace, short* clear) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	*(active + tid) = 1;
	for (int i = 0; i < 192; i++) {
		if (i < 31) {
			e[tid * 32 + i] = 0;
		}
		if (i < 128) {
			clear[tid] = 0;
		}
		trace[tid * 192 + i] = -1;
	}	
}

void agent_init_episode()
{
	cuda_agent_init << <2, 64 >> > (active, e, trace, clear);
}


float agent_adjustepsilon()
{
	epsilon = 0.0005;
	return epsilon;
}

__global__ void cuda_action(int2* cstate, short* d_action, float epsilon, float qtable[32 * 32 * 4], float* out, int k, int* p, short *active, short* clear) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ short act[128];
	__shared__ float rand[128];
	__shared__ float r[128];
	__shared__ short clr[128];
	r[tid] = out[*p + tid * 128];
	rand[tid] = out[k + tid * 128];
	act[tid] = active[tid];
	__syncthreads();
	float m = max(qtable[(cstate[tid].x + cstate[tid].y * 32) * 4], qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1]);
	m = max(m, qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2]);
	m = max(m, qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3]);
	if (act[tid] != 0) {
		if (rand[tid] > epsilon) { //exploit
			clr[tid] = 0;
			if (m == 0) {
				float random = r[tid];
				random *= 4;
				(*p)++;
				if (random > 3 || random == 3) {
					d_action[tid] = 3;
				}
				else if ((random > 2 || random == 2) && (random < 3)) {
					d_action[tid] = 2;
				}
				else if ((random > 1 || random == 1) && (random < 2)) {
					d_action[tid] = 1;
				}
				else if ((random > 0 || random == 0) && (random < 1)) {
					d_action[tid] = 0;
				}
			}
			else {
				if (m == qtable[(cstate[tid].x + cstate[tid].y * 32) * 4]) {
					d_action[tid] = 0;
				}
				else if (m == qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1]) {
					d_action[tid] = 1;
				}
				else if (m == qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2]) {
					d_action[tid] = 2;
				}
				else if (m == qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3]) {
					d_action[tid] = 3;
				}
			}
		}
		else {
			clr[tid] = 1;
			float random = r[tid];
			random *= 4;
			(*p)++;
			if (random > 3 || random == 3) {
				d_action[tid] = 3;
			}
			else if ((random > 2 || random == 2) && (random < 3)) {
				d_action[tid] = 2;
			}
			else if ((random > 1 || random == 1) && (random < 2)) {
				d_action[tid] = 1;
			}
			else if ((random > 0 || random == 0) && (random < 1)) {
				d_action[tid] = 0;
			}
		}
	}
	clear[tid] = clr[tid];
}
short* agent_action(int2* cstate)
{
	k++;
	cuda_action << <2, 64 >> > (cstate, d_action, epsilon, qtable, dOut, k, p, active, clear);
	return d_action;
}

__global__ void cuda_update(int2* cstate, int2*nstate, float* rewards, float *qtable, short* d_action, short* active, float* e, float* out, int* p, int * trace, short* clear) {
	float a = 0.1, y = 0.9;
	float m;
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ short act[128];
	__shared__ float rand[128];
	__shared__ short clr[128];
	act[tid] = active[tid];
	rand[tid] = out[*p + tid * 128];
	clr[tid] = clear[tid];
	__syncthreads();
	int test= 0;
	int idx=0;
	while (test != -1) {
		test = trace[tid * 192 + idx];
		idx++;
	}
	idx--;
	trace[tid * 192 + idx] = cstate[tid].x;
	trace[tid * 192 + idx + 1] = cstate[tid].y;
	trace[tid * 192 + idx + 2] = d_action[tid];
	if (d_action[tid] == 0) {
		e[(cstate[tid].x + cstate[tid].y * 32) * 4] ++;
	}
	else if (d_action[tid] == 1) {
		e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1] ++;
	}
	else if (d_action[tid] == 2) {
		e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2] ++;
	}
	else if (d_action[tid] == 3) {
		e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3] ++;
	}
	/*short next_act[128];
	float random = rand[tid];
	random *= 4;
	if (random > 3 || random == 3) {
		next_act[tid] = 3;
	}
	else if ((random > 2 || random == 2) && (random < 3)) {
		next_act[tid] = 2;
	}
	else if ((random > 1 || random == 1) && (random < 2)) {
		next_act[tid] = 1;
	}
	else if ((random > 0 || random == 0) && (random < 1)) {
		next_act[tid] = 0;
	}*/
	if (act[tid] != 0) {
		for (int i = 0; i < idx; i += 3) {
			/*cstate[tid].x = trace[i];
			cstate[tid].y = trace[i + 1];
			d_action[tid] = trace[i + 2];*/
			if (d_action[tid] == 0) {
				if (cstate[tid].x < 31) {
					m = max(qtable[(cstate[tid].x + 1 + cstate[tid].y * 32) * 4], qtable[(cstate[tid].x + 1 + cstate[tid].y * 32) * 4 + 1]);
					m = max(m, qtable[(cstate[tid].x + 1 + cstate[tid].y * 32) * 4 + 2]);
					m = max(m, qtable[(cstate[tid].x + 1 + cstate[tid].y * 32) * 4 + 3]);
					qtable[(cstate[tid].x + cstate[tid].y * 32) * 4] +=
						a * (rewards[tid] + y * m
							- qtable[(cstate[tid].x + cstate[tid].y * 32) * 4])
						*e[(cstate[tid].x + cstate[tid].y * 32) * 4];
				}
			}
			else if (d_action[tid] == 1) {
				if (cstate[tid].y < 31) {
					m = max(qtable[(cstate[tid].x + (cstate[tid].y + 1) * 32) * 4], qtable[(cstate[tid].x + (cstate[tid].y + 1) * 32) * 4 + 1]);
					m = max(m, qtable[(cstate[tid].x + (cstate[tid].y + 1) * 32) * 4 + 2]);
					m = max(m, qtable[(cstate[tid].x + (cstate[tid].y + 1) * 32) * 4 + 3]);
					qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1] +=
						a * (rewards[tid] + y * m
							- qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1])
						*e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 1];
				}
			}
			else if (d_action[tid] == 2) {
				if (cstate[tid].x > 0) {
					m = max(qtable[(cstate[tid].x - 1 + cstate[tid].y * 32) * 4], qtable[(cstate[tid].x - 1 + cstate[tid].y * 32) * 4 + 1]);
					m = max(m, qtable[(cstate[tid].x - 1 + cstate[tid].y * 32) * 4 + 2]);
					m = max(m, qtable[(cstate[tid].x - 1 + cstate[tid].y * 32) * 4 + 3]);
					qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2] +=
						a * (rewards[tid] + y * m
							- qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2])
						*e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 2];
				}
			}
			else if (d_action[tid] == 3) {
				if (cstate[tid].y > 0) {
					m = max(qtable[(cstate[tid].x + (cstate[tid].y - 1) * 32) * 4], qtable[(cstate[tid].x + (cstate[tid].y - 1) * 32) * 4 + 1]);
					m = max(m, qtable[(cstate[tid].x + (cstate[tid].y - 1) * 32) * 4 + 2]);
					m = max(m, qtable[(cstate[tid].x + (cstate[tid].y - 1) * 32) * 4 + 3]);
					qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3] +=
						a * (rewards[tid] + y * m
							- qtable[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3])
						*e[(cstate[tid].x + cstate[tid].y * 32) * 4 + 3];
				}
			}
		}
	}
	for (int i = 0; i < 32; i++) {
		e[tid * 32 + i] *= y;
	}
	if (rewards[tid] != 0) {
		act[tid] = 0;
	}
	if (clr[tid] == 1) {
		for (int i = 0; i < idx; i += 3) {
			trace[i + tid * 192] = -1;
			trace[i + tid * 192 + 1] = -1;
			trace[i + tid * 192 + 2] = -1;
		}
	}
	active[tid] = act[tid];
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	cuda_update << <2, 64 >> > (cstate, nstate, rewards, qtable, d_action, active, e, dOut, p, trace, clear);
}