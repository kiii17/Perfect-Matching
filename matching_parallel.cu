//compile using nvcc matching_parallel.cu -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include "timerc.h"

void randomGraph(int *graph, int dim){
	for(int i = 0; i<dim; i++){
		for(int j = i; j<dim; j++){
			int n = rand()%2;
			graph[i*dim+j] = n;
			graph[j*dim+i] = n;
			if(i==j) graph[i*dim+j] = 0;
		}
	}
}

int countEdge(int *graph, int dim){
	int count = 0;
	for(int i = 0; i<dim*dim; i++){
		if(graph[i]>0){
			count++;
		}
	}
	return count/2;
}

void pr(int *graph, int dim){
	for(int i = 0; i<dim; i++){
		for(int j = 0; j<dim; j++){
			printf("(%d, %d) = %d\n", i, j, graph[i*dim+j]);
		}
		printf("\n");
	}
}

void prF(float *graph, int dim){
	for(int i = 0; i<dim; i++){
		for(int j = 0; j<dim; j++){
			printf("(%d, %d) = %f\n", i, j, graph[i*dim+j]);
		}
		printf("\n");
	}
}


__global__ void setup_kernel(curandState *state, unsigned long seed){
	int id = threadIdx.x+blockIdx.x*blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void randomWeight(int *graph, int dim, int edge, curandState* globalState){
	int Tutte = 0;
	if(blockIdx.x<threadIdx.x && graph[blockIdx.x*dim+threadIdx.x]!=0){	
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		curandState localState = globalState[id];
		float RANDOM = curand_uniform(&localState);
		globalState[id] = localState;
		int number  = RANDOM*2*edge+1;
		graph[blockIdx.x*dim+threadIdx.x] = number;
		Tutte = (int) pow(2.0, graph[id]);
		graph[id] = Tutte;
		graph[threadIdx.x*dim+blockIdx.x] = -Tutte;	
	}
}

__global__ void adjoint(int *graph1, float *graph, int *result, int dim, float det, int weight){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	float temp = ((float)graph1[id]) *graph[id]*det;	
	int t = temp;
	if(t<0) {t = -t;}
	if(t%2==1){
		result[id] = 1;
	}else{
		result[id] = 0;
	}

}




int main(){
	srand(time(0));
	//dim represents the number of vertices in the graph
	int dim = 4;
	int *graph = (int *)malloc(dim*dim*sizeof(int));
	randomGraph(graph, dim);
	//pr prints out the adjacency matrix of the randomly generated graph
	pr(graph, dim);
	//count the number of edges in the graph
	int edge = countEdge(graph, dim);
	
	int *dev_a;
	cudaMalloc((void**)&dev_a, dim*dim*sizeof(int));
	cudaMemcpy(dev_a, graph, dim*dim*sizeof(int), cudaMemcpyHostToDevice);

	curandState *devStates;
	cudaMalloc(&devStates, sizeof(curandState));
	int seed  = rand();
	//set up random number generator
	setup_kernel<<<dim, dim>>>(devStates,seed);
	//assign random weight and produce the Tutte matrix
	randomWeight<<<dim, dim>>>(dev_a, dim, edge, devStates);	
	int *weighted = (int *)malloc(dim*dim*sizeof(int));
	cudaMemcpy(weighted, dev_a, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	//copy the int matrix to a float matrix
	float *weightedF = (float *)malloc(dim*dim*sizeof(float));
	for(int i = 0; i<dim*dim; i++){
		weightedF[i] = (float) weighted[i];
	}
	float *dev_b;
	cudaMalloc((void **)&dev_b, dim*dim*sizeof(float));
	cudaMemcpy(dev_b, weightedF, dim*dim*sizeof(float), cudaMemcpyHostToDevice);
	//allocate space and prepare for LU decompposition	
	cublasHandle_t hdl;
	cublasCreate(&hdl);		
	int *info;
	cudaMalloc((void **)&info, sizeof(int));
	int *infoH =(int *)malloc(sizeof(int)); 
	int batch = 1;
	int *p;
	cudaMalloc((void**)&p, dim*sizeof(int)); 
	float **ha = (float **)malloc(sizeof(float *));
	ha[0] = dev_b;
	float **a;
	cudaMalloc((void**)&a, sizeof(float *));
	cudaMemcpy(a, ha, sizeof(float *), cudaMemcpyHostToDevice);
	//calculate LU decomposition
	cublasSgetrfBatched(hdl, dim, a, dim, p, info, batch);
	cudaMemcpy(infoH, info, sizeof(int), cudaMemcpyDeviceToHost);	
	//copy LU decomposition to host
	cudaMemcpy(weightedF, dev_b, dim*dim *sizeof(float), cudaMemcpyDeviceToHost);
	//calculate determinant
	float d = 1;	
	for(int i = 0; i<dim; i++){
		d = d*weightedF[i*dim+i];
	}
	printf("det on GPU: %f\n", d);
	if(d==0){
		printf("No perfect matching.\n");
		return 0;
	}	
	int d1 =(int) d;
	int i = 1;
	int k;
	//calculate the weight i of the perfect matching
	while(1){
		k = d1%4;
		if(k!=0) break;
		d1 = d1/4;
		i++;
	}	
	//allocate space for the inverse matrix
	float **hc = (float **)malloc(sizeof(float *));
	float **c, *c1;
	cudaMalloc((void **)&c, sizeof(float *));
	cudaMalloc((void **)&c1, dim*dim*sizeof(float));
	hc[0] = c1;
	cudaMemcpy(c, hc, sizeof(float *), cudaMemcpyHostToDevice);
	//calculate the inverse matrix
	cublasSgetriBatched(hdl, dim, a, dim, p, c, dim, info, batch);
	//copy the inverse matrix to host
	cudaMemcpy(weightedF, c1, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
	//allocate space for the result matrix	
	int *resultH  = (int *)malloc(dim*dim*sizeof(int));
	int *result;
	cudaMalloc((void **)&result, dim*dim*sizeof(int));
	cudaMemcpy(resultH, result, dim*dim*sizeof(int), cudaMemcpyHostToDevice);
	//launch kernels to determine final result
	adjoint<<<dim, dim>>>(dev_a, c1, result, dim, d1, i);
	
	cudaMemcpy(resultH, result, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	printf("Results:\n");
	pr(resultH, dim);
	cudaFree(dev_a);
	return 0;
}
