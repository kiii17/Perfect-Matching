//compile using nvcc matching_serial.cu -lgsl -lgslcblas -lm


#include <stdio.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_permutation.h>
#include "timerc.h"

void randomGraph(int *graph, int dim){
	for(int i = 0; i<dim; i++){
		for(int j = i+1; j<dim; j++){
			int n = rand()%2;
			graph[i*dim+j] = n;
			graph[j*dim+i] = n;
		}
	}
}

int countEdge(int * graph, int dim){
  int count  = 0;
  for(int i = 0; i<dim*dim; i++){
    if(graph[i]>0){
      count++;
    }
  }
  return count/2;
}

void randomWeight(int * graph, int * newGraph, int dim, int edge){
  srand(time(NULL));
  for(int i = 0; i<dim; i++){
    for(int j = i; j<dim; j++){
      if(graph[i*dim+j]>0){
        newGraph[i*dim+j] = rand()%(2*edge)+1;
        newGraph[j*dim+i] = newGraph[i*dim+j];
      }else{
        newGraph[i*dim+j] = 0;
        newGraph[j*dim+i] = 0;
      }
    }
  }
}

void Tutte(int * graph, double * newGraph, int dim){
  for(int i = 0; i<dim; i++){
    for(int j = 0; j<dim; j++){
      if(graph[i*dim+j]!=0){
	if(i>j){
		newGraph[i*dim+j] = pow(2.0,(double) graph[i*dim+j]);
	}else{
		newGraph[i*dim+j] = -pow(2.0, (double) graph[i*dim+j]);
	}
      }else{
        newGraph[i*dim+j] = 0.0;
      }
    }
  }
}

void pr(int *graph, int dim){
  for(int i = 0; i<dim; i++){
    for(int j = 0; j<dim; j++){
      printf("(%d, %d) = %d\n", i, j, graph[i*dim+j]);
    }
  }
}



int main(void) {
	srand(time(0));
	//dim represents the number of vertices in the graph
	int dim = 4;
	int *graph  = (int *) malloc(dim*dim*sizeof(int));
	randomGraph(graph, dim);
	//pr prints out the adjacency matrix of the randomly generated graph
	pr(graph, dim);
	//count the number of edges in the graph
	int edge = countEdge(graph, dim);
	int *weighted = (int *)malloc(dim*dim*sizeof(int));
	//assign random weight for all edges in the graph
	randomWeight(graph, weighted, dim, edge);
	double *weighted2 = (double *)malloc(dim*dim*sizeof(double));
	//produce the Tutte matrix
	Tutte(weighted, weighted2, dim);
	//copy the Tutte matrix to a gsl_matrix
	gsl_matrix *m = gsl_matrix_alloc(dim, dim);
	for (int i = 0; i<dim; i++){
		for(int j = 0; j<dim; j++){
		gsl_matrix_set(m, i, j, weighted2[i*dim+j]);
		}
	}
	//prepare the sign and permutation matrix for LU decomposition
	int* sign = (int *)malloc(sizeof(int));
	sign[0] = (int) pow(-1, (double) dim);
	gsl_permutation *p = gsl_permutation_alloc(dim);
	gsl_linalg_LU_decomp(m, p, sign);
	//get the determinant from LU decomposition
	double d = gsl_linalg_LU_det(m, 1);
	printf("determinant is %f\n", d);
	if(d==0){
		printf("No perfect matching.\n");
		return 0;
	}
	int d1 = d;
	int b = 0;
	int k;
	//generate the weight b for the perfect matching
	while(1){
		k = d1%4;
		if(k!=0){
			break;
		} 
		d1 = d1/4;
		b++;
	}
	//printf("The weight of the perfect matching is %d\n", b);
	
	gsl_matrix *inv = gsl_matrix_alloc(dim, dim);
	//compute the inverse matrix and store it in inv
	int y = gsl_linalg_LU_invert(m, p, inv);
	//calculation to determine if each edge belong to the perfect matching
	gsl_matrix *madj = gsl_matrix_alloc (dim, dim);
	double temp;
	for (int i = 0; i<dim; i++){
		for(int j=  0; j<dim; j++){
			temp = (double)gsl_matrix_get(inv, i, j);
			gsl_matrix_set(madj, i, j, temp*abs(d1));
		}
	}
	int* result = (int *)malloc(dim*dim*sizeof(int));
	int t;
	double z;
	for (int i = 0; i<dim; i++){
		for(int j = 0; j<dim; j++){
			if(graph[i*dim+j]!=0){
				z = pow(2.0, (double) weighted[i*dim+j]);
				temp = (double)gsl_matrix_get(madj, j, i);
				temp = abs(temp)*z;
				t = (int) temp;
				if(t%2 == 1){
					result[i*dim+j] = 1;
				}
			}			
		}
	}
	printf("Result:\n");
	pr(result, dim);
	return 0;
}
