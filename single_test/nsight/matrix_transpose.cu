#include<iostream>
#include<stdio.h>
#include<cstdlib>

__global__ void transposeNative(float *input, float *output, int m, int n)
{
    int colID_input = threadIdx.x + blockDim.x*blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y*blockIdx.y;

    if (rowID_input < m && colID_input < n)
    {
        int index_input  = colID_input + rowID_input*n;
        int index_output = rowID_input + colID_input*m;

        output[index_output] = input[index_input];
    }
}

__global__ void kernel_init(float *input, int m, int n)
{
    int colID_input = threadIdx.x + blockDim.x*blockIdx.x;
    int rowID_input = threadIdx.y + blockDim.y*blockIdx.y;
    if (rowID_input <m && colID_input<n)
    {	    
	int index_input = colID_input + rowID_input*n;
	    input[index_input] = index_input%100;
    }
}

int main(int argc,char**argv)
{
	int m = 8192;
	int n = 4096;
	
	float *h_input, *h_output;
	float *d_input, *d_output;
	
	h_input = new float[m*n];
	h_output = new float[m*n];

	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			h_input[j+i*n] = rand()%100;
		}
	}

	printf("\nhere\n");	
	cudaMalloc((void**)&d_input, sizeof(float)*m*n);
	cudaMalloc((void**)&d_output,sizeof(float)*m*n);
	
	dim3 block(32,32);
	dim3 grid((n+31)/32,(m+31)/32);
	
	kernel_init<<<grid,block>>>(d_input,m,n);

	cudaMemcpy(d_input,h_input,sizeof(float)*m*n,cudaMemcpyHostToDevice);

	transposeNative<<<grid,block>>>(d_input,d_output,m,n);
	
	cudaMemcpy(h_output,d_output,sizeof(float)*m*n,cudaMemcpyDeviceToHost);

	for(int i=0;i<m;i++)
	for(int j=0;j<n;j++)
	if(h_output[i+j*m] != h_input[j+i*n]) 
	{
		printf("i=%d, j=%d,value=%f,value=%f\n",i,j,h_output[i+j*m],h_input[j+i*n]);
		printf("Wrong!!!\n");
		return 0;
	}
	printf("Right!!!!\n");
	return 0;
}

