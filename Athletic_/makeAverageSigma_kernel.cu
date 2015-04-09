
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <helper_math.h>
#include <helper_cuda.h>
#include <stdio.h>

typedef unsigned short ushort;
typedef unsigned short uchar;

int fx, fy, fz;
size_t size_ushort;
size_t size_float;
size_t g_size;
int maskSize;

__global__ void Sqkernel(ushort* volume_p, float* sqvolume_p, const int fx, const int fy, const int fz)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= fx || ty >= fy) return;

	float temp=0;

	for(int i=0; i<fz; i++){
		temp = (float)volume_p[i*fx*fy + ty*fx + tx];

		sqvolume_p[i*fx*fy + ty*fx + tx] = temp*temp;
	}

}

__global__ void Sqkernel_float(float* volume_p, float* sqvolume_p, const int fx, const int fy, const int fz)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= fx || ty >= fy) return;

	float temp=0;

	for(int i=0; i<fz; i++){
		temp = volume_p[i*fx*fy + ty*fx + tx];

		sqvolume_p[i*fx*fy + ty*fx + tx] = temp*temp;
	}

}

__global__ void makeLineAverage(ushort* volume_p, float *lineAverage, const int fx, const int fy, const int fz, const int maskSize)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= fx || ty >= fy) return;

	float sum=0.0f;
	int size = (maskSize+1)/2;
	float Divsize = (float)size;

	for(int i=0; i<size; i++)
		sum += (float)volume_p[i*fx*fy + ty*fx + tx];
	
	lineAverage[ty*fx + tx] = sum/Divsize;


	for(int i=1; i<fz; i++){
		if(i-size >= 0){
			sum -= (float)volume_p[(i-size)*fx*fy + ty*fx + tx];
			Divsize--;
		}
		if(i+size-1 < fz){
			sum += (float)volume_p[(i+size-1)*fx*fy + ty*fx + tx];
			Divsize++;
		}

		lineAverage[i*fx*fy + ty*fx + tx] = sum/Divsize;
	}

}

__global__ void makeSQ_lineAverage(float* volume_p, float *lineAverage, const int fx, const int fy, const int fz, const int maskSize)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= fx || ty >= fy) return;

	float sum=0.0f;
	int size = (maskSize+1)/2;
	float Divsize = (float)size;

	for(int i=0; i<size; i++)
		sum += volume_p[i*fx*fy + ty*fx + tx];
	
	lineAverage[ty*fx + tx] = sum/Divsize;


	for(int i=1; i<fz; i++){
		if(i-size >= 0){
			sum -= volume_p[(i-size)*fx*fy + ty*fx + tx];
			Divsize--;
		}
		if(i+size-1 < fz){
			sum += volume_p[(i+size-1)*fx*fy + ty*fx + tx];
			Divsize++;
		}

		lineAverage[i*fx*fy + ty*fx + tx] = sum/Divsize;
	}

}

__global__ void makeSideAverage(float* lineAverage, float* SideAverage, const int fx, const int fy, const int fz, const int maskSize)
{
	int tz = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int tx = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tz >= fz || tx >= fx) return;

	float sum=0.0f;
	int size = (maskSize+1)/2;
	float Divsize = (float)size;

	for(int i=0; i<size; i++)
		sum += lineAverage[tz*fx*fy + i*fx + tx];
	
	SideAverage[tz*fx*fy + tx] = sum/Divsize;

	for(int i=1; i<fy; i++){
		if(i-size >= 0){
			sum -= lineAverage[tz*fx*fy + (i-size)*fx + tx];
			Divsize--;
		}
		if(i+size-1 < fy){
			sum += lineAverage[tz*fx*fy + (i+size-1)*fx + tx];
			Divsize++;
		}

		SideAverage[tz*fx*fy + i*fx + tx] = sum/Divsize;
	}

}

__global__ void makeCubeAverage(float* SideAverage, float* CubeAverage, const int fx, const int fy, const int fz, const int maskSize)
{
	int ty = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int tz = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (ty >= fy || tz >= fz) return;

	float sum=0.0f;
	int size = (maskSize+1)/2;
	float Divsize = (float)size;

	for(int i=0; i<size; i++)
		sum += SideAverage[tz*fx*fy + ty*fx + i];
	
	CubeAverage[tz*fx*fy + ty*fx] = sum/Divsize;

	for(int i=1; i<fx; i++){
		if(i-size >= 0){
			sum -= SideAverage[tz*fx*fy + ty*fx + (i-size)];
			Divsize--;
		}
		if(i+size-1 < fx){
			sum += SideAverage[tz*fx*fy + ty*fx + (i+size-1)];
			Divsize++;
		}

		CubeAverage[tz*fx*fy + ty*fx + i] = sum/Divsize;
	}

}

__global__ void minus_kernel(float* knSigmaVolume, float* SQ_CubeAverage, float* CubeAverage_SQ, const int fx, const int fy, const int fz)
{
	int tx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int ty = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	if (tx >= fx || ty >= fy) return;

	float temp;
	for(int i=0; i<fz; i++){
		temp = SQ_CubeAverage[i*fx*fy + ty*fx + tx] - CubeAverage_SQ[i*fx*fy + ty*fx + tx];
		temp = max(temp, 0.0f);
		knSigmaVolume[i*fx*fy + ty*fx + tx] = sqrt(temp);
	}

}

extern "C"
bool MakeAverageSigma(ushort* volume, int dim[3], float* Average, float* Sigma, int cubeSize)
{
	fx=dim[0];
	fy=dim[1];
	fz=dim[2];
	size_ushort = fx*fy*fz*sizeof(ushort);
	size_float = fx*fy*fz*sizeof(float);
	g_size = fx*fy*fz;
	maskSize = cubeSize;

	ushort *volume_p;	
	checkCudaErrors(cudaMalloc((void**)&volume_p, size_ushort));
	checkCudaErrors(cudaMemcpy(volume_p, volume, size_ushort, cudaMemcpyHostToDevice));
	
	dim3 Dbx = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dgx = dim3((fy+Dbx.x-1)/Dbx.x, (fz+Dbx.y-1)/Dbx.y);

	dim3 Dby = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dgy = dim3((fz+Dby.x-1)/Dby.x, (fx+Dby.y-1)/Dby.y);

	dim3 Dbz = dim3(32, 32);		// block dimensions are fixed to be 512 threads
    dim3 Dgz = dim3((fx+Dbz.x-1)/Dbz.x, (fy+Dbz.y-1)/Dbz.y);

	//----------------------------------------------------------------------
	//¿øº¼·ý - ushort
	//Á¦°öº¼·ý - float

	//----------------------------------------------------------------------
	//¿øº¼·ýºÎÅÍ 
	float *lineAverage;
	checkCudaErrors(cudaMalloc((void**)&lineAverage, size_float));
	checkCudaErrors(cudaMemset(lineAverage, 0, size_float));

	float *lineAverage_p = new float[size_float];
	memset((void*)lineAverage_p, 0, size_float);

	printf("-makeLineAverage...\n");
	makeLineAverage<<<Dgz, Dbz>>>(volume_p, lineAverage, fx, fy, fz, maskSize); //¼± Æò±Õ
	if (cudaGetLastError() != cudaSuccess){
        printf("makeLineAverage() failed to launch error = %d\n", cudaGetLastError()); 
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeLineAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeLineAverage %s \n", str); //debug

	cudaMemcpy(lineAverage_p, lineAverage, size_float, cudaMemcpyDeviceToHost);


	//for(int i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", lineAverage_p[i]);
	//printf("\n");
	checkCudaErrors(cudaFree(volume_p));

	//saveFileAverage(lineAverage_p);

	float *SideAverage;
	checkCudaErrors(cudaMalloc((void**)&SideAverage, size_float));
	checkCudaErrors(cudaMemset(SideAverage, 0, size_float));

	float *SideAverage_p = new float[size_float];
	memset((void*)SideAverage_p, 0, size_float);

	printf("-makeSideAverage...\n");
	makeSideAverage<<<Dgy, Dby>>>(lineAverage, SideAverage, fx, fy, fz, maskSize);
	if (cudaGetLastError() != cudaSuccess){
        printf("makeSideAverage() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");	
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeSideAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeSideAverage %s \n", str); //debug
	
	cudaMemcpy(SideAverage_p, SideAverage, size_float, cudaMemcpyDeviceToHost);

	//for(int i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", SideAverage_p[i]);
	//printf("\n");
	//saveFileAverage(SideAverage_p);

	checkCudaErrors(cudaFree(lineAverage));

	float *CubeAverage;
	checkCudaErrors(cudaMalloc((void**)&CubeAverage, size_float));
	checkCudaErrors(cudaMemset(CubeAverage, 0, size_float));

	printf("-makeCubeAverage...\n");
	makeCubeAverage<<<Dgx, Dbx>>>(SideAverage, CubeAverage, fx, fy, fz, maskSize);
	if (cudaGetLastError() != cudaSuccess){
        printf("makeCubeAverage() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");	
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeCubeAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeCubeAverage %s \n", str); //debug

	checkCudaErrors(cudaMemcpy(Average, CubeAverage, size_float, cudaMemcpyDeviceToHost));

	//for(int i=fx*fy; i<fx*fy+fz; i++)
	//	printf("%.1f ", CubeAverage_p[i]); 
	//printf("\n");
	checkCudaErrors(cudaFree(SideAverage));

	//----------------------------------------------------------------------
	//¿ø º¼·ýÀÇ 7*7*7 Å¥ºê ¿¡¹ö¸®Áö - CubeAverage
	printf("Making Average Success\n");

	//ÀÌÁ¦ Á¦°öº¼·ý

	checkCudaErrors(cudaMalloc((void**)&volume_p, size_ushort));
	checkCudaErrors(cudaMemcpy(volume_p, volume, size_ushort, cudaMemcpyHostToDevice));

	float *sqvolume_p;
	checkCudaErrors(cudaMalloc((void**)&sqvolume_p, size_float));
	checkCudaErrors(cudaMemset(sqvolume_p, 0, size_float));

	float *sqvolume = new float[g_size];
	memset((void*)sqvolume, 0, size_float);

	printf("-Sqkernel...\n");
	Sqkernel<<<Dgz, Dbz>>>(volume_p, sqvolume_p, fx, fy, fz); 
	if (cudaGetLastError() != cudaSuccess){
        printf("Sqkernel() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("Sqkernel %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("Sqkernel %s \n", str); //debug

	checkCudaErrors(cudaMemcpy(sqvolume, sqvolume_p, size_float, cudaMemcpyDeviceToHost));

	//for(int  i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", sqvolume[i]);

	checkCudaErrors(cudaFree(volume_p));


	float *SQ_lineAverage;
	checkCudaErrors(cudaMalloc((void**)&SQ_lineAverage, size_float));
	checkCudaErrors(cudaMemset(SQ_lineAverage, 0, size_float));

	printf("-makeSQ_lineAverage...\n");
	makeSQ_lineAverage<<<Dgz, Dbz>>>(sqvolume_p, SQ_lineAverage, fx, fy, fz, maskSize);
	if (cudaGetLastError() != cudaSuccess){
        printf("makeSQ_lineAverage() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeSQ_lineAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeSQ_lineAverage %s \n", str); //debug

	checkCudaErrors(cudaFree(sqvolume_p));


	float *SQ_SideAverage;
	checkCudaErrors(cudaMalloc((void**)&SQ_SideAverage, size_float));
	checkCudaErrors(cudaMemset(SQ_SideAverage, 0, size_float));

	printf("-makeSideAverage...\n");
	makeSideAverage<<<Dgy, Dby>>>(SQ_lineAverage, SQ_SideAverage, fx, fy, fz, maskSize);
	if (cudaGetLastError() != cudaSuccess){
        printf("makeSideAverage() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeSideAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeSideAverage %s \n", str); //debug

	checkCudaErrors(cudaFree(SQ_lineAverage));

	float *SQ_CubeAverage;
	checkCudaErrors(cudaMalloc((void**)&SQ_CubeAverage, size_float));
	checkCudaErrors(cudaMemset(SQ_CubeAverage, 0, size_float));

	float *SQ_CubeAverage_p = new float[g_size];
	memset((void*)SQ_CubeAverage_p, 0, size_float);

	printf("-makeCubeAverage...\n");
	makeCubeAverage<<<Dgx, Dbx>>>(SQ_SideAverage, SQ_CubeAverage, fx, fy, fz, maskSize);
	if (cudaGetLastError() != cudaSuccess){
        printf("makeCubeAverage() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("makeCubeAverage %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("makeCubeAverage %s \n", str); //debug

	checkCudaErrors(cudaMemcpy(SQ_CubeAverage_p, SQ_CubeAverage, size_float, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(SQ_SideAverage));

	//for(int  i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", SQ_CubeAverage_p[i]);

	//----------------------------------------------------------------------
	//Á¦°ö º¼·ýÀÇ 7*7*7 Å¥ºê ¿¡¹ö¸®Áö - SQ_CubeAverage

	float *CubeAverage_SQ;
	checkCudaErrors(cudaMalloc((void**)&CubeAverage_SQ, size_float));
	checkCudaErrors(cudaMemset(CubeAverage_SQ, 0, size_float));

	float *CubeAverage_SQ_p = new float[g_size];
	memset((void*)CubeAverage_SQ_p, 0, size_float);

	printf("-Sqkernel_float...\n");
	Sqkernel_float<<<Dgz, Dbz>>>(CubeAverage, CubeAverage_SQ, fx, fy, fz);
	if (cudaGetLastError() != cudaSuccess){
        printf("Sqkernel_float() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("Sqkernel_float %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("Sqkernel_float %s \n", str); //debug

	checkCudaErrors(cudaMemcpy(CubeAverage_SQ_p, CubeAverage_SQ, size_float, cudaMemcpyDeviceToHost));

	//for(int  i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", CubeAverage_SQ_p[i]);

	checkCudaErrors(cudaFree(CubeAverage));


	float *knSigmaVolume;
	checkCudaErrors(cudaMalloc((void**)&knSigmaVolume, size_float));
	checkCudaErrors(cudaMemset(knSigmaVolume, 0, size_float));

	printf("-minus_kernel...\n");
	minus_kernel<<<Dgz, Dbz>>>(knSigmaVolume, SQ_CubeAverage, CubeAverage_SQ, fx, fy, fz);
	if (cudaGetLastError() != cudaSuccess){
        printf("minus_kernel() failed to launch error = %d\n", cudaGetLastError());
		return false;
	}
	//printf("\n");
	//str = cudaGetErrorString(cudaPeekAtLastError());
	//printf("minus_kernel %s \n", str);
	//str = cudaGetErrorString(cudaThreadSynchronize());
	//printf("minus_kernel %s \n", str); //debug
	
	checkCudaErrors(cudaMemcpy(Sigma, knSigmaVolume, size_float, cudaMemcpyDeviceToHost));

	//for(int  i=(fx*fy*fz)-fz; i<fx*fy*fz; i++)
	//	printf("%.1f ", SigmaVolume[i]); 

	printf("Making Sigma Success\n");

	checkCudaErrors(cudaFree(knSigmaVolume));
	checkCudaErrors(cudaFree(CubeAverage_SQ));
	checkCudaErrors(cudaFree(SQ_CubeAverage));
	
	delete[] SQ_CubeAverage_p;
	delete[] CubeAverage_SQ_p;

	return true;
}
