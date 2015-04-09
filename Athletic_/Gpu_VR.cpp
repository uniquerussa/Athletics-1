#include "StdAfx.h"
#include "Gpu_VR.h"


Gpu_VR::Gpu_VR(void)
{

}


Gpu_VR::~Gpu_VR(void)
{
}

unsigned char* Gpu_VR::VR_basic(Volume *vol, TFManager *tf, const int *imgSize, const float *ViewingPoint)
{
	printf("-in GPU Render class\n");
	ushort *pVol = vol->GetDensityPointer();
	int *dim = vol->GetDimension();
	double* spacing = vol->GetVoxelSpacing();
	double zResolution = spacing[0]/spacing[2];

	int bufferSize = imgSize[0]*imgSize[1]*3;

	uchar *image = new uchar[bufferSize];
	memset(image, 0, sizeof(uchar)*bufferSize);

	TF *transfer = tf->GetTFData();
	int tf_size = tf->GetSize();
	
	GPU_Render(image, imgSize, pVol, dim, transfer, tf_size, zResolution, vol->m_bInitVolumeInGPU, vol->m_bInitTFInGPU, ViewingPoint);

	return image;
}


unsigned char* Gpu_VR::VR_AmbientOcclusion(Volume *vol, TFManager *tf, const int *imgSize, const float *ViewingPoint)
{
	printf("-in GPU AO Render class\n");
	ushort *pVol = vol->GetDensityPointer();
	int *dim = vol->GetDimension();
	double* spacing = vol->GetVoxelSpacing();
	double zResolution = spacing[0]/spacing[2];

	if(!vol->m_bLoadProb){
		vol->LoadProbability();
		vol->m_bLoadProb = true;
	}
	
	float* Average = vol->GetAveragePointer();
	float* Sigma = vol->GetSigmaPointer();
	if(!vol->m_bInitAvgSigInGPU){
		Average = new float[dim[0]*dim[1]*dim[2]];
		memset(Average, 0, sizeof(float)*dim[0]*dim[1]*dim[2]);
		Sigma = new float[dim[0]*dim[1]*dim[2]];
		memset(Sigma, 0, sizeof(float)*dim[0]*dim[1]*dim[2]);

		if(!MakeAverageSigma(pVol, dim, Average, Sigma, 15))
			return NULL;
	}

	if(!vol->m_bSmooth){
		testSmoothFilter(pVol, dim);
		vol->m_bSmooth = true;
	}

	int bufferSize = imgSize[0]*imgSize[1]*3;

	uchar *image = new uchar[bufferSize];
	memset(image, 0, sizeof(uchar)*bufferSize);

	TF *transfer = tf->GetTFData();
	int tf_size = tf->GetSize();
	
	float factor[3] = {0.15f, 0.45f, 0.4f};
	GPU_Render_AO(image, imgSize, pVol, dim, transfer, tf_size, zResolution, 
		vol->m_bInitVolumeInGPU, vol->m_bInitTFInGPU, Average, Sigma, vol->m_bInitAvgSigInGPU, vol->GetProbability(), factor, ViewingPoint);

	if(!vol->m_bInitAvgSigInGPU){
		if(Average){
			delete[] Average;
			Average = NULL;
		}
		if(Sigma){
			delete[] Sigma;
			Sigma = NULL;
		}

		vol->m_bInitAvgSigInGPU=true;
	}

	return image;
}

