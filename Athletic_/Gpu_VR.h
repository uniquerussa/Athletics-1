#pragma once

#include "Volume.h"
#include "TFManager.h"

extern "C"
{
	void GPU_Render(uchar *image, const int imgsize[2], ushort* pVol, int dim[3], 
					TF *transfer, int tf_size, double zResolution, bool &bInitVol, bool &bInitTF,
					const float *ViewingPoint);
	
	void GPU_Render_AO(uchar *image, const int imgsize[2], ushort* pVol, int dim[3], 
					TF *transfer, int tf_size, double zResolution, bool &bInitVol, bool &bInitTF,
					float *Avg, float *Sig, bool &m_bInitAvgSig, float probability[310], float factor[3],
					const float *ViewingPoint);
	bool MakeAverageSigma(ushort* volume, int dim[3], float* Average, float* Sigma, int cubeSize);
	void testSmoothFilter(ushort* pVol, int *dim);
}


class Gpu_VR
{
	
public:
	Gpu_VR(void);
	~Gpu_VR(void);

	unsigned char* VR_basic(Volume *vol, TFManager *tf, const int *imgSize, const float *ViewingPoint);
	unsigned char* VR_AmbientOcclusion(Volume *vol, TFManager *tf, const int *imgSize, const float *ViewingPoint);

};

