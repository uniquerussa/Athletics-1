#pragma once
#include "Volume.h"
#include "TFManager.h"

typedef unsigned char uchar;

class Cpu_VR
{
	private:
		void Crossproduct(float a[3], float b[3], float output[3]);
		void Raydirection(float a[3], float b[3], float output[3]);
		void Normalize(float a[3], float output[3]);
		void GetRayBound(float *t, float sdot[3], float start[3], int * volumeSize);
		float getLocalshading(float directionVector[3], float render[3],int dim[3], ushort * pVol);
	public:
		Cpu_VR(void);
		~Cpu_VR(void);

	public:
		unsigned char* VR_basic(Volume *vol, TFManager *tf, const int *imgSize, const float *ViewingPoint);
		
		
};

