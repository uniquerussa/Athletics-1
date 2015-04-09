#include "StdAfx.h"
#include "Cpu_VR.h"
#include <math.h>

Cpu_VR::Cpu_VR(void)
{
}
Cpu_VR::~Cpu_VR(void)
{
}
void Cpu_VR::Crossproduct(float a[3], float b[3], float output[3])
{
	output[0] = a[1] * b[2] - a[2] * b[1];
	output[1] = a[2] * b[0] - a[0] * b[2];
	output[2] = a[0] * b[1] - a[1] * b[0]; 

}
void Cpu_VR::Raydirection(float a[3], float b[3], float output[3])
{
	float temp[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
	float sum = sqrt((float)(temp[0]*temp[0] + temp[1]*temp[1] + temp[2]*temp[2]));

	temp[0] /= sum;
	temp[1] /= sum;
	temp[2] /= sum;

	output[0] = temp[0];
	output[1] = temp[1];
	output[2] = temp[2];

}
void Cpu_VR::Normalize(float a[3], float output[3])
{
	float temp[3] ={a[0], a[1], a[2]};
	float sum = sqrt((float)(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]));

	if(sum == 0){
		temp[0] = 0;
		temp[1] = 0;
		temp[2] = 0;
	}else{
		temp[0] /= sum;
		temp[1] /= sum;
		temp[2] /= sum;
	}

	output[0] = temp[0];
	output[1] = temp[1];
	output[2] = temp[2];
}

void Cpu_VR::GetRayBound(float *t, float sdot[3], float start[3], int * volumeSize)
{

	const float EPS = 0.00001; // epsilon
	// [0,0,0] ~ [255,255,224] box
	// eye : sdot
	// direction : start
	// get t1, t2
	float kx[2]={-20000,20000}, ky[2]={-20000,20000}, kz[2]={-20000,20000};
	// sdot.x + kx[0] * start.x = 0;
	if( fabs((float)start[0]) > EPS) {
		kx[0] = (0 - sdot[0]) / start[0];
		kx[1] = (volumeSize[0] - sdot[0]) / start[0];
		if( kx[0] > kx[1] ) { // in > out
			float temp = kx[0];
			kx[0] = kx[1];
			kx[1] = temp;
		}
	}

	if( fabs((float)start[1]) > EPS){
		ky[0] = (0 - sdot[1]) / start[1];
		ky[1] = (volumeSize[1] - sdot[1]) / start[1];
		if( ky[0] > ky[1] ) { // in > out
			float temp = ky[0];
			ky[0] = ky[1];
			ky[1] = temp;
		}
	}

	if( fabs((float)start[2]) > EPS){
		kz[0] = (0 - sdot[2]) / start[2];
		kz[1] = (volumeSize[2] - sdot[2]) / start[2];
		if( kz[0] > kz[1] ) { // in > out
			float temp = kz[0];
			kz[0] = kz[1];
			kz[1] = temp;
		}
	}

	float kin = max(max(kx[0], ky[0]), kz[0]);
	float kout = min(min(kx[1], ky[1]), kz[1]);

	t[0] = kin + 0.01f;
	t[1] = kout - 0.01f;

}

unsigned char* Cpu_VR::VR_basic(Volume *vol, TFManager *tf, const int imgSize[2], const float* ViewingPoint)
{
	printf("-in CPU Render class\n");
	ushort *pVol = vol->GetDensityPointer();
	int *dim = vol->GetDimension();
		
	int bufferSize = imgSize[0]*imgSize[1]*3;

	uchar *image = new uchar[bufferSize];
	memset(image, 0, sizeof(uchar)*bufferSize);

	TF *transfer = tf->GetTFData();

	float volCenter[3] = {dim[0]/2.0f, dim[1]/2.0f, dim[2]/2.0f};
	
	float svec[3] = {ViewingPoint[0], ViewingPoint[1], ViewingPoint[2]};
	float upvec[3] = {0.0f, 0.0f, -1.0f}; //¾÷º¤ÅÍ!
	float frontView[3] = {dim[0]/2.f, dim[1], dim[2]/2.f};
	float direction[3], crossvec[3], yvec[3], front[3];

	Raydirection(frontView, volCenter, front);
	Raydirection(volCenter, svec, direction);
	float temp_z[3] = {0.f, direction[1], direction[2]};
	if(front[0]*temp_z[0] + front[0]*temp_z[0] +front[0]*temp_z[0] < 0.f)
		upvec[2] = 1.f;

	Crossproduct(upvec, direction, crossvec);
	Normalize(crossvec, crossvec);

	Crossproduct(direction, crossvec, yvec);
	Normalize(yvec, yvec);

	#pragma omp parallel for schedule(dynamic)
	for(int i=0; i<imgSize[0]; i++){
		for(int j=0; j<imgSize[1]; j++){
			
			float sdot[3]={0.0f, 0.0f, 0.0f}; // ±¤¼± ½ÃÀÛÁ¡ (i,j,k);
			sdot[0] = svec[0] + (i-imgSize[0]/2) * crossvec[0] + (j-imgSize[1]/2) * yvec[0];
			sdot[1] = svec[1] + (i-imgSize[0]/2) * crossvec[1] + (j-imgSize[1]/2) * yvec[1];
			sdot[2] = svec[2] + (i-imgSize[0]/2) * crossvec[2] + (j-imgSize[1]/2) * yvec[2];
			
			float t[2] = {0.0f, 1000.0f};
			GetRayBound(t, sdot, direction, dim);

			//struct float4 intensity = {0.0f};
			float alpha = 0.0f;

			float intensity[4] = {0.0f};
			for(float r=t[0]; r<t[1]; r+=1.0f){
				
				float render[3]={0.0f, 0.0f, 0.0f};
				render[0] = sdot[0] + r * direction[0];
				render[1] = sdot[1] + r * direction[1];
				render[2] = sdot[2] + r * direction[2];
				
				if(render[0]>=0 && render[0]< dim[0] && 
				   render[1]>=0 && render[1]< dim[1] && 
				   render[2]>=0 && render[2]< dim[2])
				{ 	
					ushort den = vol->GetDensity(render[0], render[1], render[2]);
					TF smaple = transfer[den];

					float local_shading = getLocalshading(direction,render,dim,pVol);

					smaple.alpha*=local_shading;
					smaple.R*=local_shading;
					smaple.G*=local_shading;
					smaple.B*=local_shading;

					intensity[0] += (1.0f-intensity[3])*smaple.R*smaple.alpha;
					intensity[1] += (1.0f-intensity[3])*smaple.G*smaple.alpha;
					intensity[2] += (1.0f-intensity[3])*smaple.B*smaple.alpha;
					intensity[3] += (1.0f-intensity[3])*smaple.alpha;

					if(intensity[3] > 0.95f)
						break;
				}

			}
			//unsigned short input = ((float)max/16.f); //0-255
			image[(j*imgSize[0] + i)*3 +0] = (uchar)intensity[0];
			image[(j*imgSize[0] + i)*3 +1] = (uchar)intensity[1];
			image[(j*imgSize[0] + i)*3 +2] = (uchar)intensity[2];
			//printf("%d %d %d\n", buffer[(j*height + i)*3 +0], buffer[(j*height + i)*3 +1], buffer[(j*height + i)*3 +2]);
		}
	}

	return image;

}

float Cpu_VR::getLocalshading(float directionVector[3], float render[3], int dim[3], ushort * pVol)
{
	float nomal; float amb = 0.3f;
	float diff=0.7f;
	float lx =directionVector[0], ly = directionVector[1], lz = directionVector[2];
	float nx=0.0f, ny=0.0f, nz=0.0f;
	float local_shading=0;
 
	int pre_x,next_x;
	int pre_y,next_y;
	int pre_z,next_z;
 
	pre_x = int(render[0])-1;
	if( pre_x < 0 )
		pre_x = 0;
	next_x = int(render[0])+1;
	if( next_x > dim[0]-1 )
		next_x = dim[0]-1;
			
	pre_y = int(render[1])-1;
	if( pre_y < 0 )
		pre_y = 0;
	next_y = int(render[1])+1;
	if( next_y > dim[1]-1)
		next_y = dim[1]-1;
			
	pre_z = int(render[2])-1;
	if( pre_z < 0 )
		pre_z = 0;
	next_z = int(render[2])+1;
	if( next_z > dim[2]-1 )
		next_z = dim[2]-1;

	nx = (pVol[(int)render[2]*dim[0]*dim[1] + dim[0]*int(render[1]) + next_x] - pVol[int(render[2])*dim[0]*dim[1] + dim[0]*int(render[1]) + pre_x] )/2.0;
	ny = (pVol[(int)render[2]*dim[0]*dim[1] + dim[0]*next_y + int(render[0])] - pVol[int(render[2])*dim[0]*dim[1] + dim[0]*pre_y + int(render[0])] )/2.0;
	nz = (pVol[next_z*dim[0]*dim[1] + dim[0]*int(render[1]) + int(render[0])] - pVol[pre_z*dim[0]*dim[1] + dim[0]*int(render[1]) + int(render[0])] )/2.0;

	nomal = sqrt(pow(nx,2)+pow(ny,2)+pow(nz,2));
	if(nomal==0.0f){
		nx=0.0f;
		ny=0.0f;
		nz=0.0f;
	}
	else{
		nx=nx / nomal;
		ny=ny / nomal;
		nz=nz / nomal;
	}
	float NL = nx*lx + ny*ly + nz*lz;
	
	if(NL <=0.0f){
		NL = 0.0f;
	}
	local_shading = amb + (diff * NL);
	
	return local_shading;
	
}
