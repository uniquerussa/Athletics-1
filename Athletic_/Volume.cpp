#include "StdAfx.h"
#include "Volume.h"

Volume::Volume(void)
{
	m_density = NULL;
	m_Average = NULL;
	m_Sigma = NULL;
	m_slice_ptr = NULL;

	m_bInitVolumeInGPU = false;
	m_bInitTFInGPU = false;
	m_bInitAvgSigInGPU = false;
	m_bLoadProb = false;
	m_bSmooth = false;

}

Volume::Volume(short *den, int dim[3], double range[2])
{
	int size = dim[0]*dim[1]*dim[2];
	m_density = new ushort[size];
	

	for(int i=0; i<size; i++){
		m_density[i] = (ushort)(den[i]-range[0]);
	}

	m_dim[0] = dim[0];
	m_dim[1] = dim[1];
	m_dim[2] = dim[2];

}

Volume::~Volume(void)
{
	//printf("소멸자\n");
	if(m_density != NULL)
		delete[] m_density;
	if(m_Average != NULL)
		delete[] m_Average;
	if(m_Sigma != NULL)
		delete[] m_Sigma;
	if(m_slice_ptr != NULL){
		delete[] m_slice_ptr;
	}

	if(m_bInitVolumeInGPU){
		FreeGPUVolArray();
	}
	if(m_bInitTFInGPU){
		FreeGPUTFArray();
	}
	if(m_bInitAvgSigInGPU){
		FreeGPUEtcArray();
	}
}


void Volume::SetVolume(short *den, int dim[3], double range[2], double spacing[3])
{
	int size = dim[0]*dim[1]*dim[2];
	m_density = new ushort[size];
	m_slice_ptr = new ushort*[dim[2]];

	int nCount=0;
	for(int i=0; i<size; i++){
		m_density[i] = (ushort)(den[i]-range[0]);
		
		if(i % (dim[0]*dim[1]) == 0){
			m_slice_ptr[nCount++] =  &m_density[i]; 
		}
	}

	m_dim[0] = dim[0];
	m_dim[1] = dim[1];
	m_dim[2] = dim[2];

	m_spacing_voxel[0] = spacing[0];
	m_spacing_voxel[1] = spacing[1];
	m_spacing_voxel[2] = spacing[2];

}

void Volume::SetVolume(ushort *den, int dim[3])
{
	int size = dim[0]*dim[1]*dim[2];
	m_density = new ushort[size];

	for(int i=0; i<size; i++)
		m_density[i] = (ushort)(den[i]);

	m_dim[0] = dim[0];
	m_dim[1] = dim[1];
	m_dim[2] = dim[2];

}

void Volume::DeepCopy(Volume *output)
{
	output->~Volume();
	
	output->SetVolume(m_density, m_dim);
}

ushort Volume::GetDensity(float x, float y, float z)
{
	int next_x,next_y,next_z;
	int intx = (int)x; 
	int inty = (int)y;
	int intz = (int)z;
	
	float weightx = x - intx;
	float weighty = y - inty;
	float weightz = z - intz;

	if(x>=m_dim[0]-1)
	   next_x=m_dim[0]-1;
	else
	   next_x=(intx+1);
	
	if(y>=m_dim[1]-1)
	   next_y=m_dim[1]-1;
	else
	   next_y=(inty+1);
	
	if(z>=m_dim[2]-1)
	   next_z=m_dim[2]-1;
	else
	   next_z=(intz+1);

	return (ushort)
		(
		(
		m_density[intz * m_dim[0] * m_dim[1] + inty * m_dim[0] + intx] * (1.f-weightx) + 
		m_density[intz * m_dim[0] * m_dim[1] + inty * m_dim[0] + next_x] * weightx 
		) * (1-weighty) +
		(
		m_density[intz * m_dim[0] * m_dim[1] +  next_y * m_dim[0] + intx] * (1.f-weightx) +
		m_density[intz * m_dim[0] * m_dim[1] +  next_y * m_dim[0] + next_x] * weightx
		) * weighty
	    ) * (1.f-weightz)+
		(	
	    (
		m_density[ next_z * m_dim[0] * m_dim[1] + inty * m_dim[0] + intx] * (1.f-weightx) +
		m_density[ next_z * m_dim[0] * m_dim[1] + inty * m_dim[0] +  next_x] * weightx
		) * (1.f-weighty) +
		(
		m_density[next_z * m_dim[0] * m_dim[1] + next_y * m_dim[0] + intx] * (1.f-weightx) +
		m_density[next_z * m_dim[0] * m_dim[1] + next_y * m_dim[0] + next_x] * weightx
		) * weighty
		) * weightz;
}


Volume *Volume::GetVolume(void)
{
	Volume *pVol;
	pVol->m_density=this->m_density;
	pVol->m_dim[3] = this->m_dim[3];
	pVol->m_spacing_voxel[3] = this->m_spacing_voxel[3];

	return pVol;
}


void Volume::LoadProbability()
{
	FILE *fp = fopen("data/probability.ini", "rb");
    if (!fp) {
        printf("probability 파일 입력실패\n");
        return;
    }

	for(int i=0; i<310; i++)
		fscanf(fp, "%f", &m_probability[i]);
	
	fclose(fp);
}
