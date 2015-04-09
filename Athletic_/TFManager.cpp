#include "StdAfx.h"
#include "TFManager.h"


TFManager::TFManager(void)
{
	m_tf = NULL;
}


TFManager::~TFManager(void)
{
	if(m_tf != NULL)
		delete[] m_tf;
}


void TFManager::SetTF(int tf_size)
{
	m_tfSize = tf_size;

	m_tf = new TF[m_tfSize];

	int i;
	//-------------------------------------------------------------------
	for(i=0; i<color_start; i++){
		m_tf[i].R = Rcolor_start;
	}

	for(i=color_start; i<color_end; i++){
		m_tf[i].R = (((i-color_start) * ((float)(Rcolor_end - Rcolor_start)/(float)(color_end - color_start))) + Rcolor_start);

	}
		
	for(i=color_end; i<m_tfSize; i++){
		m_tf[i].R = Rcolor_end;
	
	}
	//-------------------------------------------------------------------

	for(i=0; i<color_start; i++){
		m_tf[i].G = Gcolor_start;
	}

	for(i=color_start; i<color_end; i++){
		m_tf[i].G = (((i-color_start) * ((float)(Gcolor_end - Gcolor_start)/(float)(color_end - color_start))) + Gcolor_start);

	}
		
	for(i=color_end; i<m_tfSize; i++){
		m_tf[i].G = Gcolor_end;
	
	}
	//-------------------------------------------------------------------

	for(i=0; i<color_start; i++){
		m_tf[i].B = Bcolor_start;
	}

	for(i=color_start; i<color_end; i++){
		m_tf[i].B = (((i-color_start) * ((float)(Bcolor_end - Bcolor_start)/(float)(color_end - color_start))) + Bcolor_start);

	}
		
	for(i=color_end; i<m_tfSize; i++){
		m_tf[i].B = Bcolor_end;
	
	}
	//-------------------------------------------------------------------

	//-------------------------------------------------------------------

	for(i=0; i<alpha_start; i++){
		m_tf[i].alpha = 0.0;
	}

	for(i=alpha_start; i<alpha_end; i++){
		m_tf[i].alpha = (i-alpha_start)/(float)(alpha_end - alpha_start);

	}
		
	for(i=alpha_end; i<m_tfSize; i++){
		m_tf[i].alpha = 1.0;

	}
	//-------------------------------------------------------------------

}
