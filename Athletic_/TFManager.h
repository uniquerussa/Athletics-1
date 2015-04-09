#pragma once
const int color_start = 0, color_end = 1;
const int alpha_start = 1200, alpha_end = 1400;
const int Rcolor_start = 192, Rcolor_end = 255;
const int Gcolor_start = 128, Gcolor_end = 255;
const int Bcolor_start = 128, Bcolor_end = 255;

//TF°ü¸®

struct TF{
	float R;	//0-255
	float G;	//0-255
	float B;	//0-255
	float alpha; //0-1
};		

class TFManager
{
	
public:
	TFManager(void);
	~TFManager(void);

	void SetTF(int tf_size);
	TF* GetTFData(void) { return m_tf; }

	int GetSize(void) { return m_tfSize; }

private:
	TF *m_tf;
	int m_tfSize;
};
