
// Athletic_View.h : CAthletic_View 클래스의 인터페이스
//

#pragma once
#include "gl/freeglut.h"
#include "gl/gl.h"
#include "gl/glu.h"
#include "Volume.h"

class CAthletic_View : public CView
{
protected: // serialization에서만 만들어집니다.
	CAthletic_View();
	DECLARE_DYNCREATE(CAthletic_View)

// 특성입니다.
public:
	CAthletic_Doc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// 구현입니다.
public:
	virtual ~CAthletic_View();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()

private:
	HGLRC m_hRC;
	HDC m_hDC;
	GLuint m_texName;

	unsigned char *m_imageBuffer;
	int m_ResultImgSize[2];

	Volume m_vol;
	TFManager m_TF;

	float m_Viewing[3];
	float m_Volcenter[3];
	int m_CurrentRenderType;

	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnDestroy();
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);

	void GLResize(int cx, int cy);
	void GLinit(void);
	void GLRenderScene(void);
	
public:
	inline void SetRenderType(int n){
		m_CurrentRenderType = n;
	};
	inline void SetBuffer(unsigned char* image){
		if(image == NULL) return;
		delete[] m_imageBuffer;

		m_imageBuffer = image;
	};
	inline void SetImageSize(int width, int height){
		m_ResultImgSize[0] = width;
		m_ResultImgSize[1] = height;
	}
	inline int* GetImageSize(void){
		return m_ResultImgSize;
	};
	inline void SetVolume(short *den, int dim[3], double range[2], double spacing[3]){
		m_vol.SetVolume(den, dim, range, spacing);
	};
	inline Volume* GetVolume(void){
		return &m_vol;
	}
	inline void SetTF(int size){
		m_TF.SetTF(4096);
	};
	inline TFManager* GetTF(void){
		return &m_TF;
	}
	float degreeX;
	float degreeY;

	float inputMatrix[4][1];
	float rotationMatrix[4][4];
	float outputMatrix[4][1]; 

	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	void MoveLeft(void);
	void MoveRight(void);
	void MoveUp(void);
	void MoveDown(void);
	float* GetViewingPoint(void) { return m_Viewing; }
	void SetViewPoint(float view[3])
	{ m_Viewing[0] = view[0]; m_Viewing[1] = view[1]; m_Viewing[2] = view[2]; }
	void SetVolumeCenter(float center[3])
	{ m_Volcenter[0] = center[0]; m_Volcenter[1] = center[1]; m_Volcenter[2] = center[2]; }
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	void Rotate(float x_, float y_);
	void multiplyMatrix(void);
	void setUpRotationMatrix(float angle, float u, float v, float w);
};

#ifndef _DEBUG  // Athletic_View.cpp의 디버그 버전
inline CAthletic_Doc* CAthletic_View::GetDocument() const
   { return reinterpret_cast<CAthletic_Doc*>(m_pDocument); }
#endif

