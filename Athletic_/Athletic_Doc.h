
// Athletic_Doc.h : CAthletic_Doc 클래스의 인터페이스
//


#pragma once
#include "Volume.h"
#include "Cpu_VR.h"
#include "Gpu_VR.h"
class CAthletic_Doc : public CDocument
{
protected: // serialization에서만 만들어집니다.
	CAthletic_Doc();
	DECLARE_DYNCREATE(CAthletic_Doc)

// 특성입니다.
public:

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// 구현입니다.
public:
	virtual ~CAthletic_Doc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// 검색 처리기에 대한 검색 콘텐츠를 설정하는 도우미 함수
	void SetSearchContent(const CString& value);
#endif // SHARED_HANDLERS
public:
	afx_msg void OnFileOpen();
	
	afx_msg void OnCpuVR();
	afx_msg void OnGpuVR();
	afx_msg void OnGpuVR_AO();

public:
	Cpu_VR m_render_cpu;
	Gpu_VR m_render_gpu;
};
