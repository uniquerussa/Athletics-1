
// Athletic_Doc.cpp : CAthletic_Doc 클래스의 구현
//

#include "stdafx.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "Athletic_.h"
#endif

#include "Athletic_Doc.h"
#include "Athletic_View.h"
#include "MainFrm.h"
#include "ChildFrm.h"
#include <propkey.h>

#include "vtkDICOMImageReader.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CAthletic_Doc

IMPLEMENT_DYNCREATE(CAthletic_Doc, CDocument)

BEGIN_MESSAGE_MAP(CAthletic_Doc, CDocument)
	ON_COMMAND(ID_FILE_OPEN, &CAthletic_Doc::OnFileOpen)
	ON_COMMAND(ID_CPU_VR, &CAthletic_Doc::OnCpuVR)
	ON_COMMAND(ID_GPU_VR, &CAthletic_Doc::OnGpuVR)
	ON_COMMAND(ID_GPU_AO, &CAthletic_Doc::OnGpuVR_AO)
END_MESSAGE_MAP()


// CAthletic_Doc 생성/소멸

CAthletic_Doc::CAthletic_Doc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.
	m_render_cpu= Cpu_VR();
	m_render_gpu= Gpu_VR();

}

CAthletic_Doc::~CAthletic_Doc()
{
}

BOOL CAthletic_Doc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CAthletic_Doc serialization

void CAthletic_Doc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}

#ifdef SHARED_HANDLERS

// 축소판 그림을 지원합니다.
void CAthletic_Doc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 문서의 데이터를 그리려면 이 코드를 수정하십시오.
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 검색 처리기를 지원합니다.
void CAthletic_Doc::InitializeSearchContent()
{
	CString strSearchContent;
	// 문서의 데이터에서 검색 콘텐츠를 설정합니다.
	// 콘텐츠 부분은 ";"로 구분되어야 합니다.

	// 예: strSearchContent = _T("point;rectangle;circle;ole object;");
	SetSearchContent(strSearchContent);
}

void CAthletic_Doc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = NULL;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != NULL)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CAthletic_Doc 진단

#ifdef _DEBUG
void CAthletic_Doc::AssertValid() const
{
	CDocument::AssertValid();
}

void CAthletic_Doc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CAthletic_Doc 명령

static int CALLBACK BrowseCallbackProc( HWND hWnd, UINT uMsg, LPARAM lParam,
										LPARAM lpData )
{
	switch( uMsg )
	{
	case BFFM_INITIALIZED:		// 폴더 선택 대화상자를 초기화 할 때, 초기 경로 설정
		{
			::SendMessage( hWnd, BFFM_SETSELECTION, TRUE, (LPARAM)lpData );
		}
		break;

	// BROWSEINFO 구조체의 ulFlags 값에 BIF_STATUSTEXT 가 설정된 경우 호출
	// 단, BIF_NEWDIALOGSTYLE 가 설정되어 있을 경우 호출되지 않음
	case BFFM_SELCHANGED:		// 사용자가 폴더를 선택할 경우 대화상자에 선택된 경로 표시
		{
			TCHAR szPath[ MAX_PATH ] = { 0, };

			::SHGetPathFromIDList( (LPCITEMIDLIST)lParam, szPath );
			::SendMessage( hWnd, BFFM_SETSTATUSTEXT, 0, (LPARAM)szPath );
		}
		break;

	// BROWSEINFO 구조체의 ulFlags 값에 BIF_VALIDATE 가 설정된 경우 호출
	// BIF_EDITBOX 와 같이 설정된 경우만 호출됨
	case BFFM_VALIDATEFAILED:	// 에디터 콘트롤에서 폴더 이름을 잘못 입력한 경우 호출
		{
			::MessageBox( hWnd, _T( "해당 폴더를 찾을 수 없습니다." ), _T( "오류" ),
				MB_ICONERROR | MB_OK );
		}
		break;
	}

	return 0;
}
void CAthletic_Doc::OnFileOpen()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	CWnd *pWnd = AfxGetMainWnd();
	HWND hWnd = pWnd->m_hWnd;

	BROWSEINFO bi;

	TCHAR szTemp[ MAX_PATH ] = { 0, };

	TCHAR * pszPath = _T( "C:\\" );
	
	::ZeroMemory( &bi, sizeof( BROWSEINFO ) );

	bi.hwndOwner	= hWnd;
	bi.lpszTitle	= _T( "파일 경로를 선택해주세요." );
	bi.ulFlags		= BIF_NEWDIALOGSTYLE | BIF_EDITBOX | BIF_RETURNONLYFSDIRS
						| BIF_STATUSTEXT | BIF_VALIDATE;
	bi.lpfn			= BrowseCallbackProc;
	bi.lParam		= (LPARAM)pszPath;

	LPITEMIDLIST pItemIdList = ::SHBrowseForFolder( &bi );

	if( !::SHGetPathFromIDList( pItemIdList, szTemp ) )
		return;

	vtkSmartPointer<vtkImageData> input = vtkSmartPointer<vtkImageData>::New();
	
	char charPath[MAX_PATH] = {0};
	WideCharToMultiByte(CP_ACP, 0, szTemp, MAX_PATH, charPath, MAX_PATH, NULL, NULL);

	printf("Run vtkDICOMImageReader.. \n"); 
	vtkSmartPointer<vtkDICOMImageReader> dicomReader = vtkSmartPointer<vtkDICOMImageReader>::New();
	dicomReader->SetDirectoryName(charPath);
	dicomReader->SetDataScalarTypeToShort();
	dicomReader->Update();
	input->DeepCopy(dicomReader->GetOutput());
	
	int dim[3];
	input->GetDimensions(dim);
	
	if(dim[0] < 2 || dim[1] < 2 || dim[2] < 2){
		return;
	}else{
		printf("load Volume data size (x,y,z) : %d, %d, %d\n", dim[0], dim[1], dim[2]);
	}

	short *h_volume = (short*)input->GetScalarPointer();

	double *range =input->GetScalarRange();
	printf("-min density : %.1f\n-max density : %.1f\n", range[0], range[1]);

	double *spacing = input->GetSpacing();
	printf("-voxel spacing : %f %f %f\n", spacing[0], spacing[1], spacing[2]);


	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
	CChildFrame *pChild = (CChildFrame *)pFrame->GetActiveFrame();
	CAthletic_View *pView = (CAthletic_View *)pChild->GetActiveView();

	pView->SetImageSize(dim[0]+200, dim[2]+200);
	float firstView[3] = {(float)dim[0]/2.f, (float)dim[1], (float)dim[2]/2.f};
	pView->SetViewPoint(firstView);
	firstView[1] /= 2.f;
	pView->SetVolumeCenter(firstView);

	pView->SetVolume(h_volume, dim, range, spacing);
	pView->SetTF(4096);

	printf("\n[Volume Load Complete]\n\n");
	
	MessageBox( hWnd, szTemp, _T("볼륨 로드 확인"), MB_OK );

}


void CAthletic_Doc::OnCpuVR()
{
	CWnd *pWnd = AfxGetMainWnd();
	HWND hWnd = pWnd->m_hWnd;

	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
	CChildFrame *pChild = (CChildFrame *)pFrame->GetActiveFrame();
	CAthletic_View *pView = (CAthletic_View *)pChild->GetActiveView();

	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if(pView->GetVolume()->GetDensityPointer() == NULL) {
		printf("no Volume \n");
		MessageBox( hWnd, _T("볼륨이 없습니다."), _T("경고"), MB_OK );
		return;
	}
	if(pView->GetTF()->GetTFData() == NULL) {
		MessageBox( hWnd, _T("변환함수가 없습니다."), _T("경고"), MB_OK );
		return;
	}
	
	//printf("%d %d \n", pView->GetImageSize()[0], pView->GetImageSize()[1]);

	uchar *image = m_render_cpu.VR_basic(pView->GetVolume(), pView->GetTF(), 
		pView->GetImageSize(), pView->GetViewingPoint());
	pView->SetBuffer(image);
	pView->SetRenderType(0);
	
	pView->Invalidate(TRUE);
}


void CAthletic_Doc::OnGpuVR()
{
	CWnd *pWnd = AfxGetMainWnd();
	HWND hWnd = pWnd->m_hWnd;

	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
	CChildFrame *pChild = (CChildFrame *)pFrame->GetActiveFrame();
	CAthletic_View *pView = (CAthletic_View *)pChild->GetActiveView();

	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if(pView->GetVolume()->GetDensityPointer() == NULL) {
		printf("no Volume \n");
		MessageBox( hWnd, _T("볼륨이 없습니다."), _T("경고"), MB_OK );
		return;
	}
	if(pView->GetTF()->GetTFData() == NULL) {
		MessageBox( hWnd, _T("변환함수가 없습니다."), _T("경고"), MB_OK );
		return;
	}
	
	//printf("%d %d \n", pView->GetImageSize()[0], pView->GetImageSize()[1]);

	uchar *image = m_render_gpu.VR_basic(pView->GetVolume(), pView->GetTF(), 
		pView->GetImageSize(), pView->GetViewingPoint());

	pView->SetBuffer(image);
	pView->SetRenderType(100);

	pView->Invalidate(TRUE);
}


void CAthletic_Doc::OnGpuVR_AO()
{
	CWnd *pWnd = AfxGetMainWnd();
	HWND hWnd = pWnd->m_hWnd;

	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
	CChildFrame *pChild = (CChildFrame *)pFrame->GetActiveFrame();
	CAthletic_View *pView = (CAthletic_View *)pChild->GetActiveView();

	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if(pView->GetVolume()->GetDensityPointer() == NULL) {
		printf("no Volume \n");
		MessageBox( hWnd, _T("볼륨이 없습니다."), _T("경고"), MB_OK );
		return;
	}
	if(pView->GetTF()->GetTFData() == NULL) {
		MessageBox( hWnd, _T("변환함수가 없습니다."), _T("경고"), MB_OK );
		return;
	}

	//printf("%d %d \n", pView->GetImageSize()[0], pView->GetImageSize()[1]);

	uchar *image = m_render_gpu.VR_AmbientOcclusion(pView->GetVolume(), pView->GetTF(), 
		pView->GetImageSize(), pView->GetViewingPoint());
	if(image == NULL){
		MessageBox( hWnd, _T("디버깅 시작 하세요."), _T("실패"), MB_OK );
		return;
	}

	pView->SetBuffer(image);
	pView->SetRenderType(101);

	pView->Invalidate(TRUE);
}
