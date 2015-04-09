
// Athletic_View.cpp : CAthletic_View 클래스의 구현
//

#include "stdafx.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "Athletic_.h"
#endif
#include "MainFrm.h"
#include "ChildFrm.h"
#include "Athletic_Doc.h"
#include "Athletic_View.h"
#include "math.h"
#include <windows.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAthletic_View

IMPLEMENT_DYNCREATE(CAthletic_View, CView)

BEGIN_MESSAGE_MAP(CAthletic_View, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CAthletic_View::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_WM_SIZE()
	ON_WM_DESTROY()
	ON_WM_CREATE()
	ON_WM_CLOSE()
	ON_WM_KEYDOWN()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()

// CAthletic_View 생성/소멸

CAthletic_View::CAthletic_View()
{
	// TODO: 여기에 생성 코드를 추가합니다.

	FILE *infile;
	infile = fopen("data/Athletics.bmp", "rb");
	if(infile==NULL) {printf("No Image File"); return;}

	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hInfo;
	fread(&hf, sizeof(BITMAPFILEHEADER),1,infile);
	
	if(hf.bfType!=0x4D42) exit(1);
	fread(&hInfo,sizeof(BITMAPINFOHEADER),1,infile);
	if(hInfo.biBitCount!=24) {printf("Bad File Format!!"); return;}

	m_imageBuffer = new unsigned char[hInfo.biSizeImage];
	fread(m_imageBuffer, sizeof(unsigned char), hInfo.biSizeImage, infile);
	fclose(infile);

	m_ResultImgSize[0] = hInfo.biWidth;
	m_ResultImgSize[1] = hInfo.biHeight;

	m_Viewing[0] = 0.f;
	m_Viewing[1] = 1.f;
	m_Viewing[2] = 0.f;

	m_Volcenter[0] = 0.f;
	m_Volcenter[1] = 0.f;
	m_Volcenter[2] = 0.f;

	m_vol = Volume();
	m_TF = TFManager();
	m_CurrentRenderType=-1;
	//0 CPU - 1 ...
	//100 GPU - 101 GPU AO - 102...
}

CAthletic_View::~CAthletic_View()
{
	delete[] m_imageBuffer;
}

BOOL CAthletic_View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// CAthletic_View 그리기

void CAthletic_View::OnDraw(CDC* /*pDC*/)
{
	CAthletic_Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
	wglMakeCurrent(m_hDC, m_hRC);

	GLRenderScene();
	SwapBuffers(m_hDC);

	wglMakeCurrent(m_hDC, NULL);
}


// CAthletic_View 인쇄


void CAthletic_View::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CAthletic_View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void CAthletic_View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CAthletic_View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}

void CAthletic_View::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CAthletic_View::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CAthletic_View 진단

#ifdef _DEBUG
void CAthletic_View::AssertValid() const
{
	CView::AssertValid();
}

void CAthletic_View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CAthletic_Doc* CAthletic_View::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CAthletic_Doc)));
	return (CAthletic_Doc*)m_pDocument;
}
#endif //_DEBUG


// CAthletic_View 메시지 처리기


void CAthletic_View::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	VERIFY(wglMakeCurrent(m_hDC, m_hRC));

	GLResize(cx, cy);

	VERIFY(wglMakeCurrent(NULL, NULL));
}

void CAthletic_View::GLResize(int cx, int cy)
{
	glViewport (0, 0, (GLsizei) cx, (GLsizei) cy);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluPerspective( 50.0f, (GLdouble) cx/cy, 1.0f, 150.0f );
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt (0.0, 0.0, 5.0 , 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void CAthletic_View::GLinit(void)
{    
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClearDepth(1.0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);

	glGenTextures(1, &m_texName);
	glBindTexture (GL_TEXTURE_2D, m_texName);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, 3, m_ResultImgSize[0], m_ResultImgSize[1], 
				 0, GL_RGB, GL_UNSIGNED_BYTE, m_imageBuffer);

	glEnable(GL_TEXTURE_2D);
	glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	
}

void CAthletic_View::OnDestroy()
{
	wglDeleteContext(m_hRC);
	::ReleaseDC(m_hWnd, m_hDC);
	
	CView::OnDestroy();
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}


int CAthletic_View::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  여기에 특수화된 작성 코드를 추가합니다.
	int nPixelFormat;
	m_hDC = ::GetDC(m_hWnd);

	static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW|
		PFD_SUPPORT_OPENGL|
		PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,
		24,
		0,0,0,0,0,0,
		0,0,
		0,0,0,0,0,
		32,
		0,
		0,
		PFD_MAIN_PLANE,
		0,
		0,0,0
	};

	nPixelFormat = ChoosePixelFormat(m_hDC, &pfd);
	VERIFY(SetPixelFormat(m_hDC,nPixelFormat, &pfd));
	m_hRC = wglCreateContext(m_hDC);
	VERIFY(wglMakeCurrent(m_hDC,m_hRC));
	wglMakeCurrent(NULL,NULL);

	return 0;
}
void CAthletic_View::GLRenderScene(void)
{
	GLinit();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	glBindTexture(GL_TEXTURE_2D, m_texName);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(-2.8, -2.8);
    
	glTexCoord2f(1.0, 0.0);
	glVertex2f(2.8, -2.8);

	glTexCoord2f(1.0, 1.0);
	glVertex2f(2.8, 2.8);

	glTexCoord2f(0.0, 1.0);
	glVertex2f(-2.8, 2.8);

	glEnd();
	glFlush();
	glDisable(GL_TEXTURE_2D);
	
}


void CAthletic_View::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
	switch(nChar)
	{
	case VK_UP:		//상
		//printf("상\n");
		MoveUp();
		break;

	case VK_DOWN:	//하
		//printf("하\n");
		MoveDown();
		break;

	case VK_RIGHT:	//우
		//printf("우\n");
		MoveRight();
		break;

	case VK_LEFT:	//좌
		//printf("좌\n");
		MoveLeft();
		break;
	}
	CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
	CAthletic_Doc *pDoc = (CAthletic_Doc *)pFrame->GetActiveDocument();

	uchar *image = NULL;

	//printf("%d %d\n", m_ResultImgSize[0], m_ResultImgSize[1]);
	if(m_CurrentRenderType == 0)
		image = pDoc->m_render_cpu.VR_basic(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);
	else if(m_CurrentRenderType == 100)
		image = pDoc->m_render_gpu.VR_basic(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);
	else if(m_CurrentRenderType == 101)
		image = pDoc->m_render_gpu.VR_AmbientOcclusion(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);

	SetBuffer(image);
	Invalidate(TRUE);
}

const float r = 200.0f;
const float PI = 3.1415926536f;
const double degree = PI/180.0;
float fMove_x=90.f, fMove_z=0.f;

void CAthletic_View::MoveUp(void)
{
	//m_Viewing[2] += 10.f;
	fMove_z += 10.f;
	m_Viewing[1] = (float)(r*cos(degree*fMove_z) + m_Volcenter[1]);
	m_Viewing[2] = (float)(r*sin(degree*fMove_z) + m_Volcenter[2]);
}
	
void CAthletic_View::MoveDown(void)
{
	//m_Viewing[2] -= 10.f;
	fMove_z -= 10.f;
	m_Viewing[1] = (float)(r*cos(degree*fMove_z) + m_Volcenter[1]);
	m_Viewing[2] = (float)(r*sin(degree*fMove_z) + m_Volcenter[2]);
}

void CAthletic_View::MoveRight(void)
{
	//m_Viewing[0] += 10.f;
	fMove_x -= 10.f;
	m_Viewing[0] = (float)(r*cos(degree*fMove_x) + m_Volcenter[0]);
	m_Viewing[1] = (float)(r*sin(degree*fMove_x) + m_Volcenter[1]);
	printf("%f, %f\n",m_Viewing[0],m_Viewing[1]);
}

void CAthletic_View::MoveLeft(void)
{
	//m_Viewing[0] -= 10.f;
	fMove_x += 10.f;
	m_Viewing[0] = (float)(r*cos(degree*fMove_x) + m_Volcenter[0]);
	m_Viewing[1] = (float)(r*sin(degree*fMove_x) + m_Volcenter[1]);
}

int ox, oy;
int buttonstate=0;
void CAthletic_View::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	ox = point.x;
	oy = point.y;
	buttonstate=1;
	CView::OnLButtonDown(nFlags, point);
}


void CAthletic_View::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	buttonstate=0;

	CView::OnLButtonUp(nFlags, point);
}


void CAthletic_View::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	float dx,dy;
	
	dx = (float)(point.x-ox);
	dy = (float)(point.y-oy);
	if(buttonstate==1){
		degreeX += dy/1000.f;
		degreeY += dx/1000.f;
		
		
		CAthletic_View::Rotate(degreeX ,degreeY);
		printf("degree %f, %f\n",degreeX,degreeY);
		Rotate(degreeX,degreeY);
		
		CMainFrame *pFrame = (CMainFrame *)AfxGetMainWnd();
		CAthletic_Doc *pDoc = (CAthletic_Doc *)pFrame->GetActiveDocument();

		uchar *image = NULL;
		//printf("%f %f %f\n",m_Viewing[0],m_Viewing[1],m_Viewing[2]);
		//printf("%d %d\n", m_ResultImgSize[0], m_ResultImgSize[1]);
		if(m_CurrentRenderType == 0)
			image = pDoc->m_render_cpu.VR_basic(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);
		else if(m_CurrentRenderType == 100)
			image = pDoc->m_render_gpu.VR_basic(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);
		else if(m_CurrentRenderType == 101)
			image = pDoc->m_render_gpu.VR_AmbientOcclusion(&m_vol, &m_TF, m_ResultImgSize, m_Viewing);

		SetBuffer(image);
		Invalidate(TRUE);
	}
	ox=point.x;
	oy=point.y;
	
	CView::OnMouseMove(nFlags, point);
}
void CAthletic_View::Rotate(float x_, float y_)
{
	float angle;
	float u,v,w;
	printf("degree %f, %f\n",x_,y_);
	
	inputMatrix[0][0] =  m_Viewing[0];
	inputMatrix[1][0] =  m_Viewing[1];
	inputMatrix[2][0] =  m_Viewing[2];
	inputMatrix[3][0] = 1.0f; 
	printf("시작->%f %f %f\n",inputMatrix[0][0],inputMatrix[1][0],inputMatrix[2][0]);	
	setUpRotationMatrix(x_, 0, 0, 1);
	multiplyMatrix();

	inputMatrix[0][0] = outputMatrix[0][0];
	inputMatrix[1][0] = outputMatrix[1][0];
	inputMatrix[2][0] = outputMatrix[2][0];
	inputMatrix[3][0] = 1.0; 
	//printf("두번째->%f %f %f\n",inputMatrix[0][0],inputMatrix[1][0],inputMatrix[2][0]);		
	setUpRotationMatrix(y_, 0, 1, 0);
	multiplyMatrix();

	m_Viewing[0]=outputMatrix[0][0];
	m_Viewing[1]=outputMatrix[1][0];
	m_Viewing[2]=outputMatrix[2][0];

	//printf("---%f %f %f\n",m_Viewing[0],m_Viewing[1],m_Viewing[2]);
	//printf("--------------------\n");
	
	
}

void CAthletic_View::multiplyMatrix(void)
{
	 for(int i = 0; i < 4; i++ ){
        for(int j = 0; j < 1; j++){
            outputMatrix[i][j] = 0;
            for(int k = 0; k < 4; k++){
               outputMatrix[i][j] += rotationMatrix[i][k] * inputMatrix[k][j];
            }
        }
    }
	printf("밑에와 같아야 %f %f %f\n",inputMatrix[0][0],inputMatrix[1][0],inputMatrix[2][0]);	
}


void CAthletic_View::setUpRotationMatrix(float angle, float u, float v, float w)
{
	 float L = (u*u + v * v + w * w);
    angle = angle * 3.14 / 180.0; //converting to radian value
    float u2 = u * u;
    float v2 = v * v;
    float w2 = w * w; 
 
    rotationMatrix[0][0] = (u2 + (v2 + w2) * cos(angle)) / L;
    rotationMatrix[0][1] = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
    rotationMatrix[0][2] = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;
    rotationMatrix[0][3] = 0.0; 
 
    rotationMatrix[1][0] = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
    rotationMatrix[1][1] = (v2 + (u2 + w2) * cos(angle)) / L;
    rotationMatrix[1][2] = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;
    rotationMatrix[1][3] = 0.0; 
 
    rotationMatrix[2][0] = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
    rotationMatrix[2][1] = (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
    rotationMatrix[2][2] = (w2 + (u2 + v2) * cos(angle)) / L;
    rotationMatrix[2][3] = 0.0; 
 
    rotationMatrix[3][0] = 0.0;
    rotationMatrix[3][1] = 0.0;
    rotationMatrix[3][2] = 0.0;
    rotationMatrix[3][3] = 1.0;
}
