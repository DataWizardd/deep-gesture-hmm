
// Spot.cpp : Defines the class behaviors for the application.
//

// #include "pch.h"
#include <SDKDDKVer.h>       // by hklee

#include "stdafx.h"
#include "Spot.h"
#include "CMyMIL.h"
#include "MyHand.h"

#include "MainFrm.h"
#include "ChildFrm.h"
#include "SpotDoc.h"
#include "SpotView.h"
#include "GestureDoc.h"
#include "GestureView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSpotApp

BEGIN_MESSAGE_MAP(CSpotApp, CWinApp)
	//{{AFX_MSG_MAP(CSpotApp)
	ON_COMMAND(ID_APP_ABOUT, OnAppAbout)
	ON_COMMAND(IDM_DUMP_FRAME, OnDumpFrame)
	ON_UPDATE_COMMAND_UI(IDM_DUMP_FRAME, OnUpdateDumpFrame)
	ON_COMMAND(ID_DISP_BY_FRAME, OnDispByFrame)
	ON_UPDATE_COMMAND_UI(ID_DISP_BY_FRAME, OnUpdateDispByFrame)
	ON_COMMAND(ID_DISP_CONTINUOUS, OnDispContinuous)
	ON_UPDATE_COMMAND_UI(ID_DISP_CONTINUOUS, OnUpdateDispContinuous)
	ON_COMMAND(ID_GET_I, OnGetI)
	ON_UPDATE_COMMAND_UI(ID_GET_I, OnUpdateGetI)
	ON_COMMAND(ID_GET_Y, OnGetY)
	ON_UPDATE_COMMAND_UI(ID_GET_Y, OnUpdateGetY)
	ON_COMMAND(ID_SHOW_GRAY, OnShowGray)
	ON_UPDATE_COMMAND_UI(ID_SHOW_GRAY, OnUpdateShowGray)
	ON_COMMAND(ID_AUTOSAVE, OnAutosave)
	ON_UPDATE_COMMAND_UI(ID_AUTOSAVE, OnUpdateAutosave)
	ON_COMMAND(IDM_TRACK_HAND, OnTrackHand)
	ON_UPDATE_COMMAND_UI(IDM_TRACK_HAND, OnUpdateTrackHand)
	ON_COMMAND(IDM_RECOG_HAND_GESTURE, OnRecogHandGesture)
	ON_UPDATE_COMMAND_UI(IDM_RECOG_HAND_GESTURE, OnUpdateRecogHandGesture)
	ON_COMMAND(IDM_SET_LASTDATA, OnSetLastData)
	ON_UPDATE_COMMAND_UI(IDM_SET_LASTDATA, OnUpdateSetLastData)
	ON_COMMAND(IDM_SET_FIRSTDATA, OnSetFirstData)
	ON_UPDATE_COMMAND_UI(IDM_SET_FIRSTDATA, OnUpdateSetFirstData)
	ON_COMMAND(IDM_SET_NEXTDATA, OnSetNextData)
	ON_UPDATE_COMMAND_UI(IDM_SET_NEXTDATA, OnUpdateSetNextData)
	ON_COMMAND(IDM_SET_PREVDATA, OnSetPrevData)
	ON_UPDATE_COMMAND_UI(IDM_SET_PREVDATA, OnUpdateSetPrevData)
	ON_COMMAND(IDM_SET_QUITDATA, OnSetQuitData)
	ON_UPDATE_COMMAND_UI(IDM_SET_QUITDATA, OnUpdateSetQuitData)
	ON_COMMAND(IDM_SET_TESTDATA, OnSetTestData)
	ON_UPDATE_COMMAND_UI(IDM_SET_TESTDATA, OnUpdateSetTestData)
	ON_COMMAND(IDM_SET_THRESHOLDDATA, OnSetThresholdData)
	ON_UPDATE_COMMAND_UI(IDM_SET_THRESHOLDDATA, OnUpdateSetThresholdData)
	ON_COMMAND(IDM_LABELLING, OnLabelling)
	ON_UPDATE_COMMAND_UI(IDM_LABELLING, OnUpdateLabelling)
	ON_COMMAND(IDM_SET_ALPHADATA, OnSetAlphaData)
	ON_UPDATE_COMMAND_UI(IDM_SET_ALPHADATA, OnUpdateSetAlphaData)
	ON_COMMAND(IDM_SET_BYEDATA, OnSetByeData)
	ON_UPDATE_COMMAND_UI(IDM_SET_BYEDATA, OnUpdateSetByeData)
	ON_COMMAND(IDM_SET_CIRCLEDATA, OnSetCircleData)
	ON_UPDATE_COMMAND_UI(IDM_SET_CIRCLEDATA, OnUpdateSetCircleData)
	ON_COMMAND(IDM_SET_SAWDATA, OnSetSawData)
	ON_UPDATE_COMMAND_UI(IDM_SET_SAWDATA, OnUpdateSetSawData)
	ON_COMMAND(IDM_SET_TRIANGLEDATA, OnSetTriangleData)
	ON_UPDATE_COMMAND_UI(IDM_SET_TRIANGLEDATA, OnUpdateSetTriangleData)
	ON_COMMAND(IDM_CHECK_FRAME_RATE, OnCheckFrameRate)
	ON_UPDATE_COMMAND_UI(IDM_CHECK_FRAME_RATE, OnUpdateCheckFrameRate)
	//}}AFX_MSG_MAP
	// Standard file based document commands
	ON_COMMAND(ID_FILE_NEW, CWinApp::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, CWinApp::OnFileOpen)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSpotApp construction

CSpotApp::CSpotApp()
{
	m_Flags.bContinuous = TRUE;
	m_Flags.bDumpMode = FALSE;
	m_Flags.bAutoSave = FALSE;
	m_Flags.bGetGray = FALSE;
	m_Flags.bGetY = FALSE;
	m_Flags.bTrack = TRUE;
	m_Flags.bRecognize = FALSE;
	m_Flags.bCheckSubstitution = TRUE;
	m_Flags.bCheckFrameRate = FALSE;

	m_nSaveCount = 0;
	m_pGestView = NULL;
	m_nMaxFrame = MAX_FRAME;
	m_nInputType = MODEL_TEST;

	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

/////////////////////////////////////////////////////////////////////////////
// The one and only CSpotApp object

CSpotApp theApp;
CMyMIL* theMIL = NULL;

/////////////////////////////////////////////////////////////////////////////
// CSpotApp initialization

BOOL CSpotApp::InitInstance()
{
	if (!AfxSocketInit())
	{
		AfxMessageBox(IDP_SOCKETS_INIT_FAILED);
		return FALSE;
	}

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	//  of your final executable, you should remove from the following
	//  the specific initialization routines you do not need.

#ifdef _AFXDLL
	Enable3dControls();			// Call this when using MFC in a shared DLL
#else
	Enable3dControlsStatic();	// Call this when linking to MFC statically
#endif

	LoadStdProfileSettings(8);  // Load standard INI file options (including MRU)

	// Register the application's document templates.  Document templates
	//  serve as the connection between documents, frame windows and views.

	CMultiDocTemplate* pDocTemplate;
	pDocTemplate = new CMultiDocTemplate(
		IDR_SPOTTYPE,
		RUNTIME_CLASS(CSpotDoc),
		RUNTIME_CLASS(CChildFrame), // custom MDI child frame
		RUNTIME_CLASS(CSpotView));
	AddDocTemplate(pDocTemplate);

	pDocTemplate = new CMultiDocTemplate(
		IDR_GESTTYPE,
		RUNTIME_CLASS(CGestureDoc),
		RUNTIME_CLASS(CChildFrame), // custom MDI child frame
		RUNTIME_CLASS(CGestureView));
	AddDocTemplate(pDocTemplate);

	// create main MDI Frame window
	CMainFrame* pMainFrame = new CMainFrame;
	if (!pMainFrame->LoadFrame(IDR_MAINFRAME))
		return FALSE;
	m_pMainWnd = pMainFrame;

	theMIL = new CMyMIL;

#ifdef COMMAND_LINE_PROCESSING
	// Parse command line for standard shell commands, DDE, file open
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);

	// Dispatch commands specified on the command line
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;
#endif

	m_pGestDoc = OpenDocumentFile("C:\\Users\\hyeon\\source\\repos\\Camera\\Main\\Camera.dat");
	POSITION pos = m_pGestDoc->GetFirstViewPosition();
	m_pGestView = m_pGestDoc->GetNextView(pos);

	// The main window has been initialized, so show and update it.
	pMainFrame->ShowWindow(m_nCmdShow);
	pMainFrame->UpdateWindow();

	return TRUE;
}

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

	// Dialog Data
		//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
		// No message handlers
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

// App command to run the dialog
void CSpotApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

/////////////////////////////////////////////////////////////////////////////
// CSpotApp commands

int CSpotApp::ExitInstance()
{
	// TODO: Add your specialized code here and/or call the base class
	if (theMIL)
		delete theMIL;

	return CWinApp::ExitInstance();
}

void CSpotApp::OnDumpFrame()
{
	// TODO: Add your command handler code here

	if (theMIL->m_bGrabStart)
		MessageBeep(MB_ICONHAND);
	else {
		m_Flags.bDumpMode = !m_Flags.bDumpMode;
		theMIL->m_myHand->SetDumpBuffer(m_Flags.bDumpMode);
	}
}

void CSpotApp::OnUpdateDumpFrame(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bDumpMode);
}

void CSpotApp::OnDispByFrame()
{
	// TODO: Add your command handler code here

	m_Flags.bContinuous = FALSE;
}

void CSpotApp::OnUpdateDispByFrame(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->Enable(m_Flags.bContinuous);
}

void CSpotApp::OnDispContinuous()
{
	// TODO: Add your command handler code here

	m_Flags.bContinuous = TRUE;
}

void CSpotApp::OnUpdateDispContinuous(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->Enable(!m_Flags.bContinuous);
}

void CSpotApp::OnGetI()
{
	// TODO: Add your command handler code here

	m_Flags.bGetY = FALSE;
}

void CSpotApp::OnUpdateGetI(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(!m_Flags.bGetY);
}

void CSpotApp::OnGetY()
{
	// TODO: Add your command handler code here

	m_Flags.bGetY = TRUE;
	m_Flags.bGetGray = TRUE;
}

void CSpotApp::OnUpdateGetY(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bGetY);
}

void CSpotApp::OnShowGray()
{
	// TODO: Add your command handler code here

	m_Flags.bGetGray = !m_Flags.bGetGray;
	if (m_Flags.bGetGray)
		m_Flags.bTrack = FALSE;
}

void CSpotApp::OnUpdateShowGray(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bGetGray);
}

void CSpotApp::OnAutosave()
{
	// TODO: Add your command handler code here
	m_Flags.bAutoSave = !m_Flags.bAutoSave;
}

void CSpotApp::OnUpdateAutosave(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bAutoSave);
}

void CSpotApp::OnTrackHand()
{
	// TODO: Add your command handler code here

	m_Flags.bTrack = !m_Flags.bTrack;
}

void CSpotApp::OnUpdateTrackHand(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bTrack);
}

void CSpotApp::SetRecogMode()
{
	m_Flags.bRecognize = TRUE;

	CGestureView* pView = (CGestureView*)m_pGestView;

	if (!pView->m_Flags.bImmediateRecog) {
		pView->m_Flags.bImmediateRecog = TRUE;
		pView->SetImmediateRecog();
	}
	if (m_Flags.bRecognize)
		m_Flags.bLabelling = FALSE;
}

void CSpotApp::OnRecogHandGesture()
{
	// TODO: Add your command handler code here

	m_Flags.bRecognize = !m_Flags.bRecognize;
	if (m_Flags.bRecognize)
		SetRecogMode();
}

void CSpotApp::OnUpdateRecogHandGesture(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bRecognize);
}

void CSpotApp::OnSetLastData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_LAST;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetLastData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_LAST);
}

void CSpotApp::OnSetFirstData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_FIRST;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetFirstData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_FIRST);
}

void CSpotApp::OnSetNextData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_NEXT;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetNextData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_NEXT);
}

void CSpotApp::OnSetPrevData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_PREV;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetPrevData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_PREV);
}

void CSpotApp::OnSetQuitData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_QUIT;
	m_nMaxFrame = MAX_QUIT_FRAME;
}

void CSpotApp::OnUpdateSetQuitData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_QUIT);
}

void CSpotApp::OnSetAlphaData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_ALPHA;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetAlphaData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_ALPHA);
}

void CSpotApp::OnSetByeData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_BYE;
	m_nMaxFrame = MAX_QUIT_FRAME;
}

void CSpotApp::OnUpdateSetByeData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_BYE);
}

void CSpotApp::OnSetCircleData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_CIRCLE;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetCircleData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_CIRCLE);
}

void CSpotApp::OnSetSawData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_SAW;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetSawData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_SAW);
}

void CSpotApp::OnSetTriangleData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_TRIANGLE;
	m_nMaxFrame = MAX_DATA_FRAME;
}

void CSpotApp::OnUpdateSetTriangleData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_TRIANGLE);
}

void CSpotApp::OnSetThresholdData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_THRESHOLD;
	m_nMaxFrame = MAX_THRESH_FRAME;
}

void CSpotApp::OnUpdateSetThresholdData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_THRESHOLD);
}

void CSpotApp::OnSetTestData()
{
	// TODO: Add your command handler code here

	m_nInputType = MODEL_TEST;
	m_nMaxFrame = MAX_FRAME;
}

void CSpotApp::OnUpdateSetTestData(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_nInputType == MODEL_TEST);
}

void CSpotApp::OnLabelling()
{
	// TODO: Add your command handler code here

	m_Flags.bLabelling = !m_Flags.bLabelling;
	if (m_Flags.bLabelling)
		m_Flags.bRecognize = FALSE;
}

void CSpotApp::OnUpdateLabelling(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bLabelling);
}

void CSpotApp::OnCheckFrameRate()
{
	// TODO: Add your command handler code here
	m_Flags.bCheckFrameRate = !m_Flags.bCheckFrameRate;
}

void CSpotApp::OnUpdateCheckFrameRate(CCmdUI* pCmdUI)
{
	// TODO: Add your command update UI handler code here

	pCmdUI->SetCheck(m_Flags.bCheckFrameRate);
}
