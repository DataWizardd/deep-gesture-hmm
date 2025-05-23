// Spot.h : main header file for the SPOT application
//

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"       // main symbols
// #include "mil.h"         // by hklee
// #include "mwinmil.h"     // by hklee
// #include "milsetup.h"    // by hklee
// #include "milerr.h"      // by hklee

#pragma warning(disable:4267)
#pragma warning(disable:4473)
#pragma warning(disable:4477)
#pragma warning(disable:4996)

typedef double				REAL;
typedef short				SHORT;
typedef unsigned short		USHORT;

typedef long            MIL_ID;   // by hklee


#define MORE_MODEL          1    // by hklee

#define MAX_BUFFER			128
#define MAGIC_SIZE			64

#define MAX_MODEL			6
#define	MAX_OBSERVATION		200
#define MIN_POINTS			5

#define TEST_LEVEL			16
#define TEST_DIMENSION		2
#define	TEST_STATE			10

#define WIDTH				320
#define HEIGHT				240

#define MAX_FRAME_INPUT		64
#define MAX_RESAMPLE_DATA	(MAX_OBSERVATION - 30)

#define	MAX_FRAME			200
#define	MAX_DATA_FRAME		32
#define	MAX_QUIT_FRAME		48
#define	MAX_THRESH_FRAME	64

#define MODEL_LAST			0
#define MODEL_FIRST			1
#define MODEL_NEXT			2
#define MODEL_PREV			3
#define MODEL_QUIT			4

#ifdef MORE_MODEL
#	define MODEL_BYE			5	// new
#	define MODEL_CIRCLE			6	// new
#	define MODEL_TRIANGLE		7
#	define MODEL_SAW			8
#	define MODEL_ALPHA			9
#	define MODEL_THRESHOLD		10
#else
#	define MODEL_THRESHOLD		5
#endif

#define MODEL_TEST			(MODEL_THRESHOLD+1)

#define MAX_INPUTTYPE		(MODEL_TEST+1)

#define MAX_MYLABEL		30

#define MAX_GESTURE		20


typedef struct tagMYLABEL {
	SHORT	nModel;
	SHORT	nStart;
	SHORT	nLast;
} MYLABEL;

typedef struct tagMODEL {
	SHORT	nState;
	SHORT	nNumFile;
	SHORT	nMinTime;
	char	szName[10];
} MODEL;

typedef struct tagRECOUT {
	SHORT	nStartT;
	SHORT	nFirstEndT;
	SHORT	nEndT;
	SHORT	nModel;
} RECOUT;

/////////////////////////////////////////////////////////////////////////////
// CSpotApp:
// See Spot.cpp for the implementation of this class
//

class CSpotApp : public CWinApp
{
public:
	CSpotApp();
	void	SetRecogMode();

	struct {
		BOOL	bContinuous: 2;
		BOOL	bDumpMode: 2;
		BOOL	bAutoSave: 2;
		BOOL	bGetGray: 2;
		BOOL	bGetY: 2;
		BOOL	bTrack: 2;
		BOOL	bRecognize: 2;
		BOOL	bLabelling: 2;
		BOOL	bCheckSubstitution: 2;
		BOOL	bCheckFrameRate: 2;
	} m_Flags;
	SHORT	m_nInputType;
	SHORT	m_nMaxFrame;
	SHORT	m_nSaveCount;
	CView*	m_pGestView;
	CDocument* m_pGestDoc;

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CSpotApp)
	public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(CSpotApp)
	afx_msg void OnAppAbout();
	afx_msg void OnDumpFrame();
	afx_msg void OnUpdateDumpFrame(CCmdUI* pCmdUI);
	afx_msg void OnDispByFrame();
	afx_msg void OnUpdateDispByFrame(CCmdUI* pCmdUI);
	afx_msg void OnDispContinuous();
	afx_msg void OnUpdateDispContinuous(CCmdUI* pCmdUI);
	afx_msg void OnGetI();
	afx_msg void OnUpdateGetI(CCmdUI* pCmdUI);
	afx_msg void OnGetY();
	afx_msg void OnUpdateGetY(CCmdUI* pCmdUI);
	afx_msg void OnShowGray();
	afx_msg void OnUpdateShowGray(CCmdUI* pCmdUI);
	afx_msg void OnAutosave();
	afx_msg void OnUpdateAutosave(CCmdUI* pCmdUI);
	afx_msg void OnTrackHand();
	afx_msg void OnUpdateTrackHand(CCmdUI* pCmdUI);
	afx_msg void OnRecogHandGesture();
	afx_msg void OnUpdateRecogHandGesture(CCmdUI* pCmdUI);
	afx_msg void OnSetLastData();
	afx_msg void OnUpdateSetLastData(CCmdUI* pCmdUI);
	afx_msg void OnSetFirstData();
	afx_msg void OnUpdateSetFirstData(CCmdUI* pCmdUI);
	afx_msg void OnSetNextData();
	afx_msg void OnUpdateSetNextData(CCmdUI* pCmdUI);
	afx_msg void OnSetPrevData();
	afx_msg void OnUpdateSetPrevData(CCmdUI* pCmdUI);
	afx_msg void OnSetQuitData();
	afx_msg void OnUpdateSetQuitData(CCmdUI* pCmdUI);
	afx_msg void OnSetTestData();
	afx_msg void OnUpdateSetTestData(CCmdUI* pCmdUI);
	afx_msg void OnSetThresholdData();
	afx_msg void OnUpdateSetThresholdData(CCmdUI* pCmdUI);
	afx_msg void OnLabelling();
	afx_msg void OnUpdateLabelling(CCmdUI* pCmdUI);
	afx_msg void OnSetAlphaData();
	afx_msg void OnUpdateSetAlphaData(CCmdUI* pCmdUI);
	afx_msg void OnSetByeData();
	afx_msg void OnUpdateSetByeData(CCmdUI* pCmdUI);
	afx_msg void OnSetCircleData();
	afx_msg void OnUpdateSetCircleData(CCmdUI* pCmdUI);
	afx_msg void OnSetSawData();
	afx_msg void OnUpdateSetSawData(CCmdUI* pCmdUI);
	afx_msg void OnSetTriangleData();
	afx_msg void OnUpdateSetTriangleData(CCmdUI* pCmdUI);
	afx_msg void OnCheckFrameRate();
	afx_msg void OnUpdateCheckFrameRate(CCmdUI* pCmdUI);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

extern CSpotApp theApp;
extern CStatusBar* theStatus;
