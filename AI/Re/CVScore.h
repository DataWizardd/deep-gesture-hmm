#ifndef	_CVSCORE_H
#define	_CVSCORE_H

#define NORMAL_MODEL		0
#define THRESHOLD_MODEL		1
#define UNITED_MODEL		2

// United transition type
#define UNITED_FORWARD		0
#define UNITED_THRESHOLD	1
#define UNITED_NORMAL		2

// Class Viterbi score
class CVScore {
public:
	CVScore();
	CVScore( CView* pView, SHORT nTime, SHORT nState );
	~CVScore();
	BOOL	New( CView* pView, SHORT nTime, SHORT nState );
	SHORT	GetBestThresholdState( SHORT nModel, REAL* prLL );
	SHORT	GetBestUnited( REAL* prLL );
	void	DisplayMatchPath( const char* pszModelName, BOOL bBest );
	void	DisplayUnitedMatchPath( MODEL* pModel );
	BOOL	SetSubViterbiScore( class CDHmm* pHmm, MODEL* pModel, 
							SHORT nModel, SHORT nThresholdModel );
	REAL	GetViterbiScore( SHORT nCurTime, BOOL bFixedFinalState );
	void	BacktrackViterbi( SHORT nCurTime, SHORT nFinal=-1, 
					SHORT nKindOfModel=NORMAL_MODEL );

	BOOL	ShowRecognitionPath( SHORT nCurModel, class CStroke* pStroke );
	void	SetWidthHeight();
	SHORT	GetSubState( SHORT nCurState, SHORT* pnModel );
	POINT	GetStateLocation( CRect* prc, SHORT nCurState );
	void	DrawStateDiagram( CDC* pDC, CRect* prc, 
						SHORT nCurModel, BOOL bUnited);
	void	DrawInitialDiagram( CMetaFileDC* pMetaDC, 
								CRect* prc, SHORT nCurModel,
								COLORREF rgbEnable, BOOL bUnited );
	void	HighlightTransition( CMetaFileDC* pMetaDC, 
								 CVScore* pVScore, SHORT nT, 
								 CRect* prc, COLORREF rgbEnable );
	void	DrawUnitedTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
								  SHORT nModel, SHORT nType );
	BOOL	IsUnitedStartState( SHORT nState );
	BOOL	IsUnitedFinalState( SHORT nState );
	void	DrawUnitedInitTransition( CDC* pDC, POINT ptStart, POINT ptEnd, 
							SHORT nT, SHORT nState, BOOL bForward );
	void	HighlightUnitedTransition( CMetaFileDC* pMetaDC, SHORT nT, 
							CRect* prc, COLORREF rgbEnable );
	BOOL	DrawHmmNetworks( CMetaFileDC* pMetaDC, BOOL bImmediate, REAL rWait,
						  CVScore* pVScore, MODEL* pModel, 
						  SHORT nNumModel, CStroke* pStroke );
	void	DrawOriginalHmmNetworks( CMetaFileDC* pMetaDC );
	void	DrawCurCandidate( CMetaFileDC* pMetaDC, 
					SHORT nCurTime, CRect* prc, COLORREF rgbEnable );
	BOOL	DrawUnitedModel( CMetaFileDC* pMetaDC, BOOL bImmediate,	REAL rWait,
							MODEL* pModel, class CStroke* pStroke );
	BOOL	ShowCurCandidate( CRect* prc, SHORT nT, COLORREF rgbEnable );
	BOOL	ShowUnitedProcess( BOOL bImmediate, REAL rWait, MODEL* pModel, 
							class CStroke* pStroke );

protected:
	void	InitVariables();
	void	DrawState( CDC* pDC, POINT pt, SHORT nID, BOOL bSelfLoop );
	void	DrawMyState( CDC* pDC, POINT pt, SHORT nID, 
						COLORREF rgbBrush );
	void	DrawSelfTransition( CDC* pDC, POINT pt );
	void	DrawTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
						SHORT nKindOfModel );
	void	DrawThresholdTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
					    SHORT nFromState, SHORT nToState );
	void	DrawSkipTransition( CDC* pDC, POINT ptFrom, POINT ptTo );
	void	DrawViterbiScore( CDC* pDC, POINT ptLoc, REAL rScore, 
						BOOL bEraseBk );
	void	DrawUnitedThreshold( CDC* pDC, SHORT nT, CRect* prc, 
					SHORT nCurModel, SHORT nState, BOOL bForward );
	void	ModelToUnitedStart( CDC* pDC, SHORT nT, CRect* prc );
	void	UnitedStartToModel( CDC* pDC, SHORT nT, CRect* prc );
	void	HighlightUnitedInternal( CMetaFileDC* pMetaDC, SHORT nT, 
							CRect* prc, COLORREF rgbEnable );

public:
	BOOL	m_bSingleTransition;
	REAL	m_rLikelihood;
	REAL	m_rPenalty;
	REAL**	m_pprScore;
	SHORT**	m_ppnState;
	SHORT*	m_pnPath;
	SHORT	m_nNumTime;
	SHORT	m_nNumState;
	SHORT	m_nFinalState;
	SHORT	m_nTopY;
	SHORT	m_nWidth;
	SHORT	m_nHeight;
	SHORT	m_nKindOfModel;
	SHORT	m_nStartState;

	// For united model processing
	SHORT	m_nSubModel;		// Number of sub models
	SHORT*	m_pnSubFinal;		// Final states of sub models
	class	CVScore* m_pSubVScore; // VScore class for sub model drawing
	CView*	m_pView;

protected:
	;
};


BOOL	DrawAllModel( CMetaFileDC* pMetaDC, BOOL bImmediate, REAL rWait,
				CVScore* pVScore, MODEL* pModel, 
				SHORT nNumModel, class CStroke* pStroke );
BOOL	ShowMatchProcess( BOOL bImmediate, REAL rWait, CVScore* pVScore, 
			MODEL* pModel, SHORT nNumModel, class CStroke* pStroke );
void	CalcAllModel( CVScore* pVScore, BOOL bUnited,
			   SHORT nNumModel, CStroke* pStroke, CRect* prc );

#endif
