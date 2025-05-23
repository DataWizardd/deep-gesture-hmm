// #include "pch.h"
#include <SDKDDKVer.h>       // by hklee

#include "stdafx.h"
#include "Spot.h"
#include "Cstroke.h"

#include "utils.h"
#include <math.h>
#include "Cdhmm.h"
#include "CVScore.h"

void CVScore::InitVariables()
{
	m_rLikelihood = 0.0;
	m_rPenalty = 0.0;
	m_nNumTime  = 0;
	m_nNumState = 0;
	m_nFinalState = 0;
	m_nWidth = 0;
	m_nHeight = 0;
	m_nTopY = 0;
	m_bSingleTransition = FALSE;
	m_nKindOfModel = FALSE;
	m_pprScore = NULL;
	m_ppnState = NULL;
	m_pnPath = NULL;
	
	// For sub models
	m_nStartState = 0;
	m_nSubModel  = 0;
	m_pnSubFinal = NULL;
	m_pSubVScore = NULL;

	m_pView = NULL;
}

CVScore::CVScore()
{
	InitVariables();
}

CVScore::CVScore( CView* pView, SHORT nTime, SHORT nState )
{
	InitVariables();
	New( pView, nTime, nState );
}

CVScore::~CVScore()
{
	int i;

	if (m_pprScore) {
		for(i=0; i<m_nNumTime; i++) {
			if ( m_pprScore[i] )
				delete[] m_pprScore[i];
		}
		delete[] m_pprScore;
	}

	if ( m_ppnState ) {
		for(i=0; i<m_nNumTime; i++) {
			if ( m_ppnState[i] )
				delete[] m_ppnState[i];
		}
		delete[] m_ppnState;
	}

	if ( m_pnPath )
		delete[] m_pnPath;

	if ( m_pnSubFinal )
		delete[] m_pnSubFinal;

	if ( m_pSubVScore )
		delete[] m_pSubVScore;
}

BOOL CVScore::New( CView* pView, SHORT nTime, SHORT nState )
{
	m_pView = pView;
	m_nNumTime  = nTime;
	m_nNumState = nState;

	m_pprScore = new REAL *[nTime];
	m_ppnState = new SHORT*[nTime];
	m_pnPath   = new SHORT [nTime];
	if ( !m_pprScore || !m_ppnState || !m_pnPath ) {
		if ( m_pprScore ) delete[] m_pprScore;
		if ( m_ppnState ) delete[] m_ppnState;
		if ( m_pnPath )   delete[] m_pnPath;
		return FALSE;
	}

	memset( m_pprScore, 0, sizeof(REAL *) * nTime );
	memset( m_ppnState, 0, sizeof(SHORT*) * nTime );

	for(int i=0; i<nTime; i++) {
		m_pprScore[i] = new REAL [nState];
		m_ppnState[i] = new SHORT[nState];
		if ( !m_pprScore[i] || !m_ppnState[i] ) 
			return FALSE;
	}

	return TRUE;
}

BOOL CVScore::SetSubViterbiScore( CDHmm* pHmm, MODEL* pModel,
								 SHORT nModel, SHORT nThresholdModel )
{
	m_nSubModel = nModel;
	m_pnSubFinal = new SHORT[m_nSubModel];
	m_pSubVScore = new CVScore[nModel];

	SHORT iState = 0;

	for(int i=0; i<nModel; i++) {
		m_pSubVScore[i].m_pView = m_pView;
		m_pSubVScore[i].m_nNumTime  = 1;
		m_pSubVScore[i].m_nNumState = pHmm[i].m_nState;
		m_pSubVScore[i].m_nStartState  = iState;
		m_pSubVScore[i].m_nFinalState  = iState + pHmm[i].m_nState - 1;
		m_pSubVScore[i].m_bSingleTransition = m_bSingleTransition;
		if ( i == nThresholdModel ) {
			m_pSubVScore[i].m_nKindOfModel = THRESHOLD_MODEL;
			m_pnSubFinal[i] = m_pSubVScore[i].m_nStartState;
		} else {
			m_pSubVScore[i].m_nKindOfModel = NORMAL_MODEL;
			m_pnSubFinal[i] = m_pSubVScore[i].m_nFinalState;
		}

		iState += pHmm[i].m_nState;
	}

	return TRUE;
}

SHORT CVScore::GetBestThresholdState( SHORT nModel, REAL* prLL )
{
	SHORT nBest = -1;

	if (m_pSubVScore[nModel].m_nKindOfModel != THRESHOLD_MODEL)
		return nBest;

	for( int i=m_pSubVScore[nModel].m_nStartState; 
			 i<=m_pSubVScore[nModel].m_nFinalState; i++) {
		if( *prLL > m_pprScore[m_nNumTime-1][i] ) {
			*prLL = m_pprScore[m_nNumTime-1][i];
			nBest = i;
		}
	}

	return nBest;
}


//#define SINGLE_MODEL_TEST

SHORT CVScore::GetBestUnited( REAL* prLL )
{
	SHORT	nBest = -1;
	SHORT	nFinal;

	*prLL = cMaxREAL;

#ifdef SINGLE_MODEL_TEST
	SHORT nSoloModel = MODEL_FIRST;

	theApp.m_Flags.bCheckSubstitution = FALSE;
	nFinal = m_pSubVScore[nSoloModel].m_nFinalState;
	*prLL = m_pprScore[m_nNumTime-1][nFinal];
	nBest = nSoloModel;

	SHORT nTmp = GetBestThresholdState( MODEL_THRESHOLD, prLL );
	if (nTmp >= 0)
		nBest = MODEL_THRESHOLD;
#else
	for( int i=0; i<m_nSubModel; i++ ) {
		if (m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
			SHORT nTmp = GetBestThresholdState( i, prLL );
			if (nTmp >= 0)
				nBest = i;
		} else {
			nFinal = m_pSubVScore[i].m_nFinalState;
			if( *prLL > m_pprScore[m_nNumTime-1][nFinal] ) {
				*prLL = m_pprScore[m_nNumTime-1][nFinal];
				nBest = i;
			}
		}
	}
#endif

	return nBest;
}

REAL CVScore::GetViterbiScore( SHORT nCurTime, BOOL bFixedFinalState )
{
	if ( bFixedFinalState && nCurTime == (m_nNumTime - 1) )
		return m_rLikelihood;
	else
		return m_pprScore[ nCurTime ][ m_pnPath[nCurTime] ];
}
  
void CVScore::BacktrackViterbi( SHORT nCurTime, SHORT nFinal, 
							   SHORT nKindOfModel )
{
	int		i, t;
	SHORT	iState;
	REAL	mscore = cMaxREAL;

	if (nCurTime < 0)
		t = m_nNumTime-1;
	else
		t = nCurTime;

	iState = nFinal;
	if ( nFinal < 0 ) {
		if (m_nKindOfModel == UNITED_MODEL) {
			for( i=0; i<m_nSubModel; i++ ) {
				if (m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
					SHORT nTmp = GetBestThresholdState(i, &mscore);
					if (nTmp >= 0)
						iState = nTmp;
				} else {
					nFinal = m_pSubVScore[i].m_nFinalState;
					if( mscore > m_pprScore[m_nNumTime-1][nFinal] ) {
						mscore = m_pprScore[m_nNumTime-1][nFinal];
						iState = nFinal;
					}
				}
			}
		} else {
			if ( nCurTime < 0 )
				iState = m_nNumState - 1;
			else {
				for( i=0; i<m_nNumState; i++ ) {
					if( mscore > m_pprScore[t][i] ) {
						mscore = m_pprScore[t][i];
						iState = i;
					}
				}
			}
		}
	} else if ( nKindOfModel == THRESHOLD_MODEL ) {
		for( i=nFinal; i<m_nNumState; i++ ) {
			if( mscore > m_pprScore[t][i] ) {
				mscore = m_pprScore[t][i];
				iState = i;
			}
		}
	}

	m_pnPath[t] = iState;
	for( ; t > 0 ; t -- )
		m_pnPath[t-1] = m_ppnState[t][m_pnPath[t]];
}

void CVScore::DisplayMatchPath( const char* pszModelName, BOOL bBest )
{
	if (bBest)
		PutProgressString("*%s: ", pszModelName);
	else
		PutProgressString(" %s: ", pszModelName);

	for(int t=0; t<m_nNumTime; t++) {
		PutProgressString( " %2d(%.2f)", m_pnPath[t], 
				 m_pprScore[t][m_pnPath[t]] );
	}
	PrintProgressString("");
}

void CVScore::DisplayUnitedMatchPath( MODEL* pModel )
{
	REAL	mscore;
	SHORT	nBest = GetBestUnited( &mscore );

	for(int i=0; i<m_nSubModel; i++) {
		BacktrackViterbi( m_nNumTime-1, m_pnSubFinal[i], 
					m_pSubVScore[i].m_nKindOfModel );
		DisplayMatchPath( pModel[i].szName, i == nBest );
	}

	BacktrackViterbi( m_nNumTime-1, -1 );
}


#define RADIUS				8	//12
#define X_GAP				70
#define Y_GAP				30	//50
#define Y_TOP				(Y_GAP*10)	//(Y_GAP*5)
#define META_Y_TOP			(Y_GAP + RADIUS)
#define MODEL_GAP			10	//30
#define LABEL_GAP			10	//20
#define SELF_OFFSET			(RADIUS+3)
#define THRESHOLD_GAP		30	//40   // Reduced
//#define THRESHOLD_GAP		20	// Original
#define START_X_POS			(THRESHOLD_GAP*5)	//(THRESHOLD_GAP*2 + THRESHOLD_GAP/2)
#define UNITED_START_X_POS	MODEL_GAP
#define PEN_WIDTH			1

#ifdef OLD_VARIABLE
	static REAL		rPIXEL;
	static SHORT	RADIUS, X_GAP, Y_GAP, Y_TOP, META_Y_TOP;
	static SHORT	MODEL_GAP, LABEL_GAP, SELF_OFFSET, THRESHOLD_GAP;
	static SHORT	START_X_POS, UNITED_START_X_POS;
	static SHORT	PEN_WIDTH;
#endif

#define START_UNITED		-1
#define START_THRESHOLD		-2
#define FINAL_THRESHOLD		-3

#define WAIT_TIME	10000.0

extern COLORREF PenColor[3];

#ifdef OLD_VARIABLE
void InitStaticVariable( CView* pView )
{
	rPIXEL	= GetLogOnePixel( pView );

	RADIUS	= (SHORT)(12 * rPIXEL);
	X_GAP	= (SHORT)(70 * rPIXEL);
	Y_GAP	= (SHORT)(50 * rPIXEL);
	Y_TOP	= Y_GAP * 5;
	META_Y_TOP = Y_GAP + RADIUS;

	MODEL_GAP	= (SHORT)(30 * rPIXEL);
	LABEL_GAP	= (SHORT)(20 * rPIXEL);
	SELF_OFFSET	= (SHORT)(15 * rPIXEL);
	THRESHOLD_GAP		= (SHORT)(40 * rPIXEL);

	START_X_POS			= (SHORT)(THRESHOLD_GAP * 2.5);
	UNITED_START_X_POS	= MODEL_GAP;

	PEN_WIDTH = rPIXEL;
}
#endif

void CVScore::SetWidthHeight()
{
	if (m_nKindOfModel == THRESHOLD_MODEL) {
		m_nWidth  = THRESHOLD_GAP * m_nNumState + RADIUS + THRESHOLD_GAP;
		m_nHeight =	3 * Y_GAP + 2 * RADIUS + SELF_OFFSET;
	} else {
		m_nWidth  = X_GAP * (m_nNumState-1) + 2 * RADIUS;
		m_nHeight =	2 * RADIUS + SELF_OFFSET;
	}
}

void CVScore::DrawSkipTransition( CDC* pDC, POINT ptFrom, POINT ptTo )
{
	pDC->Arc( ptFrom.x, ptFrom.y - RADIUS*3, ptTo.x, ptTo.y + RADIUS,
			  ptTo.x, ptTo.y - RADIUS, ptFrom.x, ptFrom.y - RADIUS );
}

void CVScore::DrawThresholdTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
								SHORT nFromState, SHORT nToState )
{
	SHORT nWid = (m_nNumState+1) * THRESHOLD_GAP / 2;
	SHORT DX, DY;

	DX = abs( ptFrom.x - ptTo.x );
	DY = abs( ptFrom.y - ptTo.y );

	if (nToState == START_THRESHOLD) {
		if (nFromState == FINAL_THRESHOLD) {
			pDC->Arc( ptFrom.x - nWid + RADIUS, ptTo.y, 
					  ptTo.x + nWid + RADIUS, ptFrom.y, 
					  ptFrom.x + RADIUS, ptFrom.y, 
					  ptTo.x + RADIUS, ptTo.y );
		} else if ((nFromState-m_nStartState)*2 == m_nNumState) {
			pDC->Arc( ptTo.x - RADIUS, ptTo.y + RADIUS, 
					  ptFrom.x + RADIUS, ptFrom.y - RADIUS, 
					  ptFrom.x, ptFrom.y - RADIUS, 
					  ptTo.x, ptTo.y + RADIUS );
		} else if ((nFromState-m_nStartState)*2 < m_nNumState) {
			pDC->MoveTo( ptFrom.x, ptFrom.y - RADIUS );
			pDC->LineTo( ptTo.x - RADIUS, ptTo.y );
		} else {
			pDC->MoveTo( ptFrom.x, ptFrom.y - RADIUS );
			pDC->LineTo( ptTo.x + RADIUS, ptTo.y );
		}
	} else if (nToState == FINAL_THRESHOLD) {
		if ((nFromState-m_nStartState)*2 == m_nNumState) {
			pDC->MoveTo( ptFrom.x, ptFrom.y + RADIUS );
			pDC->LineTo( ptTo.x, ptTo.y - RADIUS );
		} else if ((nFromState-m_nStartState)*2 < m_nNumState) {
			pDC->Arc( ptFrom.x, ptFrom.y + RADIUS - DY, 
					  ptTo.x - RADIUS + DX, ptTo.y, 
					  ptFrom.x, ptFrom.y + RADIUS, 
					  ptTo.x - RADIUS, ptTo.y );
		} else {
			pDC->Arc( ptTo.x + RADIUS - DX, ptFrom.y + RADIUS - DY, 
					  ptFrom.x, ptTo.y, 
					  ptTo.x + RADIUS, ptTo.y, 
					  ptFrom.x, ptFrom.y + RADIUS );
		}
	} else {
		if ((nToState-m_nStartState)*2 == m_nNumState) {
			pDC->MoveTo( ptFrom.x, ptFrom.y + RADIUS );
			pDC->LineTo( ptTo.x, ptTo.y - RADIUS );
		} else if ((nToState-m_nStartState)*2 < m_nNumState) {
			pDC->Arc( ptTo.x, ptFrom.y, 
					  ptFrom.x - RADIUS + DX, ptTo.y - RADIUS + DY, 
					  ptFrom.x - RADIUS, ptFrom.y, 
					  ptTo.x, ptTo.y - RADIUS );
		} else {
			pDC->Arc( ptFrom.x + RADIUS - DX, ptFrom.y, 
					  ptTo.x, ptTo.y - RADIUS + DY, 
					  ptTo.x, ptTo.y - RADIUS, 
					  ptFrom.x + RADIUS, ptFrom.y );
		}
	}
}

void CVScore::DrawTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
							 SHORT nKindOfModel )
{
	if ( nKindOfModel == THRESHOLD_MODEL ) {
		pDC->MoveTo( ptFrom.x, ptFrom.y + RADIUS );
		pDC->LineTo( ptTo.x, ptTo.y - RADIUS );
	} else {
		pDC->MoveTo( ptFrom.x + RADIUS, ptFrom.y );
		pDC->LineTo( ptTo.x - RADIUS, ptTo.y );
	}
}

void CVScore::DrawSelfTransition( CDC* pDC, POINT pt )
{
	SHORT NY = pt.y - SELF_OFFSET;
	SHORT nBottom = NY + RADIUS / 2;

	pDC->Arc( pt.x - RADIUS / 2, NY - RADIUS, 
			  pt.x + RADIUS / 2, nBottom, 
			  pt.x + RADIUS / 2, nBottom, 
			  pt.x - RADIUS / 2, nBottom );
}

void CVScore::DrawUnitedTransition( CDC* pDC, POINT ptFrom, POINT ptTo, 
								   SHORT nModel, SHORT nType )
{
	switch ( nType ) {
		case UNITED_FORWARD:
			pDC->MoveTo( ptFrom.x + RADIUS, ptFrom.y );
			pDC->LineTo( ptTo.x - RADIUS, ptTo.y );
			break;
		case UNITED_THRESHOLD:
			pDC->MoveTo( ptFrom.x - RADIUS, ptFrom.y );
			pDC->LineTo( ptTo.x, ptFrom.y );
			pDC->LineTo( ptTo.x, ptTo.y + RADIUS );
			break;
		default:
			SHORT nTop;

			if (pDC->m_hAttribDC)
				nTop = Y_TOP;
			else
				nTop = META_Y_TOP;

			pDC->MoveTo( ptFrom.x + RADIUS, ptFrom.y );
			pDC->LineTo( ptFrom.x + RADIUS + (THRESHOLD_GAP + THRESHOLD_GAP * (nModel % 3)) / 3, 
						 ptFrom.y );
			pDC->LineTo( ptFrom.x + RADIUS + (THRESHOLD_GAP + THRESHOLD_GAP * (nModel % 3)) / 3, 
					 nTop - Y_GAP );
			pDC->LineTo( ptTo.x, nTop - Y_GAP );
			pDC->LineTo( ptTo.x, ptTo.y - RADIUS );
			break;
	}
}

void CVScore::DrawMyState( CDC* pDC, POINT pt, SHORT nID, COLORREF rgbBrush )
{
	char	szNum[10];
	CBrush	*oldBrush, *newBrush;

	newBrush = new CBrush( rgbBrush );

	oldBrush = pDC->SelectObject( newBrush );
	pDC->Ellipse( pt.x-RADIUS, pt.y-RADIUS, pt.x+RADIUS, pt.y+RADIUS );
	pDC->SelectObject( oldBrush );
	delete newBrush;

	switch ( nID ) {
	case START_THRESHOLD: strcpy(szNum, "ST"); break;
	case FINAL_THRESHOLD: strcpy(szNum, "FT"); break;
	case START_UNITED: strcpy(szNum, "S"); break;
	default:
		sprintf( szNum, "%d", nID + m_nStartState );
		break;
	}
	DrawMyText( pDC, m_pView, pt, TRANSPARENT, 
					ALIGN_LRCENTER | ALIGN_TBCENTER, FALSE, 
					FALSE, FSIZE_SMALL, szNum );
}

void CVScore::DrawState( CDC* pDC, POINT pt, SHORT nID, BOOL bSelfLoop )
{
	DrawMyState( pDC, pt, nID, MY_PASSED );
	if (bSelfLoop)
		DrawSelfTransition( pDC, pt );
}

SHORT CVScore::GetSubState( SHORT nCurState, SHORT* pnModel )
{
	int		i;

	*pnModel = -1;

	switch( nCurState ) {
	case START_THRESHOLD:
	case FINAL_THRESHOLD:
		for(i=0; i<m_nSubModel; i++) {
			if (m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
				*pnModel = i;
				break;
			}
		}
	case START_UNITED:
		return (nCurState);
	default: break;
	}

	for(i=0; i<m_nSubModel; i++) {
		if ( nCurState >= m_pSubVScore[i].m_nStartState && 
			 nCurState <= m_pSubVScore[i].m_nFinalState ) {
			*pnModel = i;
			break;
		}
	}

	if ( *pnModel < 0)
		return nCurState;
	else
		return (nCurState - m_pSubVScore[i].m_nStartState);
}

POINT CVScore::GetStateLocation( CRect* prc, SHORT nCurState )
{
	POINT	pt;
	int		i;
	SHORT	nStartX, nStartY;
	SHORT	nModel, nState;

	nStartX = START_X_POS;

	switch ( m_nKindOfModel ) {
		case UNITED_MODEL:
			nState = GetSubState( nCurState, &nModel );
			if (nModel >= 0)
				return m_pSubVScore[nModel].GetStateLocation( prc, nState );

			i = 0;
			for(i=0; i<m_nSubModel; i++) {
				if (m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL)
					break;
			}
			pt.x = UNITED_START_X_POS;
			pt.y = (m_pSubVScore[i].m_nTopY == 0) 
					? (prc->bottom - 2 * Y_GAP) / 2 : m_pSubVScore[i].m_nTopY;
			break;

		case THRESHOLD_MODEL:
			nStartY = (m_nTopY == 0) ? (prc->bottom - 2 * Y_GAP) / 2 : m_nTopY;

			if ( nCurState == START_THRESHOLD || nCurState == FINAL_THRESHOLD ) {
				pt.x = nStartX + THRESHOLD_GAP * m_nNumState / 2;
				pt.y = nStartY;
				if (nCurState == FINAL_THRESHOLD)
					pt.y += 2 * Y_GAP;
			} else {
				pt.x = nStartX + nCurState * THRESHOLD_GAP;
				pt.y = nStartY + Y_GAP;
			}
			break;

		default:
			pt.x = nStartX + nCurState * X_GAP;
			pt.y = (m_nTopY == 0) ? prc->bottom / 2 : m_nTopY;
			break;
	}

	return pt;
}

void CVScore::DrawStateDiagram( CDC* pDC, CRect* prc, SHORT nCurModel, BOOL bUnited ) 
{
	int		i;
	POINT	ptStart, ptEnd, ptMid;

	ptStart = GetStateLocation( prc, 0 );
	if (nCurModel >= 0 && nCurModel <= MODEL_THRESHOLD) {
		ptStart.x = LABEL_GAP;
		DrawMyText( pDC, m_pView, ptStart, TRANSPARENT, 
				ALIGN_LEFT | ALIGN_TBCENTER, FALSE, FALSE, 
				FSIZE_NORMAL, GetModelFullName(nCurModel) );
	}

	if ( m_nKindOfModel == THRESHOLD_MODEL ) {
		ptStart = GetStateLocation( prc, START_THRESHOLD );
		ptEnd = GetStateLocation( prc, FINAL_THRESHOLD );

		DrawState( pDC, ptStart, START_THRESHOLD, FALSE );
		DrawState( pDC, ptEnd, FINAL_THRESHOLD, FALSE );

		if (!bUnited)
			DrawThresholdTransition( pDC, ptEnd, ptStart, FINAL_THRESHOLD, START_THRESHOLD );

		for(i=0; i<m_nNumState; i++) {
			ptMid = GetStateLocation( prc, i );
			DrawState( pDC, ptMid, i, TRUE );
			DrawThresholdTransition( pDC, ptStart, ptMid, START_THRESHOLD, 
					m_nStartState+i );
			DrawThresholdTransition( pDC, ptMid, ptEnd, m_nStartState+i, 
				FINAL_THRESHOLD );
		}
	} else {
		for(i=0; i<m_nNumState; i++) {
			ptEnd = GetStateLocation( prc, i );
			DrawState( pDC, ptEnd, i, (i != 0 && i != m_nNumState-1) );
			if ( i >= 1 ) {
				ptStart = GetStateLocation( prc, i-1 );
				DrawTransition( pDC, ptStart, ptEnd, m_nKindOfModel );
			}
			if ( !m_bSingleTransition && i >= 2 ) {
				ptStart = GetStateLocation( prc, i-2 );
				DrawSkipTransition( pDC, ptStart, ptEnd );
			}
		}
	}
}

void CVScore::DrawInitialDiagram( CMetaFileDC* pMetaDC, 
								 CRect* prc, SHORT nCurModel,
								 COLORREF rgbEnable, BOOL bUnited )
{
	POINT	ptStart;
	CDC*	pDC;
	CPen	*oldPen, *newPen;
	
	if ( pMetaDC )
		pDC = (CDC *)pMetaDC;
	else
		pDC = m_pView->GetDC();

	newPen = new CPen(PS_SOLID, PEN_WIDTH, MY_BLACK);
	oldPen = (CPen *)pDC->SelectObject( newPen );

	DrawStateDiagram( pDC, prc, nCurModel, bUnited );
	if ( !bUnited && m_nKindOfModel != THRESHOLD_MODEL) {		// Not unified model
		ptStart = GetStateLocation( prc, m_pnPath[0] );
		DrawMyState( pDC, ptStart, m_pnPath[0], rgbEnable );
	}

	pDC->SelectObject(oldPen);
	delete newPen;

	if (!pMetaDC)
		m_pView->ReleaseDC( pDC );
}

void CVScore::DrawViterbiScore( CDC* pDC, POINT ptLoc, REAL rScore, 
					  BOOL bEraseBk )
{
	char szTmp[20];

	sprintf( szTmp, "%.2f", rScore );
	ptLoc.y += (SHORT)(1.5 * RADIUS);
	DrawMyText( pDC, m_pView, ptLoc, OPAQUE, 
					ALIGN_LRCENTER | ALIGN_TOP, TRUE, FALSE, 
					FSIZE_SMALL, szTmp );
}

void CVScore::HighlightTransition( CMetaFileDC* pMetaDC, 
								  CVScore* pVScore, SHORT nT, 
								  CRect* prc, COLORREF rgbEnable )
{
	POINT	ptStart, ptEnd;
	CPen	*oldPen, *newPen;
	CDC*	pDC;

	if ( pMetaDC )
		pDC = (CDC *)pMetaDC;
	else
		pDC = m_pView->GetDC();

	if (nT > 0) {
		ptEnd = pVScore->GetStateLocation( prc, pVScore->m_pnPath[nT] );
		ptStart = pVScore->GetStateLocation( prc, pVScore->m_pnPath[nT-1] );
		DrawMyState( pDC, ptStart, 
				pVScore->m_pnPath[nT-1] - m_nStartState, MY_PASSED );

		newPen = new CPen(PS_SOLID, PEN_WIDTH, PenColor[(nT+1) % 3]);
		oldPen = (CPen *)pDC->SelectObject( newPen );

		if (pVScore->m_pnPath[nT] == pVScore->m_pnPath[nT-1])
			DrawSelfTransition( pDC, ptStart );
		else if ( m_nKindOfModel == THRESHOLD_MODEL ) {
			ptEnd = GetStateLocation( prc, FINAL_THRESHOLD );
			DrawThresholdTransition( pDC, ptStart, ptEnd, 
							pVScore->m_pnPath[nT-1], 
							FINAL_THRESHOLD );
			ptStart = ptEnd;
			ptEnd = GetStateLocation( prc, START_THRESHOLD );
			DrawThresholdTransition( pDC, ptStart, ptEnd, 
							FINAL_THRESHOLD, START_THRESHOLD );
			ptStart = ptEnd;
			ptEnd = pVScore->GetStateLocation( prc, pVScore->m_pnPath[nT] );
			DrawThresholdTransition( pDC, ptStart, ptEnd, 
							START_THRESHOLD, pVScore->m_pnPath[nT] );
		} else {
			if ( !m_bSingleTransition && 
				  (pVScore->m_pnPath[nT-1] == (pVScore->m_pnPath[nT]-2)) )
				DrawSkipTransition( pDC, ptStart, ptEnd );
			else
				DrawTransition( pDC, ptStart, ptEnd, m_nKindOfModel );
		}
	
		pDC->SelectObject(oldPen);
		delete newPen;
	} else if ( nT == 0 ) {
		ptEnd = pVScore->GetStateLocation( prc, pVScore->m_pnPath[nT] );
		if (m_nKindOfModel == THRESHOLD_MODEL) {			
			newPen = new CPen(PS_SOLID, PEN_WIDTH, PenColor[(nT+1) % 3]);
			oldPen = (CPen *)pDC->SelectObject( newPen );

			ptStart = pVScore->GetStateLocation( prc, START_THRESHOLD );
			DrawThresholdTransition( pDC, ptStart, ptEnd, 
							START_THRESHOLD, pVScore->m_pnPath[nT] );
			pDC->SelectObject(oldPen);
			delete newPen;
		}
	}

	if ( nT < 0 ) {
		ptEnd = GetStateLocation( prc, START_THRESHOLD );
		DrawMyState( pDC, ptEnd, START_THRESHOLD, rgbEnable );
		DrawViterbiScore( pDC, ptEnd, 0.0, TRUE );
	} else {
		DrawMyState( pDC, ptEnd, pVScore->m_pnPath[nT] - m_nStartState, rgbEnable );
		DrawViterbiScore( pDC, ptEnd, pVScore->GetViterbiScore(nT, FALSE), FALSE );
	}

	if ( !pMetaDC )
		m_pView->ReleaseDC( pDC );
}

BOOL CVScore::IsUnitedStartState( SHORT nState )
{
	for(int i=0; i<m_nSubModel; i++) {
		if ( m_pSubVScore[i].m_nKindOfModel != THRESHOLD_MODEL &&
			 nState == m_pSubVScore[i].m_nStartState)
			return TRUE;
	}

	return FALSE;
}

BOOL CVScore::IsUnitedFinalState( SHORT nState )
{
	for(int i=0; i<m_nSubModel; i++) {
		if ( m_pSubVScore[i].m_nKindOfModel != THRESHOLD_MODEL &&
			 nState == m_pSubVScore[i].m_nFinalState)
			return TRUE;
	}

	return FALSE;
}

void CVScore::DrawUnitedInitTransition( CDC* pDC, POINT ptStart, POINT ptEnd, 
					SHORT nT, SHORT nState, BOOL bForward )
{
	SHORT	nModel;
	CPen	*oldPen, *newPen;

	newPen = new CPen(PS_SOLID, PEN_WIDTH, PenColor[(nT+1) % 3]);
	oldPen = (CPen *)pDC->SelectObject( newPen );

	if ( bForward ) {
		DrawUnitedTransition( pDC, ptStart, ptEnd, NORMAL_MODEL, UNITED_FORWARD );
	} else {
		GetSubState( nState, &nModel );

		if (m_pSubVScore[nModel].m_nKindOfModel == THRESHOLD_MODEL )
			DrawUnitedTransition( pDC, ptStart, ptEnd, nModel, UNITED_THRESHOLD );
		else
			DrawUnitedTransition( pDC, ptStart, ptEnd, nModel, UNITED_NORMAL );
	}
	pDC->SelectObject(oldPen);
	delete newPen;
}

void CVScore::DrawUnitedThreshold( CDC* pDC, SHORT nT, CRect* prc, 
								 SHORT nCurModel, SHORT nState, BOOL bForward )
{
	POINT	ptStart, ptEnd;
	CPen	*oldPen, *newPen;

	newPen = new CPen(PS_SOLID, PEN_WIDTH, PenColor[(nT+1) % 3]);
	oldPen = (CPen *)pDC->SelectObject( newPen );

	if (bForward) {
		ptStart = GetStateLocation( prc, START_THRESHOLD );
		ptEnd = GetStateLocation( prc, nState );
		m_pSubVScore[nCurModel].DrawThresholdTransition( pDC, ptStart, ptEnd, 
					START_THRESHOLD, nState );
	} else {
		ptStart = GetStateLocation( prc, nState );
		ptEnd = GetStateLocation( prc, FINAL_THRESHOLD );
		m_pSubVScore[nCurModel].DrawThresholdTransition( pDC, ptStart, ptEnd, 
					nState, FINAL_THRESHOLD );
	}

	pDC->SelectObject(oldPen);
	delete newPen;
}

void CVScore::ModelToUnitedStart( CDC* pDC, SHORT nT, CRect* prc )
{
	SHORT	nCurModel;
	POINT ptStart, ptEnd;

	GetSubState( m_pnPath[nT], &nCurModel );
	ptStart = GetStateLocation( prc, m_pnPath[nT] );

	if (m_pSubVScore[nCurModel].m_nKindOfModel == THRESHOLD_MODEL) {
		DrawUnitedThreshold( pDC, nT, prc, nCurModel, m_pnPath[nT], FALSE );
		DrawMyState( pDC, ptStart, m_pnPath[nT], MY_PASSED );

		ptStart = GetStateLocation( prc, FINAL_THRESHOLD );
		ptEnd = GetStateLocation( prc, START_UNITED );
		DrawUnitedInitTransition( pDC, ptStart, ptEnd, 
					nT, FINAL_THRESHOLD, FALSE );
		DrawMyState( pDC, ptStart, FINAL_THRESHOLD, MY_PASSED );
	} else {
		ptEnd = GetStateLocation( prc, START_UNITED );
		DrawUnitedInitTransition( pDC, ptStart, ptEnd, 
					nT, m_pnPath[nT], FALSE );
		DrawMyState( pDC, ptStart, m_pnPath[nT], MY_PASSED );
	}
	ptStart = GetStateLocation( prc, m_pnPath[nT] );
	DrawViterbiScore( pDC, ptStart, GetViterbiScore(nT, FALSE), TRUE );
}

void CVScore::UnitedStartToModel( CDC* pDC, SHORT nT, CRect* prc )
{
	SHORT	nCurModel;
	POINT	ptStart, ptEnd;

	GetSubState( m_pnPath[nT], &nCurModel );
	ptStart = GetStateLocation( prc, START_UNITED );

	if (m_pSubVScore[nCurModel].m_nKindOfModel == THRESHOLD_MODEL) {
		ptEnd = GetStateLocation( prc, START_THRESHOLD );
		DrawUnitedInitTransition( pDC, ptStart, ptEnd, nT, 
					START_UNITED, TRUE );
		DrawMyState( pDC, ptStart, START_UNITED, MY_PASSED );

		DrawUnitedThreshold( pDC, nT, prc, nCurModel, m_pnPath[nT], TRUE );
		DrawMyState( pDC, ptEnd, START_THRESHOLD, MY_PASSED );
	} else {
		ptEnd = GetStateLocation( prc, m_pnPath[nT] );
		DrawUnitedInitTransition( pDC, ptStart, ptEnd, 
					nT, m_pnPath[nT], TRUE );
		DrawMyState( pDC, ptStart, START_UNITED, MY_PASSED );
	}
}

void CVScore::HighlightUnitedTransition( CMetaFileDC* pMetaDC,
						SHORT nT, CRect* prc, COLORREF rgbEnable )
{
	POINT	 ptLoc;
	CDC*	 pDC;

	if ( pMetaDC )
		pDC = (CDC *)pMetaDC;
	else
		pDC = m_pView->GetDC();
	
	if (nT > 0)
		ModelToUnitedStart( pDC, nT-1, prc );

	if ( nT >= 0 ) {
		UnitedStartToModel( pDC, nT, prc );

		ptLoc = GetStateLocation( prc, m_pnPath[nT] );
		DrawMyState( pDC, ptLoc, m_pnPath[nT], rgbEnable );
		DrawViterbiScore( pDC, ptLoc, GetViterbiScore(nT, FALSE), FALSE );
	} else {
		ptLoc = GetStateLocation( prc, START_UNITED );
		DrawMyState( pDC, ptLoc, START_UNITED, rgbEnable );
		DrawViterbiScore( pDC, ptLoc, 0.0, TRUE );
	}

	if ( !pMetaDC )
		m_pView->ReleaseDC( pDC );
}

BOOL CVScore::ShowRecognitionPath( SHORT nCurModel, CStroke* pStroke )
{
	POINT ptLoc;

	if ( !m_pnPath )
		return TRUE;

	CRect	 rc;

	m_pView->GetClientRect( &rc );

	ptLoc.x = rc.right - Y_GAP
			- (pStroke->m_Buf.rcBound.right - pStroke->m_Buf.rcBound.left);
	ptLoc.y = Y_TOP;
	pStroke->RepositionTrace( ptLoc );

	DrawInitialDiagram( NULL, &rc, nCurModel, MY_RED, FALSE );
	if (m_nTopY == 0) {
		if (!WaitMoment( WAIT_TIME, FALSE ))
			return FALSE;
	}

	for(int t=1; t<m_nNumTime; t++) {
		if (pStroke)
			pStroke->DrawTraceLine( m_pView, NULL, t );

		HighlightTransition( NULL, this, t, &rc, MY_RED );
		if (m_nTopY == 0) {
			if (!WaitMoment( WAIT_TIME, FALSE ))
				return FALSE;
		}
	}

	return TRUE;
}

BOOL CVScore::DrawHmmNetworks( CMetaFileDC* pMetaDC, BOOL bImmediate, REAL rWait,
					CVScore* pVScore, MODEL* pModel, SHORT nNumModel, 
					CStroke* pStroke )
{
	CRect	rc;
	SHORT	nTopY, nCurModel;

	if ( !pVScore )
		return TRUE;

	m_pView->GetClientRect( &rc );

	if ( pMetaDC )
		nTopY = META_Y_TOP;
	else
		nTopY = Y_TOP;

	for(int i=0; i<nNumModel; i++) {
		pVScore[i].SetWidthHeight();
		pVScore[i].m_nTopY = nTopY;

		if (pModel)
			nCurModel = i;
		else
			nCurModel = -1;

		pVScore[i].DrawInitialDiagram( pMetaDC, &rc, 
			nCurModel, MY_PASSED, m_nKindOfModel == UNITED_MODEL );
		nTopY += pVScore[i].m_nHeight + MODEL_GAP;
	}

	if (m_nKindOfModel == UNITED_MODEL) {
		POINT	ptFrom, ptTo, ptQuit;
		CDC*	pDC;
		CPen	*oldPen, *newPen;
		int     i;
		
		if ( pMetaDC )
			pDC = (CDC *)pMetaDC;
		else
			pDC = m_pView->GetDC();

		newPen = new CPen(PS_SOLID, PEN_WIDTH, MY_BLACK);
		oldPen = (CPen *)pDC->SelectObject( newPen );

		ptFrom = GetStateLocation( &rc, START_UNITED );
		DrawState( pDC, ptFrom, START_UNITED, FALSE );

		ptQuit = GetStateLocation( &rc, pVScore[MODEL_QUIT].m_nFinalState );
		ptQuit.x += RADIUS*3 + (THRESHOLD_GAP + THRESHOLD_GAP * (MODEL_QUIT % 3)) / 3;

		for(i=0; i<nNumModel; i++) {
			if (pVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
				ptTo = GetStateLocation( &rc, START_THRESHOLD );
				ptQuit.y = ptTo.y - RADIUS * 2;
				DrawUnitedTransition( pDC, ptFrom, ptTo, i, UNITED_FORWARD );
				ptTo = GetStateLocation( &rc, FINAL_THRESHOLD );
				DrawUnitedTransition( pDC, ptTo, ptFrom, i, UNITED_THRESHOLD );
			} else {
				ptTo = GetStateLocation( &rc, pVScore[i].m_nStartState );
				DrawUnitedTransition( pDC, ptFrom, ptTo, i, UNITED_FORWARD );
				ptTo = GetStateLocation( &rc, pVScore[i].m_nFinalState );
				DrawUnitedTransition( pDC, ptTo, ptFrom, i, UNITED_NORMAL );
				ptQuit.y = ptTo.y;
			}
			DrawMyText( pDC, m_pView, ptQuit, OPAQUE, 
					ALIGN_LEFT | ALIGN_TBCENTER, FALSE, FALSE, 
					FSIZE_LARGE, GetModelFullName(i) );
		}
	
		pDC->SelectObject(oldPen);
		delete newPen;
	
		if ( !pMetaDC )
			m_pView->ReleaseDC( pDC );
	}

	if ( !pMetaDC && pStroke ) {
		SHORT nFinal = pVScore[MODEL_QUIT].m_nFinalState 
					- pVScore[MODEL_QUIT].m_nStartState;
		POINT ptLoc = pVScore[MODEL_QUIT].GetStateLocation( &rc, nFinal );
		ptLoc.x += THRESHOLD_GAP + RADIUS;
		ptLoc.y = pVScore[0].m_nTopY;
		pStroke->RepositionTrace( ptLoc );
	}

	if ( !bImmediate && pStroke )
		pStroke->ClearTraceDraw( m_pView, pMetaDC );

	if ( !bImmediate && !pMetaDC && !WaitMoment( rWait, FALSE ) )
		return FALSE;

	return TRUE;
}

void CVScore::DrawOriginalHmmNetworks( CMetaFileDC* pMetaDC )
{
	CRect	 rc;
	SHORT	 nStart;
	POINT	 ptLoc;
	COLORREF rgb;
	int      i, j;

	if (!pMetaDC)
		return;

	m_pView->GetClientRect( &rc );
	DrawHmmNetworks( pMetaDC, TRUE, NO_WAIT, m_pSubVScore, 
				NULL, m_nSubModel, NULL );

	nStart = m_pSubVScore[m_nSubModel-1].m_nStartState;

	for(i=0; i<m_nSubModel-1; i++) {
		switch (i) {
			case 0:  rgb = MY_PINK;			break;
			case 1:  rgb = MY_GRAYRED;		break;
			case 2:  rgb = MY_GREEN;		break;
			case 3:  rgb = MY_SKYBLUE;		break;
			case 4:  rgb = MY_YELLOW;		break;
			case 5:	 rgb = MY_LIGHTPINK;	break;
			case 6:  rgb = MY_PURPLE;		break;
			case 7:  rgb = MY_DARKGREEN;	break;
			case 8:  rgb = MY_LIGHTPURPLE;	break;
			default:  rgb = MY_ORANGE;		break;
		}
		for(j=m_pSubVScore[i].m_nStartState + 1; 
					j<m_pSubVScore[i].m_nFinalState; j++) {

			ptLoc = GetStateLocation( &rc, j );
			DrawMyState( pMetaDC, ptLoc, j, rgb );

#ifdef OLD_COLORED_NETWORK
			ptLoc = GetStateLocation( &rc, nStart );
			DrawMyState( pMetaDC, ptLoc, nStart, rgb );
			nStart++;
#endif
		}
	}

#ifndef OLD_COLORED_NETWORK
	rgb = MY_GRAY;
	for( i = m_pSubVScore[m_nSubModel-1].m_nStartState; 
		 i <= m_pSubVScore[m_nSubModel-1].m_nFinalState; i++ ) {
		ptLoc = GetStateLocation( &rc, i );
		DrawMyState( pMetaDC, ptLoc, i, rgb );
	}
#endif
}

void CVScore::HighlightUnitedInternal( CMetaFileDC* pMetaDC, SHORT nT, 
				CRect* prc, COLORREF rgbEnable )
{
	SHORT curMdl, preMdl;

	GetSubState( m_pnPath[nT], &curMdl );

	if ( m_pSubVScore[curMdl].m_nKindOfModel == THRESHOLD_MODEL &&
		 m_pnPath[nT] != m_pnPath[nT-1] ) {
		HighlightUnitedTransition( pMetaDC, nT, prc, rgbEnable );
	} else {
		GetSubState( m_pnPath[nT-1], &preMdl );
		if ( curMdl != preMdl ) {
			return;
		}
		m_pSubVScore[curMdl].HighlightTransition( pMetaDC, 
					this, nT, prc, rgbEnable );
	}

	if ( nT == m_nNumTime-1 && rgbEnable == MY_RED &&
		 (IsUnitedFinalState(m_pnPath[nT]) ||
		  m_pSubVScore[curMdl].m_nKindOfModel == THRESHOLD_MODEL) ) {
		POINT	ptStart, ptEnd;
		CDC*	pDC;

		if ( pMetaDC )
			pDC = (CDC *)pMetaDC;
		else
			pDC = m_pView->GetDC();

		if (m_pSubVScore[curMdl].m_nKindOfModel == THRESHOLD_MODEL) {
			DrawUnitedThreshold( pDC, nT+1, prc, curMdl, m_pnPath[nT], FALSE );
			ptStart = GetStateLocation( prc, FINAL_THRESHOLD );
		} else
			ptStart = GetStateLocation( prc, m_pnPath[nT] );

		ptEnd = GetStateLocation( prc, START_UNITED );
		DrawUnitedInitTransition( pDC, ptStart, ptEnd, nT+1, 
						m_pnPath[nT], FALSE );
		DrawMyState( pDC, ptStart, m_pnPath[nT], MY_YELLOW );
		DrawMyState( pDC, ptEnd, START_UNITED, rgbEnable );
		DrawViterbiScore( pDC, ptEnd, GetViterbiScore(nT, FALSE), TRUE );

		if ( !pMetaDC )
			m_pView->ReleaseDC( pDC );
	}
}

void CVScore::DrawCurCandidate( CMetaFileDC* pMetaDC, 
						SHORT nT, CRect* prc, COLORREF rgbEnable )
{
	if ( IsUnitedStartState(m_pnPath[nT]) ||
		 IsUnitedFinalState(m_pnPath[nT-1]) ) {
		HighlightUnitedTransition( pMetaDC, nT, prc, rgbEnable );
	} else {
		HighlightUnitedInternal( pMetaDC, nT, prc, rgbEnable );
	}
}

BOOL CVScore::DrawUnitedModel( CMetaFileDC* pMetaDC, BOOL bImmediate, REAL rWait,
							  MODEL* pModel, CStroke* pStroke )
{
	CRect	rc;
	int		t;

	if ( !m_pSubVScore )
		return TRUE;

	if ( !DrawHmmNetworks(pMetaDC, bImmediate, rWait, m_pSubVScore, 
						  pModel, m_nSubModel, pStroke) )
		return FALSE;

	m_pView->GetClientRect( &rc );

	HighlightUnitedTransition( pMetaDC, -1, &rc, MY_RED );

	for(t=0; t<m_nNumTime; t++) {
		if (!bImmediate && pStroke)
			pStroke->DrawTraceLine( m_pView, pMetaDC, t );

		for(int i=0; i<m_nSubModel; i++) {
			BacktrackViterbi( m_nNumTime-1, m_pnSubFinal[i],
						m_pSubVScore[i].m_nKindOfModel ); 
			DrawCurCandidate( pMetaDC, t, &rc, MY_GREEN );
		}
		BacktrackViterbi( m_nNumTime-1, -1 );
		DrawCurCandidate( pMetaDC, t, &rc, MY_RED );

		if ( !bImmediate && !pMetaDC && !WaitMoment( rWait, FALSE ) )
			return FALSE;
	}

	return TRUE;
}


BOOL CVScore::ShowCurCandidate( CRect* prc, SHORT nT, COLORREF rgbEnable )
{
	if ( IsUnitedStartState(m_pnPath[nT]) ||
		 IsUnitedFinalState(m_pnPath[nT-1]) ) {
		HighlightUnitedTransition( NULL, nT, prc, rgbEnable );
	} else {
		HighlightUnitedInternal( NULL, nT, prc, rgbEnable );
	}

	return TRUE;
}

BOOL CVScore::ShowUnitedProcess( BOOL bImmediate, REAL rWait, MODEL* pModel, 
								CStroke* pStroke )
{
	CRect	rc;
	int		t;

	if ( !m_pSubVScore )
		return TRUE;

	if ( !DrawHmmNetworks(NULL, bImmediate, rWait, m_pSubVScore, pModel, 
						  m_nSubModel, pStroke) )
		return FALSE;

	m_pView->GetClientRect( &rc );

	for(t=0; t<m_nNumTime; t++) {
		if (!bImmediate && pStroke)
			pStroke->DrawTraceLine( m_pView, NULL, t );

		HighlightUnitedTransition( NULL, 0, &rc, MY_RED );
		if ( t > 2 ) {
			for(int m=1; m<=t; m++) {
				if (m < m_nNumTime - 1) {
					for(int mdl=0; mdl<m_nSubModel; mdl++) {
						BacktrackViterbi( t, m_pnSubFinal[mdl],
								m_pSubVScore[mdl].m_nKindOfModel );
						ShowCurCandidate( &rc, m, MY_GREEN );
					}
					BacktrackViterbi( m_nNumTime-1, -1 );
				}
				ShowCurCandidate( &rc, m, MY_RED );
			}

			if (!bImmediate && !WaitMoment( rWait, FALSE ))
				return FALSE;
		}
	}

	BacktrackViterbi( m_nNumTime-1, -1 );

	return TRUE;
}

BOOL DrawAllModel( CMetaFileDC* pMetaDC, BOOL bImmediate, REAL rWait,
			CVScore* pVScore, MODEL* pModel, SHORT nNumModel, CStroke* pStroke )
{
	CRect	rc;
	int		i, t;
	SHORT	mi;
	REAL	tmp, mscore;
	CView*	pView;

	if ( !pVScore[0].DrawHmmNetworks(pMetaDC, bImmediate, rWait, pVScore, 
									 pModel, nNumModel, pStroke) )
		return FALSE;

	pView = pVScore[0].m_pView;
	pView->GetClientRect( &rc );

	for(t=0; t<pVScore[0].m_nNumTime; t++) {
		if (!bImmediate && pStroke)
			pStroke->DrawTraceLine( pView, pMetaDC, t );

		for(mscore = cMaxREAL, i=0; i<nNumModel; i++) {
			tmp = pVScore[i].GetViterbiScore( t, TRUE );
			if( mscore > tmp ) {
				mi = i;
				mscore = tmp;
			}
			pVScore[i].HighlightTransition( pMetaDC, &pVScore[i], 
								t, &rc, MY_GREEN );
		}
		pVScore[mi].HighlightTransition( pMetaDC, &pVScore[mi], 
								t, &rc, MY_RED );

		if ( !bImmediate && !pMetaDC && !WaitMoment( rWait, FALSE ) )
			return FALSE;
	}

	return TRUE;
}

BOOL ShowMatchProcess( BOOL bImmediate, REAL rWait, CVScore* pVScore, 
					  MODEL* pModel, SHORT nNumModel, CStroke* pStroke )
{
	CRect	rc;
	int		i, t;
	SHORT	mi;
	REAL	tmp, mscore;
	CView*	pView;

	if ( !pVScore[0].DrawHmmNetworks(NULL, bImmediate, rWait, pVScore, 
					pModel, nNumModel, pStroke) )
		return FALSE;

	pView = pVScore[0].m_pView;
	pView->GetClientRect( &rc );

	for(t=0; t<pVScore[0].m_nNumTime; t++) {
		if (!bImmediate && pStroke)
			pStroke->DrawTraceLine( pView, NULL, t );

		for(i=0; i<nNumModel; i++)
			pVScore[i].BacktrackViterbi( t );

		for(int m=0; m<=t; m++) {
			for(mscore = cMaxREAL, i=0; i<nNumModel; i++) {
				tmp = pVScore[i].GetViterbiScore( m, FALSE );
				if( mscore > tmp ) {
					mi = i;
					mscore = tmp;
				}
				pVScore[i].HighlightTransition( NULL, &pVScore[i], 
								m, &rc, MY_GREEN );
			}
			pVScore[mi].HighlightTransition( NULL, &pVScore[mi], 
							m, &rc, MY_RED );
		}

		if (!bImmediate && !WaitMoment( rWait, FALSE ))
			return FALSE;
	}

	for(i=0; i<nNumModel; i++)
		pVScore[i].BacktrackViterbi( -1 );

	return TRUE;
}

void CalcAllModel( CVScore* pVScore, BOOL bUnited,
				   SHORT nNumModel, CStroke* pStroke, CRect* prc )
{
	CRect	rc;
	SHORT	nTopY;
	POINT	ptLoc;

	prc->left = 0;
	prc->top = 0;
	
	nTopY = META_Y_TOP;

	for(int i=0; i<nNumModel; i++) {
		pVScore[i].SetWidthHeight();
		pVScore[i].m_nTopY = nTopY;
		nTopY += pVScore[i].m_nHeight + MODEL_GAP;
	}

	pVScore[0].m_pView->GetClientRect( &rc );

	SHORT nFinal = pVScore[MODEL_THRESHOLD].m_nFinalState
					- pVScore[MODEL_THRESHOLD].m_nStartState;

	ptLoc = pVScore[MODEL_THRESHOLD].GetStateLocation( &rc, nFinal );
	prc->right = ptLoc.x + RADIUS*2;
	if ( !bUnited )
		prc->right += RADIUS*2;

	ptLoc = pVScore[MODEL_THRESHOLD].GetStateLocation( &rc, FINAL_THRESHOLD );
	prc->bottom = ptLoc.y + RADIUS*4;

	if ( pStroke ) {
		nFinal = pVScore[MODEL_QUIT].m_nFinalState
				- pVScore[MODEL_QUIT].m_nStartState;
		ptLoc = pVScore[MODEL_QUIT].GetStateLocation( &rc, nFinal );
		ptLoc.x += THRESHOLD_GAP + RADIUS;
		ptLoc.y = META_Y_TOP;
		pStroke->RepositionTrace( ptLoc );

		if (pStroke->m_Buf.rcBound.right > prc->right)
			prc->right = pStroke->m_Buf.rcBound.right + RADIUS;
	}
}
