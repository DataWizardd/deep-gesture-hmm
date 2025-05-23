// #include "pch.h"
#include <SDKDDKVer.h>       // by hklee#include <SDKDDKVer.h>       // by hklee

#include "stdafx.h"
#include "Spot.h"
#include "utils.h"
#include <math.h>
#include "CFlist.h"
#include "Cmgvq.h"
#include "Cdhmm.h"
#include "CVScore.h"
#include "Cstroke.h"
#include "GestureDoc.h"
#include "GestureView.h"

SHORT* CGestureDoc::GetCode( char* pszFileName, SHORT* pnData )
{
	CFVec	fvec;
	SHORT*	pnOutput;

	fvec.Read( pszFileName );
	pnOutput = new SHORT[fvec.m_nVector];
	if( fvec.m_nVecDim != m_pMGVQ->GetDimension() )
		Warning( "(GetCode) dimension mismatch" );
 
	for( int i = 0 ; i < fvec.m_nVector ; i ++ )
		pnOutput[i] = m_pMGVQ->Lookup( fvec, i );
	*pnData = fvec.m_nVector;

	return pnOutput;
}

SHORT* CGestureDoc::GetCode( CFVec* pFVec, SHORT* pnData )
{
	SHORT*	pnOutput;

	pnOutput = new SHORT[pFVec->m_nVector];
	if( pFVec->m_nVecDim != m_pMGVQ->GetDimension() )
		Warning( "(GetCode) dimension mismatch" );
 
	for( int i = 0 ; i < pFVec->m_nVector ; i ++ )
		pnOutput[i] = m_pMGVQ->Lookup( *pFVec, i );
	*pnData = pFVec->m_nVector;

	return pnOutput;
}

SHORT CGestureDoc::GetModelFromList( FILE* fp, char* pszLstFile, const char* pszModelName )
{
	SHORT	nM, nModel	= -1;
	char	szModel[10];
	char*	pszLine	= new char[MAX_BUFFER];
	
	while( fgets( pszLine, MAX_BUFFER-1, fp ) != NULL )
	{
		pszLstFile[0] = 0;
		if( pszLine[0] == '\n' || pszLine[0] == '#' )
			continue;
		sscanf( pszLine, "%s %s", szModel, pszLstFile );
		if (!szModel[0] || !pszLstFile[0])
			continue;
		
		if ( pszModelName && strcmpi(szModel, pszModelName) )
			continue;

		for(nM=0; nM<m_nModel; nM++) {
			if (!strcmpi(szModel, m_pModel[nM].szName)) {
				nModel = nM;
				break;
			}
		}

		if (nM >= m_nModel )
			PrintProgressString( "(GetModelFromList) invalid model name <%s>", szModel );
		else 
			break;
	}
	delete[] pszLine;

	return nModel;
}

BOOL CGestureDoc::TrainThresholdModel()
{
	int		i;
	REAL	prevLL=0.0, newLL;
	SHORT**	ppnOutput;
	SHORT*	pnOutput;

	if (!m_Flags.bLoadThresholdModel)
		return FALSE;

	GenerateThresholdModel();

	CFilelist filelist(	MakeMyFileName("RawData\\", "EtcRaw.dat") );

	ppnOutput = new SHORT*[filelist.m_nFile];
	pnOutput  = new SHORT[filelist.m_nFile];

	for(i=0; i<filelist.m_nFile; i++)
		ppnOutput[i] = GetCode(	filelist.GetFile(i), &pnOutput[i] );

	BOOL  bRet = TRUE;

	bRet = PrintProgressString( "==================== ( etc ) ===================" ); 
	while (bRet) {
		for(i=0; i<filelist.m_nFile; i++)
			m_pHMM[m_nThresholdModel].SetABCounts( ppnOutput[i], pnOutput[i] );
		newLL = m_pHMM[m_nThresholdModel].Reestimation();

		bRet = PrintProgressString( "etc: prevLL=%g   newLL=%g   fabs(prevLL-newLL)=%g",
				 prevLL, newLL, fabs(prevLL-newLL) );

		if( prevLL != 0.0 && fabs( newLL - prevLL ) < FLOATRESN ) {
//			printf("\n --- %g %g : %g ---\n", prevLL, newLL, newLL-prevLL );
			break;
		}
		prevLL = newLL;
	}

	for(i=0; i<filelist.m_nFile; i++)
		delete[] ppnOutput[i];
	delete[] ppnOutput;
	delete[] pnOutput;

	if (bRet) {
		m_pHMM[m_nThresholdModel].WriteText( MakeMyFileName("Model\\", "etc.tmd") );
		m_pHMM[m_nThresholdModel].Write( MakeMyFileName("Model\\", "etc.bmd") );
	}
	PrintProgressString( "Done etc" ); 

	return bRet;
}

BOOL CGestureDoc::TrainHMM( CFilelist& filelist, SHORT nState, SHORT nCurModel )
{
	int		i;
	CDHmm	hmm;
	REAL	prevLL=0.0, newLL;
	SHORT	**ppnOutput;
	SHORT	*pnOutput;

	ppnOutput = new SHORT*[filelist.m_nFile];
	pnOutput = new SHORT[filelist.m_nFile];

	if ( m_Flags.bSingleTransition )
		hmm.InitializeModel( nState, m_nSymbol );
	else
		hmm.NewModel( nState, m_nSymbol );

	for(i=0; i<filelist.m_nFile; i++)
		ppnOutput[i] = GetCode(	filelist.GetFile(i), &pnOutput[i] );

	BOOL bRet = TRUE;

	bRet = PrintProgressString( "==================== ( %s ) ===================", 
								m_pModel[nCurModel].szName );
	while ( bRet ) {
		for(i=0; i<filelist.m_nFile; i++)
			hmm.SetABCounts( ppnOutput[i], pnOutput[i] );
		newLL = hmm.Reestimation();

		bRet = PrintProgressString( "%s: prevLL=%g   newLL=%g   fabs(prevLL-newLL)=%g",
				 m_pModel[nCurModel].szName, prevLL, newLL, fabs(prevLL-newLL) );

		if( prevLL != 0.0 && fabs( newLL - prevLL ) < FLOATRESN ) {
//			printf("\n --- %g %g : %g ---\n", prevLL, newLL, newLL-prevLL );
			break;
		}
		prevLL = newLL;
	}

	for(i=0; i<filelist.m_nFile; i++)
		delete[] ppnOutput[i];
	delete[] ppnOutput;
	delete[] pnOutput;

	if (bRet) {
		hmm.WriteText( MakeMyFileName("Model\\", m_pModel[nCurModel].szName, "tmd") );
		hmm.Write( MakeMyFileName("Model\\", m_pModel[nCurModel].szName, "bmd") );
	}

	return bRet;
}

BOOL CGestureDoc::TrainGesture( const char* pszModelName )
{
	FILE*	fp;
	SHORT	nM;

	ClearDisplay( theApp.m_pGestView );

	if (!m_pMGVQ) {
		m_pMGVQ = new CMGVQ;
		m_pMGVQ->NewCodebook( m_nSymbol, TEST_DIMENSION );
		m_pMGVQ->Read( MakeMyFileName("VQ.out\\", "mgvq") );
	}
	if (!m_pMGVQ)
		return FALSE;

	if( !(fp = fopen( m_szTrainFile, "r" )) ) {
		Warning( "(TrainGesture) file <%s> not found", m_szTrainFile );
		return( FALSE );
	}

	BOOL  bRet = FALSE;
	char* pszLstFile = new char[MAX_BUFFER];

	while ( (nM = GetModelFromList(fp, pszLstFile, pszModelName)) >= 0 ) {
		if ( nM == m_nThresholdModel ) {
			PrintProgressString( "(TestGesture) invalid model name <etc>" );
			continue;
		}

		CFilelist filelist( pszLstFile );
		if ( !TrainHMM( filelist, m_pModel[nM].nState, nM ) ) {
			bRet = FALSE;
			break;
		}
		if (pszModelName)
			break;
	}
	fclose( fp );

	delete[] pszLstFile;

	ReadTrainedModel();

#ifdef GARBAGE_AUTO_TRAIN
	if ( bRet && m_Flags.bLoadThresholdModel )
		TrainThresholdModel();
#endif

	PrintProgressString( "Done!!!" );

	return bRet;
}

BOOL CGestureDoc::PrintRecogResult( char* pszBaseName, REAL* prLL, REAL rBest, 
								   SHORT nBest, SHORT nInModel )
{
	if ( prLL ) {
		for(int i=0; i<m_nModel; i++) 
			PutProgressString( "(%s = %.2f), ", m_pModel[i].szName, prLL[i] );
		PrintProgressString("");
	}

	if (nInModel < 0) {
		PrintProgressString( "LL=%g: %s", rBest, m_pModel[nBest].szName );
	} else {
		if (nInModel == nBest) {
			PrintProgressString( "%s:  %s   %s", pszBaseName, 
					m_pModel[nInModel].szName, m_pModel[nBest].szName );
		} else {
			if ( nBest < 0 ) {
				PrintProgressString( "%s:  %s   *******   XXXX   *******", 
						pszBaseName, m_pModel[nInModel].szName );
			} else {
				PrintProgressString( "%s:  %s   *******   %s   *******", 
						pszBaseName, m_pModel[nInModel].szName, 
						m_pModel[nBest].szName );
			}
		}
	}

	return TRUE;
}

CVScore* CGestureDoc::MatchNormalModel( char* pszBaseName, SHORT* pnOutput, 
		SHORT nOutput, SHORT nCurModel, SHORT* pnBestModel, BOOL bImmediate )
{
	int		 i;
	REAL	 bestLL, newLL;
	CVScore* pVScore = NULL;

	bestLL = cMaxREAL;
	*pnBestModel = -1;

	if (m_Flags.bShowRecognition)
		pVScore = new CVScore[m_nModel];

	for( i=0; i<m_nModel; i++ ) {
		if (pVScore) {
			pVScore[i].New( theApp.m_pGestView, nOutput, m_pHMM[i].m_nState );
			pVScore[i].m_bSingleTransition = m_Flags.bSingleTransition;
			if ( i == m_nThresholdModel )
				pVScore[i].m_nKindOfModel = THRESHOLD_MODEL;
			else
				pVScore[i].m_nKindOfModel = NORMAL_MODEL;
			newLL = m_pHMM[i].Viterbi( pnOutput, nOutput, pVScore[i] );
		} else {
			newLL = m_pHMM[i].Viterbi( pnOutput, nOutput );
		}

		if (newLL < bestLL)	{
			bestLL = newLL;
			*pnBestModel = i;
		}
	}

	REAL* prLL = NULL;

	if ( pVScore ) {
		prLL = new REAL[m_nModel];
		for(i=0; i<m_nModel; i++)
			prLL[i] = pVScore[i].m_rLikelihood;
	}

	if ( !bImmediate )
		PrintRecogResult( pszBaseName, prLL, bestLL, *pnBestModel, nCurModel );
	if ( prLL )
		delete[] prLL;

	return pVScore;
}

CVScore* CGestureDoc::MatchUnitedModel( char* pszBaseName, SHORT* pnOutput, 
		SHORT nOutput, SHORT nCurModel, SHORT* pnBestModel, BOOL bImmediate )
{
	int		 i, j;
	REAL	 bestLL;
	CVScore* pVScore = NULL;

	if (m_Flags.bShowRecognition) {
		pVScore = new CVScore;

		pVScore->New( theApp.m_pGestView, nOutput, m_pUnited->m_nState );
		pVScore->m_bSingleTransition = m_Flags.bSingleTransition;
		pVScore->m_nKindOfModel = UNITED_MODEL;
		pVScore->SetSubViterbiScore( m_pHMM, m_pModel, m_nModel, m_nThresholdModel );
		m_pUnited->UnitedViterbi( pnOutput, nOutput, 0, *pVScore );

		*pnBestModel = pVScore->GetBestUnited( &bestLL );	
	} else {
		SHORT*	pnPath  = new SHORT[nOutput];
		SHORT	nFinal  = m_nModel + m_pHMM[m_nThresholdModel].m_nState - 1;
		SHORT*	pnFinal = new SHORT[nFinal];

		pnFinal[0] = m_pHMM[0].m_nState - 1;

		for(i=1; i<m_nModel-1; i++) {
			pnFinal[i] = pnFinal[i-1] + m_pHMM[i].m_nState;
		}

		pnFinal[i] = pnFinal[i-1] + 1;
		for(j=1; j<m_pHMM[m_nThresholdModel].m_nState; j++) {
			pnFinal[i+j] = pnFinal[i] + j;
		}

		bestLL = m_pUnited->UnitedViterbi( pnOutput, nOutput, 
						pnPath, pnFinal, nFinal );

		SHORT nLast = pnPath[nOutput-1];
		delete[] pnPath;

		*pnBestModel = -1;
		for(i=0; i<nFinal; i++) {
			if ( nLast == pnFinal[i] ) {
				if (i < m_nThresholdModel)
					*pnBestModel = i;
				else
					*pnBestModel = m_nThresholdModel;
				break;
			}
		}
		delete[] pnFinal;
	}

	REAL* prLL = NULL;

	if ( pVScore ) {
		prLL = new REAL[m_nModel];
		for(i=0; i<m_nModel; i++) {
			if (i == m_nThresholdModel) {
				prLL[i] = cMaxREAL;
				pVScore->GetBestThresholdState( i, &prLL[i] );
			} else {
				prLL[i] = pVScore->m_pprScore[nOutput-1][pVScore->m_pnSubFinal[i]]; 
			}
		}
	}

	if ( !bImmediate )
		PrintRecogResult( pszBaseName, prLL, bestLL, *pnBestModel, nCurModel );
	if ( prLL )
		delete[] prLL;

	return pVScore;
}

CVScore* CGestureDoc::MatchModel( char* pszFileName, CFVec* pFVec,
					SHORT nCurModel, SHORT* pnBestModel, BOOL bImmediate )
{
	char*	pszBaseName;
	SHORT*	pnOutput;
	SHORT	nOutput;
	CVScore *pVScore;

	*pnBestModel = 0;

	if ( pszFileName )
		pnOutput = GetCode(	pszFileName, &nOutput );
	else if ( pFVec )
		pnOutput = GetCode(	pFVec, &nOutput );
	else
		return NULL;

	if (nCurModel < 0)
		pszBaseName = NULL;
	else
		pszBaseName = GetBaseName( pszFileName );

	if ( m_Flags.bTestUnitedModel ) {
		pVScore = MatchUnitedModel( pszBaseName, pnOutput, nOutput, 
							nCurModel, pnBestModel, bImmediate );
	} else {
		pVScore = MatchNormalModel( pszBaseName, pnOutput, nOutput,
							nCurModel, pnBestModel, bImmediate );
	}
	delete[] pnOutput;

	return pVScore;
}

BOOL CGestureDoc::DisplayUnitedMatching( CMetaFileDC* pMetaDC, 
				BOOL bImmediate, REAL rWait, CVScore* pVScore, CStroke* pStroke )
{
	pVScore->DisplayUnitedMatchPath( m_pModel );

	if ( m_Flags.bShowMatchProcess && !m_Flags.bMakeMetafile ) {
		if ( !pVScore->ShowUnitedProcess(bImmediate, rWait, m_pModel, pStroke) )
			return FALSE;
		PrintProgressString("Matching process display done!!" );
		if ( !bImmediate && !pMetaDC && !WaitMoment(LONG_WAIT, FALSE) )
			return FALSE;
	}

	if ( !pVScore->DrawUnitedModel(pMetaDC, bImmediate, rWait, m_pModel, pStroke) )
		return FALSE;

	if ( !bImmediate )
		PrintProgressString("Testing 'united' model done!!" );

	if ( rWait > NO_WAIT )
		rWait = LONG_WAIT;
	if ( !bImmediate && !pMetaDC && !WaitMoment(rWait, FALSE) )
		return FALSE;

	return TRUE;
}

void CGestureDoc::CalcMetaBoundingRect( CVScore* pVScore, SHORT nBestModel, 
							 CStroke* pStroke, CRect* prc )
{
	if (m_Flags.bTestUnitedModel) {
		CalcAllModel( pVScore->m_pSubVScore, TRUE,
				   pVScore->m_nSubModel, pStroke, prc );
	} else {
		CalcAllModel( pVScore, FALSE, m_nModel, pStroke, prc );
	}
}

BOOL CGestureDoc::FinishMetaRecording( CMetaFileDC* pMetaDC, SHORT nCurModel, 
							BOOL bImmediate, REAL rWait, CRect* prc, 
							BOOL bOK, SHORT nKind )
{
	if ( !pMetaDC )
		return bOK;

	HENHMETAFILE hMF;

	if ( !(hMF = pMetaDC->CloseEnhanced()) ) {
		delete pMetaDC;
		return FALSE;
	}

	if ( bOK ) {
		char* pszFile = NULL;
		
		switch (nKind) {
		case MO_STROKE:
			pszFile = MakeMyMetaName( "Meta\\", "strk", 
								m_nMetafileCount );
			break;
		case MO_GRAPH:
			pszFile = MakeMyMetaName( "Meta\\", "graph", 
								m_nMetafileCount );
			break;
		case MO_MODEL:
			if (nCurModel < 0) {
				pszFile = MakeMyMetaName( "Meta\\", "real", 
								m_nMetafileCount );
			} else {
				pszFile = MakeMyMetaName( "Meta\\", m_pModel[nCurModel].szName,
								m_nMetafileCount );
			}
			break;
		case MO_NETWORK:
			pszFile = MakeMyMetaName( "Meta\\", "net", 
								m_nMetafileCount );
			break;
		}

		if (pszFile) {
			CView* pView = theApp.m_pGestView;
			CDC* pDC = pView->GetDC();
			if ( ExportMetafile(pDC, hMF, pszFile, prc) && nKind == MO_MODEL)
				m_nMetafileCount++;
			pView->ReleaseDC( pDC );
		}
	}
	DeleteEnhMetaFile( hMF );

	delete pMetaDC;

	PrintProgressString("MetaFile recording done!!" );

	if ( rWait > NO_WAIT )
		rWait = LONG_WAIT;
	if ( !bImmediate && !pMetaDC && !WaitMoment(rWait, FALSE) )
		return FALSE;

	return bOK;
}

BOOL CGestureDoc::DisplayMatchingOutput( CVScore* pVScore, SHORT nCurModel,
			SHORT nBestModel, BOOL bImmediate, REAL rWait, CStroke* pStroke )
{
	if ( !pVScore || nBestModel < 0 || nBestModel >= m_nModel )
		return TRUE;

	BOOL		 bOK;
	CRect		 rc;
	CMetaFileDC* pMetaDC = NULL;

	if ( !m_Flags.bShowMatchProcess && m_Flags.bMakeMetafile ) {
		if (m_Flags.bTestUnitedModel && pVScore->m_pSubVScore) {
			CalcMetaBoundingRect( pVScore, nBestModel, NULL, &rc );
			pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
			pVScore->DrawOriginalHmmNetworks( pMetaDC );
			FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, 
					TRUE, MO_NETWORK );
		}
		CalcMetaBoundingRect( pVScore, nBestModel, pStroke, &rc );
		pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
	}

	if (m_Flags.bTestUnitedModel) {
		bOK = DisplayUnitedMatching( pMetaDC, bImmediate, rWait, pVScore, pStroke );
		return FinishMetaRecording( pMetaDC, nCurModel, bImmediate, rWait, 
					&rc, bOK, MO_MODEL );
	}

	for(int i=0; i<m_nModel; i++) {
		pVScore[i].DisplayMatchPath( m_pModel[i].szName, 
					i == nBestModel );
	}

	if ( m_Flags.bShowMatchProcess && !m_Flags.bMakeMetafile ) {
		if ( !ShowMatchProcess(bImmediate, rWait, pVScore, m_pModel, 
					m_nModel, pStroke) )
			return FALSE;
		PrintProgressString("Matching process display done!!" );
		if ( !bImmediate && !pMetaDC && !WaitMoment(LONG_WAIT, FALSE) )
			return FALSE;
	}

	bOK = DrawAllModel( pMetaDC, bImmediate, rWait, pVScore, m_pModel, 
						m_nModel, pStroke );
	if ( !FinishMetaRecording(pMetaDC, nCurModel, bImmediate, rWait, 
			&rc, bOK, MO_MODEL) )
		return FALSE;

	PrintProgressString("Testing model '%s' done!!", m_pModel[nBestModel].szName );
	if ( !bImmediate && !pMetaDC && !WaitMoment(rWait, FALSE) )
		return FALSE;

	return TRUE;
}

BOOL CGestureDoc::TestHMM( CFilelist& filelist, SHORT nCurModel )
{
	BOOL	 bOK;
	char*	 pszFile;
	CVScore* pVScore;
	SHORT	 bestModel, tmp;
	int		 i, correct = 0;

	tmp = 1;
	for( i=0; i<filelist.m_nFile; i++) {
		CStroke* pStroke = NULL;
		pszFile = GetBaseName( filelist.GetFile(i) );

		if ( m_Flags.bShowRecognition ) {
			pStroke = new CStroke;
			if (pStroke)
				pStroke->PrepareTrace( pszFile, NULL, 0 );
		}

		pVScore = MatchModel( filelist.GetFile(i), NULL, 
						nCurModel, &bestModel, FALSE );

		if ( m_Flags.bShowRecognition && !pVScore ) {
			if ( pStroke ) delete pStroke;
			return FALSE;
		}
		if (nCurModel == bestModel) correct++;

		bOK = DisplayMatchingOutput( pVScore, nCurModel, bestModel, 
								FALSE, SHORT_WAIT, pStroke );
		if (bOK && pVScore)
			ClearDisplay( theApp.m_pGestView );

		if ( pVScore ) {
			if (m_Flags.bTestUnitedModel) delete pVScore;
			else delete[] pVScore;
		}
		if ( pStroke ) delete pStroke;
		if (!bOK) return FALSE;
	}

	PrintProgressString( "===========================" );
	PrintProgressString( "Total data  : %d", filelist.m_nFile );
	PrintProgressString( "Correct data: %d", correct );
	PrintProgressString( "Recognition rate(%%) = %g", 
					 (double)correct / (double)filelist.m_nFile * 100.0 );

	if (nCurModel < m_nModel-1) {
		if ( !WaitMoment( LONG_WAIT, TRUE ) )
			return FALSE;
	}

	return( TRUE );
}

void CGestureDoc::AdjustSpotting()
{
	if (m_nRecogOut <= 0)
		return;

	RECOUT* pPrev = &m_pRecogOut[m_nRecogOut-1];
	RECOUT* pCur  = &m_pRecogOut[m_nRecogOut];

	if ( pPrev->nFirstEndT > pCur->nStartT ) {
		pPrev->nStartT = pCur->nStartT;
		pPrev->nEndT = pCur->nEndT;
		pPrev->nFirstEndT = pCur->nFirstEndT;
		pPrev->nModel = pCur->nModel;
		m_nRecogOut--;
		if (m_nRecogOut <= 0)
			return;

		pPrev = &m_pRecogOut[m_nRecogOut-1];
		pCur  = &m_pRecogOut[m_nRecogOut];
	}
}

void CGestureDoc::SpottingOneModel( CVScore* pVScore, SHORT nBestModel )
{
	SHORT nStart = 0;
	SHORT nStartState = pVScore->m_pSubVScore[nBestModel].m_nStartState;

	for(int i=pVScore->m_nNumTime-1; i>=0; i--) {
		if (pVScore->m_pnPath[i] == nStartState) {
			nStart = i;
			break;
		}
	}
	if ( (pVScore->m_nNumTime - nStart) < m_pModel[nBestModel].nMinTime )
		return;

	m_pRecogOut[m_nRecogOut].nModel = nBestModel;
	m_pRecogOut[m_nRecogOut].nEndT = pVScore->m_nNumTime; //-1;

	if ( m_pRecogOut[m_nRecogOut].nFirstEndT < 0 ) {
		m_pRecogOut[m_nRecogOut].nStartT = nStart;
		m_pRecogOut[m_nRecogOut].nFirstEndT = pVScore->m_nNumTime; //-1;

		AdjustSpotting();
	}
}

void CGestureDoc::FlushSpotting()
{
	if ( m_pRecogOut[m_nRecogOut].nModel != m_nThresholdModel ) {
		m_nRecogOut++;
		m_pRecogOut[m_nRecogOut].nModel = m_nThresholdModel;
		m_pRecogOut[m_nRecogOut].nFirstEndT = -1;
	}
}


void CGestureDoc::PeekDetection( CVScore* pVScore, SHORT bestModel )
{
	if ( !pVScore || !m_Flags.bTestUnitedModel 
		|| !m_pRecogOut || m_nRecogOut >= MAX_GESTURE )
		return;

	if (bestModel >= m_nThresholdModel)
		bestModel = m_nThresholdModel;

	if ( bestModel >= 0 && bestModel != m_nThresholdModel ) {
		if ( m_nRecogOut != 0 || pVScore->m_nNumTime >= 5 ) {
			if ( m_pRecogOut[m_nRecogOut].nModel == m_nThresholdModel ||
						m_pRecogOut[m_nRecogOut].nModel == bestModel) {
				SpottingOneModel( pVScore, bestModel );
			} else {
				m_nRecogOut++;
				m_pRecogOut[m_nRecogOut].nModel = m_nThresholdModel;
				m_pRecogOut[m_nRecogOut].nFirstEndT = -1;
				SpottingOneModel( pVScore, bestModel );
			}
		}
	} else if ( m_pRecogOut[m_nRecogOut].nModel != m_nThresholdModel ) {
		m_nRecogOut++;
		m_pRecogOut[m_nRecogOut].nModel = m_nThresholdModel;
		m_pRecogOut[m_nRecogOut].nFirstEndT = -1;
	}
}

SHORT CGestureDoc::TestHMM( CFVec* pFVec, SHORT nCurModel, BOOL bImmediate, 
						   REAL rWait, BOOL bLast, CStroke* pStroke )
{
	CVScore* pVScore;
	SHORT	 bestModel;

	pVScore = MatchModel( NULL, pFVec, -1, &bestModel, bImmediate );

	if ( m_Flags.bShowRecognition && !pVScore )
		return FALSE;

	if ( m_Flags.bMakeMetafile && m_Flags.bTestUnitedModel && pVScore->m_pSubVScore) {
		CRect rc;
		static BOOL bDone = FALSE;

		if ( !bDone ) {
			CalcMetaBoundingRect( pVScore, bestModel, NULL, &rc );
			CMetaFileDC* pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
			pVScore->DrawOriginalHmmNetworks( pMetaDC );
			FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, 
					TRUE, MO_NETWORK );

			bDone = TRUE;
		}
	}

	PeekDetection( pVScore, bestModel );
	if ( bLast )
		PeekDetection( pVScore, -1 );

	if ( pVScore && m_pUnited && !m_Flags.bFirstTraceViterbi )
		m_pUnited->TraceViterbiScore( *pVScore );

	if ( nCurModel > m_nModel) nCurModel = m_nModel;
	if ( bestModel > m_nModel) bestModel = m_nModel;

	if ( rWait != NO_WAIT ) {
		DisplayMatchingOutput( pVScore, nCurModel, bestModel, 
						bImmediate, rWait, pStroke );
	}

	if (pVScore) {
		if (m_Flags.bTestUnitedModel)
			delete pVScore;
		else
			delete[] pVScore;
	}

	return bestModel;
}

SHORT CGestureDoc::TestUnitedHMM( REAL rWait, BOOL bLast, CStroke* pStroke )
{
	REAL	bestLL;
	SHORT	bestModel;

	m_pUnited->UnitedViterbi( m_pnOutput, m_nOutput, m_nProcessedTime, *m_pVScore );
	m_nProcessedTime = m_nOutput;
	m_pVScore->m_nNumTime = m_nOutput;
	bestModel = m_pVScore->GetBestUnited( &bestLL );	

	if ( !m_Flags.bShowRecognition )
		return bestModel;

	if ( m_Flags.bMakeMetafile && m_pVScore->m_pSubVScore) {
		CRect rc;
		static BOOL bDone = FALSE;

		if ( !bDone ) {
			CalcMetaBoundingRect( m_pVScore, bestModel, NULL, &rc );
			CMetaFileDC* pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
			m_pVScore->DrawOriginalHmmNetworks( pMetaDC );
			FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, 
					TRUE, MO_NETWORK );

			bDone = TRUE;
		}
	}

	PeekDetection( m_pVScore, bestModel );
	if ( bLast )
		PeekDetection( m_pVScore, -1 );

	if ( !m_Flags.bFirstTraceViterbi )
		m_pUnited->TraceViterbiScore( *m_pVScore );

	if ( bestModel > m_nModel) bestModel = m_nModel;

	if ( rWait != NO_WAIT ) {
		DisplayMatchingOutput( m_pVScore, -1, bestModel, 
						TRUE, rWait, pStroke );
	}

	return bestModel;
}

BOOL CGestureDoc::TestGesture( const char* pszModelName )
{
	FILE*	fp;
	SHORT	nM;
	CFilelist* pFilelist;

	ClearDisplay( theApp.m_pGestView );

	if (m_Flags.bTestTrainData) {
		if( !(fp = fopen( m_szTrainFile, "r" )) ) {
			Warning( "(CGestureDoc::TestGesture) file <%s> not found", m_szTestFile );
			return( FALSE );
		}
	} else {
		if( !(fp = fopen( m_szTestFile, "r" )) ) {
			Warning( "(CGestureDoc::TestGesture) file <%s> not found", m_szTestFile );
			return( FALSE );
		}
	}

	if (!m_pHMM || (m_Flags.bTestUnitedModel && !m_pUnited))
		ReadTrainedModel();

	char* pszLstFile = new char[MAX_BUFFER];

	while ( (nM = GetModelFromList(fp, pszLstFile, pszModelName)) >= 0 ) {
		if ( !m_Flags.bTestUnitedModel && !m_Flags.bLoadThresholdModel && nM == m_nThresholdModel ) {
			PrintProgressString( "(TestGesture) you must load threshold model first!!!" );
			continue;
		}

		pFilelist = new CFilelist( pszLstFile );
		if (pFilelist->m_nMaxFile > 0) {
			if ( !TestHMM( *pFilelist, nM ) ) {
				delete pFilelist;
				break;
			}
		}
		delete pFilelist;

		if (pszModelName)
			break;
	}
	fclose( fp );

	delete[] pszLstFile;

	PrintProgressString( "Done!!!" );

	return( TRUE );
}

SHORT CGestureDoc::OnlineTestGesture( CFVec* pFVec, SHORT nCurModel,
					BOOL bImmediate, const POINT* pPt, const SHORT nPt )
{
	SHORT nRecogOut;

	if (!m_pHMM || (m_Flags.bTestUnitedModel && !m_pUnited))
		ReadTrainedModel();

	CStroke* pStroke = NULL;

	if ( m_Flags.bShowRecognition ) {
		pStroke = new CStroke;
		if (pStroke)
			pStroke->PrepareTrace( NULL, pPt, nPt );
	}

	nRecogOut = TestHMM( pFVec, nCurModel, bImmediate, SHORT_WAIT, FALSE, pStroke );

	if (pStroke)
		delete pStroke;

	return nRecogOut;
}

BOOL CGestureDoc::ImmediateProcessing( CStroke* pStroke )
{
	CFVec*	 pFVec;
	CStroke* pTmp;
	SHORT	 nPrevFVec = -1;

	if (!m_pHMM || (m_Flags.bTestUnitedModel && !m_pUnited))
		ReadTrainedModel();

	m_pUnited->InitTraceViterbiScore( m_nModel );
	m_Flags.bFirstTraceViterbi = FALSE;	

	ClearDisplay( theApp.m_pGestView );

	if ( !m_pRecogOut )
		m_pRecogOut = new RECOUT[MAX_GESTURE];
	m_nRecogOut = 0;
	m_pRecogOut[0].nModel = m_nThresholdModel;
	m_pRecogOut[0].nFirstEndT = -1;

	for(int i=1; i<pStroke->m_Buf.nPt; i++) {
		pFVec = pStroke->MakeImmediateVector( i );
		if (pFVec) {
			pTmp = NULL;

			if ( m_Flags.bShowRecognition ) {
				pTmp = new CStroke;
				memcpy( pTmp->m_Buf.pPt, pStroke->m_Norm.pPt,
						sizeof(POINT) * pStroke->m_Norm.nPt );
				pTmp->m_Buf.nPt = pStroke->m_Norm.nPt;
			}

			if ( i == pStroke->m_Buf.nPt-1 || nPrevFVec < pFVec->m_nVector ) {
				TestHMM( pFVec, -1, FALSE, NO_WAIT, 
						(i == pStroke->m_Buf.nPt-1), pTmp );
				if ( !WaitMoment(NO_WAIT, TRUE) ) {
					delete pFVec;
					if (pTmp)
						delete pTmp;
					m_pUnited->DestroyTraceViterbiScore();
					m_Flags.bFirstTraceViterbi = TRUE;
					return FALSE;
				}
				ClearDisplay( theApp.m_pGestView );
			}
			nPrevFVec = pFVec->m_nVector;
			delete pFVec;

			if ( pTmp )
				delete pTmp;
		}
	}

	CRect	rc;
	CView*	pView = theApp.m_pGestView;
	CMetaFileDC* pMetaDC = NULL;

	ClearDisplay( pView );

	if ( m_Flags.bMakeMetafile ) {
		m_pUnited->CalcTraceBoundingRect( &rc );
		pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
	}

	m_pUnited->DrawTraceViterbiScore( pMetaDC, pView, m_pModel, m_nModel, 
				m_pRecogOut, m_nRecogOut, -1 );
	m_pUnited->DestroyTraceViterbiScore();
	m_Flags.bFirstTraceViterbi = TRUE;
	FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, TRUE, MO_GRAPH );

	pMetaDC = NULL;
	if ( m_Flags.bMakeMetafile ) {
		pStroke->CalcSpotBoundingRect( theApp.m_pGestView, &rc );
		pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
	}
	pStroke->DrawSpottingResult( pMetaDC, pView, m_pRecogOut, m_nRecogOut, 
					-1, m_pModel, m_nThresholdModel );
	FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, TRUE, MO_STROKE );

	if ( !WaitMoment( LONG_WAIT*3.0, TRUE ) )
		return FALSE;

	ClearDisplay( pView );

	return TRUE;
}

BOOL CGestureDoc::StartGestureRecognition( CStroke* pStroke, BOOL bUseFrameID )
{
	if (!m_pHMM || !m_pUnited)
		ReadTrainedModel();

	if (!m_pHMM || !m_pUnited)
		return FALSE;

	m_nProcessedTime = 0;

	if ( m_pVScore )
		delete m_pVScore;
	m_pVScore = new CVScore;
	m_pVScore->New( theApp.m_pGestView, MAX_OBSERVATION, m_pUnited->m_nState );
	m_pVScore->m_bSingleTransition = m_Flags.bSingleTransition;
	m_pVScore->m_nKindOfModel = UNITED_MODEL;
	m_pVScore->SetSubViterbiScore( m_pHMM, m_pModel, m_nModel, m_nThresholdModel );

	if ( m_pFVec )
		delete[] m_pFVec;
	m_nFVec = MAX_OBSERVATION;
	m_pFVec = new CFVec[m_nFVec];
	m_pFVec->New( MAX_OBSERVATION, TEST_DIMENSION );

	if ( m_pnOutput )
		delete[] m_pnOutput;
	m_nOutput = 0;
	m_pnOutput = new SHORT[MAX_OBSERVATION];

	if ( !m_pRecogOut )
		m_pRecogOut = new RECOUT[MAX_GESTURE];
	m_nRecogOut = 0;
	m_pRecogOut[0].nModel = m_nThresholdModel;
	m_pRecogOut[0].nFirstEndT = -1;

	m_pTmpStroke = NULL;
	if (bUseFrameID)
		pStroke->AllocFrameID();

	if (m_Flags.bShowRecognition)
		m_pTmpStroke = new CStroke;

	m_pUnited->InitTraceViterbiScore( m_nModel );
	m_Flags.bFirstTraceViterbi = FALSE;	

	ClearDisplay( theApp.m_pGestView );

	return TRUE;
}


//-- For PowerPoint7 : Old Version

//#define PPT_GO_END		0x000101C6L			// Last
//#define	PPT_GO_HOME		0x000101C5L			// First
//#define	PPT_GO_NEXT		0x000101BFL			// Next
//#define	PPT_GO_PREV		0x000101C0L			// Previous
//#define	PPT_START		0x00010031L			// Start presentation
//#define	PPT_BYE			0x00010017L			// Quit PowerPoint
//#define PPT_WHITE		0x000101C2L			// White screen(450)
//#define PPT_BLACK		0x000101C1L			// Black screen
//#define PPT_HIDDEN		0x000101CBL			// Show hidden slide
//#define	PPT_STOP		0x000101D1L			// Finish presentation


// -- For PowerPoint97
#define PPT_GO_END		0x00010190L			// Last
#define	PPT_GO_HOME		0x0001018FL			// First
#define	PPT_GO_NEXT		0x00010189L			// Next
#define	PPT_GO_PREV		0x0001018AL			// Previous
#define	PPT_START		0x00010031L			// Start presentation
#define	PPT_BYE			0x00010019L			// Quit PowerPoint
#define PPT_WHITE		0x0001018CL			// White screen(450)
#define PPT_BLACK		0x0001018BL			// Black screen
#define PPT_HIDDEN		0x00010195L			// Show hidden slide
#define	PPT_STOP		0x0001019BL			// Finish presentation




void CGestureDoc::SendPowerPointMessage( SHORT nCurModel )
{
	if ( nCurModel < 0 || nCurModel >= m_nThresholdModel )
		return;

	HWND	pptWnd;
	
	if ( nCurModel == MODEL_QUIT || nCurModel == MODEL_BYE )
		pptWnd = FindWindow ("PP97FrameClass", NULL);
	else
		pptWnd = FindWindow ("screenClass", NULL);

	if ( !pptWnd )
		return;

    switch( nCurModel ) {
		case MODEL_LAST:
			SendMessage (pptWnd, WM_COMMAND, PPT_GO_END, 0L) ;
			break;
		case MODEL_FIRST:
			SendMessage (pptWnd, WM_COMMAND, PPT_GO_HOME, 0L) ;
			break;
		case MODEL_NEXT:
			SendMessage (pptWnd, WM_COMMAND, PPT_GO_NEXT, 0L) ;
			break;
		case MODEL_PREV:
			SendMessage (pptWnd, WM_COMMAND, PPT_GO_PREV, 0L) ;
			break;
		case MODEL_QUIT:		// Start presentation
			SendMessage (pptWnd, WM_COMMAND, PPT_START, 0L) ;
			break;
		case MODEL_BYE:			// Exit PowerPoint
			SendMessage (pptWnd, WM_COMMAND, PPT_BYE, 0L) ;
			break;
		case MODEL_CIRCLE:		// White screen
			SendMessage (pptWnd, WM_COMMAND, PPT_WHITE, 0L) ;
			break;
		case MODEL_TRIANGLE:	// Black screen
			SendMessage (pptWnd, WM_COMMAND, PPT_BLACK, 0L) ;
			break;
		case MODEL_SAW:			// Show hidden page
			SendMessage (pptWnd, WM_COMMAND, PPT_HIDDEN, 0L) ;
			break;
		case MODEL_ALPHA:		// Finish presentation
			SendMessage (pptWnd, WM_COMMAND, PPT_STOP, 0L) ;
			break;
		default:
			break;
	}
}

void CGestureDoc::ResetSpottingRecognition( CStroke* pStroke, BOOL bResetAll )
{
	SHORT nResetT = 0;
	int i;
	
	if (bResetAll)
		nResetT = pStroke->m_Norm.nPt;
	else {
		if (m_pRecogOut[0].nModel == m_nThresholdModel) {
			if (pStroke->m_Norm.nPt >= MAX_RESAMPLE_DATA)
				nResetT = MAX_FRAME_INPUT;
		} else {
			if (m_pRecogOut[1].nModel == m_nThresholdModel) {
				nResetT = m_pRecogOut[0].nFirstEndT;
			} else {
				nResetT = min(m_pRecogOut[0].nEndT, m_pRecogOut[1].nStartT);
				for(int i=2; i<=m_nRecogOut; i++) {
					if (m_pRecogOut[i].nModel != m_nThresholdModel)
						nResetT = min(nResetT, m_pRecogOut[i].nStartT);
				}
			}
		}
	}

	m_nProcessedTime = 0;
	m_nFVec = 0;

	if (nResetT == 0)
		return;

	m_pUnited->ResetTraceViterbiScore( m_nModel, &nResetT );
	for(i=nResetT; i<pStroke->m_Norm.nPt; i++) {
		pStroke->m_Norm.pPt[i-nResetT] = pStroke->m_Norm.pPt[i];
		if ( pStroke->m_Norm.pnFrameID )
			pStroke->m_Norm.pnFrameID[i-nResetT] = pStroke->m_Norm.pnFrameID[i];
	}
	pStroke->m_Norm.nPt -= nResetT;

	if (bResetAll) {
		for(i=0; i<=m_nRecogOut; i++) {
			SetStatusMessage("Recognized: %s", 
					GetModelFullName(m_pRecogOut[i].nModel));
			SendPowerPointMessage(m_pRecogOut[i].nModel);
		}

		m_nRecogOut = 0;
		m_pRecogOut[0].nModel = m_nThresholdModel;
		m_pRecogOut[0].nFirstEndT = -1;
	} else if (m_nRecogOut >= 1) {
		SetStatusMessage("Recognized: %s", GetModelFullName(m_pRecogOut[0].nModel));
		SendPowerPointMessage(m_pRecogOut[0].nModel);

		for(i=1; i<=m_nRecogOut; i++) {
			m_pRecogOut[i-1].nStartT = m_pRecogOut[i].nStartT - nResetT;
			if (m_pRecogOut[i].nFirstEndT < 0)
				m_pRecogOut[i-1].nFirstEndT = -1;
			else
				m_pRecogOut[i-1].nFirstEndT = m_pRecogOut[i].nFirstEndT - nResetT;
			m_pRecogOut[i-1].nEndT = m_pRecogOut[i].nEndT - nResetT;
			m_pRecogOut[i-1].nModel = m_pRecogOut[i].nModel;
		}
		m_nRecogOut--;
	}
}

BOOL CGestureDoc::SaveRecogOutData( SHORT nRecOut, RECOUT* pRecOut, CStroke* pStroke )
{
	FILE*	fp;
	char*	pszFile;

	pszFile = MakeMyResultName( NAME_RECOGOUT );
	if ( !pszFile )
		return FALSE;

	if (!(fp = fopen(pszFile, "a"))) {
		MessageBeep(MB_ICONHAND);
		SetStatusMessage("Invalid recog_out file: %s", pszFile);
		return FALSE;
	}

	for(int i=0; i<nRecOut; i++) {
		fprintf(fp, "%s %d %d %d\n", 
				GetModelName(pRecOut[i].nModel),
				pStroke->m_Norm.pnFrameID[pRecOut[i].nStartT],
				pStroke->m_Norm.pnFrameID[pRecOut[i].nFirstEndT],
				pStroke->m_Norm.pnFrameID[pRecOut[i].nEndT] );
	}
	fclose(fp);

	return TRUE;
}

BOOL CGestureDoc::LoadRecogOutData( SHORT* pnRecOut, RECOUT* pRecOut )
{
	FILE*	fp;
	int		t1, t2, t3;
	char*	pszFile, szName[20];

	pszFile = MakeMyResultName( NAME_RECOGOUT );
	if ( !pszFile )
		return FALSE;

	if (!(fp = fopen(pszFile, "r"))) {
		MessageBeep(MB_ICONHAND);
		SetStatusMessage("Invalid recog_out file: %s", pszFile);
		return FALSE;
	}

	*pnRecOut = 0;
	while ( fscanf(fp, "%s %d %d %d\n", szName, &t1, &t2, &t3) != EOF ) {
		pRecOut[*pnRecOut].nModel = -1;
		for(int i=0; i<MODEL_THRESHOLD; i++) {
			if ( !strcmpi(szName, GetModelName(i)) ) {
				pRecOut[*pnRecOut].nModel = i;
				break;
			}
		}

		if (pRecOut[*pnRecOut].nModel == -1)
			break;

		pRecOut[*pnRecOut].nStartT = t1;
		pRecOut[*pnRecOut].nFirstEndT = t2;
		pRecOut[*pnRecOut].nEndT = t3;

		(*pnRecOut)++;
	}
	fclose(fp);

	return TRUE;
}

BOOL CGestureDoc::DisplaySpottedGesture( CStroke* pStroke, BOOL bResetAll, REAL rWait )
{
	CRect	rc;
	SHORT	nDrawT, nRecOut;
	CView*	pView = theApp.m_pGestView;
	CMetaFileDC* pMetaDC = NULL;

	ClearDisplay( pView );

	if ( m_Flags.bMakeMetafile ) {
		m_pUnited->CalcTraceBoundingRect( &rc );
		pMetaDC = InitMetaFileDC( theApp.m_pGestView, &rc );
	}

	if (bResetAll) {
		FlushSpotting();
		nDrawT = -1;
		nRecOut = m_nRecogOut;
	} else {
		nDrawT = pStroke->m_Norm.nPt;
		nRecOut = 0;
		if (m_pRecogOut[0].nModel != m_nThresholdModel) {
			nDrawT = m_pRecogOut[0].nEndT + 1;
			nRecOut = 1;
		} else if (pStroke->m_Norm.nPt >= MAX_RESAMPLE_DATA) {
			nDrawT = MAX_FRAME_INPUT;
			nRecOut = 0;
		}
	}

	m_pUnited->DrawTraceViterbiScore( pMetaDC, pView, m_pModel, m_nModel, 
				m_pRecogOut, nRecOut, nDrawT );
	FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, TRUE, MO_GRAPH );

	pMetaDC = NULL;
	if ( m_Flags.bMakeMetafile ) {
		pStroke->CalcSpotBoundingRect( pView, &rc );
		pMetaDC = InitMetaFileDC( pView, &rc );
	}
	pStroke->DrawSpottingResult( pMetaDC, pView, m_pRecogOut, nRecOut, nDrawT,
					m_pModel, m_nThresholdModel );
	FinishMetaRecording( pMetaDC, -1, TRUE, NO_WAIT, &rc, TRUE, MO_STROKE );
	m_nMetafileCount++;

	if (pStroke->m_Norm.pnFrameID)
		SaveRecogOutData( nRecOut, m_pRecogOut, pStroke );

	ResetSpottingRecognition( pStroke, bResetAll );

	if ( !WaitMoment(rWait, FALSE) )
		return FALSE;

	return TRUE;
}
   
void CGestureDoc::DisplayCurrentProcess(CStroke* pStroke, SHORT nDrawT)
{
	SHORT	nRecOut = m_nRecogOut;
	CView*	pView = theApp.m_pGestView;

	if (m_pRecogOut[m_nRecogOut].nModel != m_nThresholdModel)
		nRecOut = m_nRecogOut + 1;

	ClearDisplay( pView );
	m_pUnited->DrawTraceViterbiScore( NULL, pView, m_pModel, m_nModel, 
				m_pRecogOut, nRecOut, nDrawT );
	pStroke->DrawSpottingResult( NULL, pView, m_pRecogOut, nRecOut, nDrawT,
				m_pModel, m_nThresholdModel );
}

void CGestureDoc::RemoveUselessInput( CStroke* pStroke, SHORT nLength )
{
	if ( m_pRecogOut[0].nModel != MODEL_THRESHOLD )
		return;

	// Dummy model for resetting
	m_pRecogOut[0].nModel = MODEL_LAST;
	m_pRecogOut[0].nStartT = m_pRecogOut[0].nFirstEndT 
						   = m_pRecogOut[0].nEndT = nLength; 
	m_pRecogOut[1].nModel = MODEL_THRESHOLD;
	m_pRecogOut[1].nFirstEndT = -1;

	ResetSpottingRecognition( pStroke, FALSE );

	m_pRecogOut[0].nModel = MODEL_THRESHOLD;
	m_pRecogOut[0].nFirstEndT = -1;
	m_nRecogOut = 0;
}

//#define DRAW_PROCESS

BOOL CGestureDoc::SpottingRecognition( CStroke* pStroke, BOOL bLast, REAL rWait )
{
	if (pStroke->m_Norm.nPt < MIN_POINTS)
		return TRUE;

	// m_nFVec == 0 : first time to processing
	// m_nFVec >= pStroke->m_Norm.nPt : Error condition
	if ( m_nFVec == 0 || m_nFVec >= pStroke->m_Norm.nPt ) {
		m_nProcessedTime = 0;
		m_nFVec = 1;
	}

	// Recalculate previous feature vector for correctness
	//	==> confirmation is required
	for(int i=m_nFVec-1; i<pStroke->m_Norm.nPt-1; i++) {
		pStroke->PointToVector( m_pFVec, i, TRUE );
		m_pnOutput[i] = m_pMGVQ->Lookup( *m_pFVec, i );
		m_nOutput = i + 1;
	}
	m_nFVec = m_nOutput = pStroke->m_Norm.nPt - 1;

	if ( m_Flags.bShowRecognition ) {
		memcpy( m_pTmpStroke->m_Buf.pPt, pStroke->m_Norm.pPt,
				sizeof(POINT) * pStroke->m_Norm.nPt );
		m_pTmpStroke->m_Buf.nPt = pStroke->m_Norm.nPt;
	}

	TestUnitedHMM( NO_WAIT, bLast, m_pTmpStroke );

	if (bLast || m_nRecogOut > 1)
		DisplaySpottedGesture(pStroke, bLast, rWait);
#ifdef DRAW_PROCESS
	else if ( !m_Flags.bMakeMetafile )
		DisplayCurrentProcess(pStroke, m_nFVec - 1);
#endif
	if (m_pRecogOut[0].nModel == m_nThresholdModel) {
		if (pStroke->m_Norm.nPt >= MAX_GESTURE_INPUT) {
			DisplayCurrentProcess(pStroke, 
					pStroke->m_Norm.nPt - MAX_GESTURE_LENGTH + 1);
			RemoveUselessInput( pStroke, 
					pStroke->m_Norm.nPt - MAX_GESTURE_LENGTH );
		}
	} else if (m_pRecogOut[1].nModel == m_nThresholdModel) {
		if (pStroke->m_Norm.nPt - m_pRecogOut[0].nStartT > MAX_GESTURE_LENGTH)
			DisplaySpottedGesture(pStroke, bLast, rWait);
	}

	return TRUE;
}

void CGestureDoc::EndSpottingRecognition(CStroke* pStroke)
{
	pStroke->FreeFrameID();
	if (m_pTmpStroke)
		delete m_pTmpStroke;
	m_pTmpStroke = NULL;

	m_nProcessedTime = 0;

	if ( m_pVScore )
		delete m_pVScore;
	m_pVScore = NULL;

	if ( m_pFVec )
		delete[] m_pFVec;
	m_pFVec = NULL;
	m_nFVec = 0;

	if ( m_pnOutput )
		delete[] m_pnOutput;
	m_pnOutput = NULL;
	m_nOutput = 0;

	m_pUnited->DestroyTraceViterbiScore();
	m_Flags.bFirstTraceViterbi = TRUE;
}
