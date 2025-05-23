// #include "pch.h"
#include <SDKDDKVer.h>       // by hklee

#include "stdafx.h"
#include "Spot.h"

#include <math.h>
#include "utils.h"
#include "Cstroke.h"
#include "CVScore.h"
#include "Cdhmm.h"
#include "GestureDoc.h"


// Discrete HMM
const char cszMagicStr[MAGIC_SIZE] = "KAIST AI Lab Hmm 1.0\n\004";
const SHORT  DEF_STATE  = 6;
const SHORT  DEF_SYMBOL = 512;

#define VERY_SMALL		(1.0e-30)

// Reference :
// [1] Deller, Proakis, Hansen : "Discrete-Time Processing of Speech Signals"
// [2] Rabiner, Juang :		 "Fundamentals of Speech Recongition"

// Constructor / destructor

CDHmm::CDHmm( const char* pszFilename )
{
	m_nFinal = 0;
	m_pnBest = NULL;
	m_pnTime = NULL;
	m_pprFinal = NULL;

	m_bLoged = FALSE;
	m_bThreshold = FALSE;
	m_nState = m_nTime = m_nSymbol = 0;
	m_pnSymbol = NULL;
	m_prPi   = m_prScale = NULL; 
	m_pprA   = m_pprB = m_pprAlpha = m_pprBeta = m_pprGamma = NULL;

	// for multiple obs. seq.
	m_nObs   = 0;
	m_rObsP  = 0.0;
	m_prGPi  = m_prA_d  = m_prB_d = NULL;
	m_pprA_n = m_pprB_n = NULL;
  
	if( pszFilename )
		LoadModel( pszFilename );
}

CDHmm::~CDHmm()
{
	Delete();
	DeleteTrainVar();
	DeleteSumVar();
	DestroyTraceViterbiScore();
}

////////////////////////////////////////////////////////////////////////////
// allocation / deallocation space

void CDHmm::New()
{
	m_prPi     = (REAL*)  MallocREAL( m_nState );
	m_pprA     = (REAL**) MallocREAL( m_nState, m_nState );
	m_pprB     = (REAL**) MallocREAL( m_nState, m_nSymbol );
}

void CDHmm::NewTrainVar()
{
	m_prScale  = (REAL*)  MallocREAL( m_nTime );
	m_pprAlpha = (REAL**) MallocREAL( m_nTime, m_nState );
	m_pprBeta  = (REAL**) MallocREAL( m_nTime, m_nState );
	m_pprGamma = (REAL**) MallocREAL( m_nTime, m_nState );
}

// used for finding the probability with multiple obs. seq.
void CDHmm::NewSumVar()
{
	m_nObs     = 0;
	m_rObsP    = 0.0;
	m_prGPi    = (REAL*)  MallocREAL( m_nState );
	m_prA_d    = (REAL*)  MallocREAL( m_nState );
	m_pprA_n   = (REAL**) MallocREAL( m_nState, m_nState );
	m_prB_d    = (REAL*)  MallocREAL( m_nState );
	m_pprB_n   = (REAL**) MallocREAL( m_nState, m_nSymbol );
}

void CDHmm::Delete()
{
	Mfree( m_prPi );
	Mfree( (void**) m_pprA, m_nState );
	Mfree( (void**) m_pprB, m_nState );
}

void CDHmm::DeleteTrainVar()
{
	Mfree( m_prScale );
	Mfree( (void**) m_pprAlpha, m_nTime );
	Mfree( (void**) m_pprBeta,  m_nTime );
	Mfree( (void**) m_pprGamma, m_nTime );
}

void CDHmm::DeleteSumVar()
{
	Mfree( m_prGPi );
	Mfree( m_prA_d );
	Mfree( (void**) m_pprA_n, m_nState );
	Mfree( m_prB_d );
	Mfree( (void**) m_pprB_n, m_nState );
}

void CDHmm::ClearVariable()
{
	int	i;
	int	nsize = sizeof(REAL) * m_nState;
  
	if( m_prPi )
		memset( m_prPi, 0, nsize );

	if( m_pprA ) {
		for( i = 0 ; i < m_nState ; i ++ )
			memset( m_pprA[i], 0, nsize );
	}

	if( m_pprB ) {
		int ksize = sizeof(REAL) * m_nSymbol;
		for( i = 0 ; i < m_nState ; i ++ )
			memset( m_pprB[i], 0, ksize );
	}
}

void CDHmm::ClearTrainVar()
{
	int	i;
	int	nsize = sizeof(REAL) * m_nState;

	if( m_prScale )
		memset( m_prScale, 0, sizeof(REAL)*m_nTime );

	if( m_pprAlpha ) {
		for( i = 0 ; i < m_nTime ; i ++ )
			memset( m_pprAlpha[i], 0, nsize );
	}

	if( m_pprBeta ) {
		for( i = 0 ; i < m_nTime ; i ++ )
			memset( m_pprBeta[i], 0, nsize );
	}

	if( m_pprGamma ) {
		for( i = 0 ; i < m_nTime ; i ++ )
			memset( m_pprGamma[i], 0, nsize );
	}
}

void CDHmm::ClearSumVar()
{
	int i;
	int	nsize = sizeof(REAL) * m_nState;

	if ( m_prGPi )
		memset( m_prGPi, 0, nsize );

	if ( m_prA_d )
		memset( m_prA_d, 0, nsize );

	if( m_pprA_n ) {
		for( i = 0 ; i < m_nState ; i ++ )
			memset( m_pprA_n[i], 0, nsize );
	}

	if ( m_prB_d )
		memset( m_prB_d, 0, nsize );

	if( m_pprB_n ) {
		int ksize = sizeof(REAL) * m_nSymbol;
		for( i = 0 ; i < m_nState ; i ++ )
			memset( m_pprB_n[i], 0, ksize );
	}
}

void CDHmm::New( const SHORT nState, const SHORT nSymbol )
{
	if( nState == m_nState && nSymbol == m_nSymbol ) {
		ClearVariable();
		return;
	} else
		Delete();
  
	m_nState  = nState;
	m_nSymbol = nSymbol;
	New();
}

void CDHmm::NewTrainVar( const SHORT nTime )
{
	if( nTime == m_nTime ) {
		ClearTrainVar();
		return;
	} else
		DeleteTrainVar();

	m_nTime = nTime;
	NewTrainVar();
}

////////////////////////////////////////////////////////////////////////////

void CDHmm::Initialize()
{
	InitialStateProb();
	InitialTransitionProb();
	InitialOutputProb();
}

// L-R model

void CDHmm::InitialStateProb()
{
	int	i;

	m_prPi[0] = 1.0;
	for( i = 1 ; i < m_nState ; i ++ )
		m_prPi[i] = 0.0;
}

void CDHmm::InitialTransitionProb()
{
	int	i, j, nsize = sizeof(REAL)*m_nState;
	REAL	sum;
  
	for( i = 0 ; i < m_nState ; i ++ ) {
		memset( m_pprA[i], 0, nsize );

		// j == i ,    recurrent
		// j == i + 1, neighbor
		// j == i + 2, next neighbor
		for( j = i ; j < m_nState && j < i+3 ; j ++ ) {
			// Initial/final state has no self-loop
			if ( (i == 0 && j == 0) || (i >= m_nState-1) )
				continue;
			m_pprA[i][j] = 1.0;
		}

		for( sum = 0.0, j = i ; j < m_nState ; j ++ )
			sum += m_pprA[i][j];

		// normalize
		for( j = i ; j < m_nState ; j ++ ) {
			if( m_pprA[i][j] ) m_pprA[i][j] /= sum;
		}
	}
}

void CDHmm::InitialOutputProb()
{
	int	i, k;
	REAL	op = 1. / m_nSymbol;	// output probability
	for( i = 0 ; i < m_nState ; i ++ )
		for( k = 0 ; k < m_nSymbol ; k ++ )
			m_pprB[i][k] = op;
}

void CDHmm::RescaleAlpha( const SHORT nTime )
{
	int i;

	m_prScale[nTime] = 0.0;
	// calculate the scaling coefficents
	for( i = 0 ; i < m_nState ; i ++ )
		m_prScale[nTime] += m_pprAlpha[nTime][i];

	// rescaling the alphA
	for( i = 0 ; i < m_nState ; i ++ )
		m_pprAlpha[nTime][i] /= m_prScale[nTime];
}

REAL CDHmm::Forward()
{
	int	i, j, t, t1;
	REAL	sum;

	// initialize
	for( j = 0; j < m_nState ; j ++ )
		m_pprAlpha[0][j] = m_prPi[j] * m_pprB[j][m_pnSymbol[0]];
	RescaleAlpha( 0 );

	// induction
	for( t = 1 ; t < m_nTime ; t ++ ) {
		for( j = 0 ; j < m_nState ; j ++ ) {
			t1 = t - 1;
			for( sum = 0.0, i = 0 ; i < m_nState ; i ++ )
				sum += m_pprAlpha[t1][i] * m_pprA[i][j];
			m_pprAlpha[t][j] = sum * m_pprB[j][m_pnSymbol[t]];
		}
		RescaleAlpha( t );
	}
  
	// termination
	// return likelihood P(y|m) using (12.40) [1]
	return( m_pprAlpha[m_nTime-1][m_nState-1] );
}

void CDHmm::RescaleBeta( const SHORT nTime )
{
	int i;

	// m_prScale[nTime] is already calculated in RescaleAlpha().
	// rescaling the alpha
	for( i = 0 ; i < m_nState ; i ++ )
		m_pprBeta[nTime][i] /= m_prScale[nTime];
}

void CDHmm::Backward()
{
	int	i, j, t = m_nTime - 1;

	// initialize
	for( i = 0 ; i < m_nState ; i ++ )
		m_pprBeta[t][i] = 0.0;
	m_pprBeta[t][m_nState-1] = 1.0;	// legal end state
	RescaleBeta( t );

	// induction
	for( --t ; t >= 0 ; t -- ) {
		for( i = 0 ; i < m_nState ; i ++ ) {
			m_pprBeta[t][i] = 0.0;
			for( j = 0 ; j < m_nState ; j ++ )
				m_pprBeta[t][i] +=
					( m_pprA[i][j] * m_pprB[j][m_pnSymbol[t+1]] * m_pprBeta[t+1][j] );
		}
		RescaleBeta( t );
	}
}

// we must multifly the m_prScale[t] to the (alpha * beta)/alpha_T,
// because the normalized alpha and beta are overlapped,
// so we must conpensate for the multiple normalized variables.
void CDHmm::CalculateGamma()
{
	int	j, t, tm = m_nTime-1, sm = m_nState-1;

	for( j = 0 ; j < m_nState ; j ++ ) {
		for( t = 0 ; t < m_nTime ; t ++ )
			m_pprGamma[t][j] =  m_prScale[t] * 
				( m_pprAlpha[t][j] * m_pprBeta[t][j] ) / m_pprAlpha[tm][sm];
	}
}

REAL CDHmm::ExpectedTransition( const SHORT i )
{
	int	t, tm = m_nTime - 1;
	REAL	sum = 0.0;

	for( t = 0 ; t < tm ; t ++ )
		sum += m_pprGamma[t][i];
	return( sum );
}

REAL CDHmm::ExpectedTransition( const SHORT i, const SHORT j )
{
	int	t, tm = m_nTime - 1;
	REAL	sum = 0.0;

	for( t = 0 ; t < tm ; t ++ ) {
		sum += ( m_pprAlpha[t][i] * m_pprA[i][j] *
			m_pprB[j][m_pnSymbol[t+1]] * m_pprBeta[t+1][j] );
	}
	sum /= m_pprAlpha[tm][m_nState-1];
	return( sum );
}

REAL CDHmm::ExpectedTimes( const SHORT nState, const SHORT Obs )
{
	int	t;
	REAL	sum = 0.0;

	for( t = 0 ; t < m_nTime ; t ++ ) {
		if( m_pnSymbol[t] == Obs )
			sum += m_pprGamma[t][nState];
	}
	return( sum );
}

// when we use the negative logarithm of Maximum Likelihood,
// the problem is turned into a minimal-cost path search.

// because we scale the alpha and beta, we represent the P(y|m) as
// Mult( m_prScale[t], 0  <= t <= T-1 ).
// See page 715~718p [1]
REAL CDHmm::LogLikelihood()
{
	int	t;
	REAL	lll = 0.0;

	for( t = 0 ; t < m_nTime ; t ++ )
		lll += log( m_prScale[t] );
	return( lll );
}

// calculate the nominator/denominator of A_ij, B_jk
void CDHmm::SumEstimation()
{
	REAL	rExpTrans;
	int		i, j;

	Forward();
	Backward();
	CalculateGamma();

	for( i = 0 ; i < m_nState ; i ++ ) {
		m_prGPi[i] += m_prPi[i];

		rExpTrans = ExpectedTransition( i );
		m_prA_d[i] += rExpTrans;
		for( j = 0 ; j < m_nState ; j ++ )
			m_pprA_n[i][j] += ExpectedTransition( i, j );
    
		m_prB_d[i] += ( rExpTrans + m_pprGamma[m_nTime-1][i] );
		for( j = 0 ; j < m_nSymbol ; j ++ )
			m_pprB_n[i][j] += ExpectedTimes( i, j );
	}
	m_rObsP += LogLikelihood();
	m_nObs  ++;	// increment opservation count
}

// Baum-Welch resestimation with multiple observation sequence.
REAL CDHmm::Reestimation()
{
	int	i, j;
	REAL	rObsPr = m_rObsP / m_nObs;

	for( i = 0 ; i < m_nState ; i ++ ) {
		m_prPi[i] = m_prGPi[i] / m_nObs;
		if( m_prGPi[i] != 0.0 && m_prPi[i] < VERY_SMALL )
			m_prPi[i] = VERY_SMALL;

		for( j = 0 ; j < m_nState ; j ++ ) {
			if( m_pprA[i][j] ) {
				// m_pprA[i][j] must check whether the A[i][j] isn't zero,
				// because the A[i][j] determined the HMM Model.

				m_pprA[i][j] = m_pprA_n[i][j] / m_prA_d[i];
				if( m_pprA[i][j] < VERY_SMALL )
					m_pprA[i][j] = VERY_SMALL;
			}

			// clear summation var
			m_pprA_n[i][j] = 0.0;
		}
    
		for( j = 0 ; j < m_nSymbol ; j ++ ) {
			m_pprB[i][j] = m_pprB_n[i][j] / m_prB_d[i];
			if( m_pprB[i][j] < VERY_SMALL )
				m_pprB[i][j] = VERY_SMALL;

			// clear summation var
			m_pprB_n[i][j] = 0.0;
		}

		// clear summation var
		m_prGPi[i] = m_prA_d[i] = m_prB_d[i] = 0.0;
	}
	// clear total observation probability and observation counter.
	m_rObsP = 0.0;
	m_nObs  = 0;

	return( rObsPr );
}

////////////////////////////////////////////////////////////////////////////

BOOL CDHmm::LoadModel( const char* pszFilename=NULL )
{
	if( pszFilename )
		return( Read( pszFilename ) );
	else {
		NewModel( DEF_STATE, DEF_SYMBOL );
		return( TRUE );
	}
}

BOOL CDHmm::NewModel( const SHORT nState, const SHORT nSymbol )
{
	New( nState, nSymbol );
	Initialize();
	return( TRUE );
}

void CDHmm::SetABCounts( SHORT* pnOutput, const SHORT nOutput )
{
	if( !m_prGPi || !m_prA_d || !m_pprA_n || !m_prB_d || !m_pprB_n ) {
		NewSumVar();
		ClearSumVar();
	}

	m_pnSymbol = pnOutput;
	NewTrainVar( nOutput );
	SumEstimation();
}

REAL CDHmm::Train( SHORT** ppnOutput, const SHORT nOutput, SHORT* pnOutput )
{
	int	n = 0;
	static REAL prevLL = 0.0;
	REAL	newLL;

	NewSumVar();

	while(1) {
		newLL = 0.0;
		for( n = 0 ; n < nOutput ; n ++ ) {
			m_pnSymbol = ppnOutput[n];
			NewTrainVar( pnOutput[n] );
			SumEstimation();
		}
		newLL = Reestimation();
		if( prevLL != 0.0 && fabs( newLL - prevLL ) < FLOATRESN )
			break;
		prevLL = newLL;
	}

	DeleteSumVar();
	return( newLL );
}

//////////////////////////////////////////////////////////////////////
//
// Viterbi algorithm taking negative logarithms
//
// Ref : [1] 692 - 697pp, [2] 339 - 340pp

// negative log probability

REAL CDHmm::NLog( const REAL rValue )
{
	REAL	r;
	if( rValue == 0.0 )
		r = -log( cMinREAL );
	else
		r = -log( rValue );
	return( r );
}

void CDHmm::NegLogProbability()
{
	int	i, j;
	for( i = 0 ; i < m_nState ; i ++ ) {
		m_prPi[i] = NLog( m_prPi[i] );

		for( j = 0 ; j < m_nState ; j ++ )
			m_pprA[i][j] = NLog( m_pprA[i][j] );

		for( j = 0 ; j < m_nSymbol ; j ++ )
			m_pprB[i][j] = NLog( m_pprB[i][j] );
	}
	m_bLoged = TRUE;
}

void CDHmm::InitViterbi( REAL* pvd, SHORT** ppBestPath )
{
	int	i;

	// initialize
	for( i = 0 ; i < m_nState ; i ++ ) {
		pvd[i] =  m_prPi[i] + m_pprB[i][m_pnSymbol[0]];
		ppBestPath[0][i] = 0;
	}
}

void CDHmm::InitViterbi( CVScore &vscore )
{
	int	i;

	// initialize
	for( i = 0 ; i < m_nState ; i ++ ) {
		vscore.m_pprScore[0][i] = m_prPi[i] + m_pprB[i][m_pnSymbol[0]];
		vscore.m_ppnState[0][i] = 0;
	}
}

REAL CDHmm::RecursionViterbi( REAL* pVd, SHORT** ppBestPath, SHORT* pnLast, 
							 SHORT* pnFinal, SHORT nFinal )
{
	int		i, j, t;
	SHORT	mi;
	REAL*	plvd;
	REAL	mscore;

	if( !( plvd = (REAL *)MallocREAL( m_nState )) )
	{	
//		Error( "(CDHmm::RecusrionViterbi) MallocREAL(%d) failed", m_nState );
		;
	}

	int	rsize = sizeof(REAL) * m_nState;
	memcpy( plvd, pVd, rsize );
  
	// recursion
	for( t = 1 ; t < m_nTime ; t ++ ) {
		for( j = 0 ; j < m_nState ; j ++ ) {

			// find minimum distance because we already taked negative logarithms
			// on initial state prob, trans prob, and output prob.
			for( mscore = cMaxREAL, i = 0 ; i < m_nState ; i ++ ) {
				if( mscore > plvd[i] + m_pprA[i][j] ) {
					mscore = plvd[i] + m_pprA[i][j];
					mi = i;
				}
			}
	
			// assign the minimum viterbi distance
			pVd[j] = mscore + m_pprB[j][m_pnSymbol[t]];

			// record the best path
			ppBestPath[t][j] = mi;
		}		// copy the current distance for reducing memory
		memcpy( plvd, pVd, rsize );
	}

	if (*pnLast < 0) {
		if (pnFinal) {
			for( mscore = cMaxREAL, i = 0 ; i < nFinal ; i ++ ) {
				if( mscore > plvd[ pnFinal[i] ] ) {
					mscore = plvd[ pnFinal[i] ];
					*pnLast = pnFinal[i];
				}
			}
		} else {
			for( mscore = cMaxREAL, i = 0 ; i < m_nState ; i ++ ) {
				if( mscore > plvd[i] ) {
					mscore = plvd[i];
					*pnLast = i;
				}
			}
		}
	}
	mscore = plvd[*pnLast];
  
	Mfree( plvd );
	return( mscore );
}

REAL CDHmm::RecursionViterbi( CVScore &vscore, SHORT nStartT, SHORT* pnLast )
{
	int		i, j, t;
	SHORT	mi;
	REAL	mscore;

	// recursion
	for( t = nStartT ; t < m_nTime ; t ++ ) {
		for( j = 0 ; j < m_nState ; j ++ ) {
			// find minimum distance because we already taked negative logarithms
			// on initial state prob, trans prob, and output prob.
			for( mscore = cMaxREAL, i = 0 ; i < m_nState ; i ++ ) {
				if( mscore > vscore.m_pprScore[t-1][i] + m_pprA[i][j] ) {
					mscore = vscore.m_pprScore[t-1][i] + m_pprA[i][j];
					mi = i;
				}
			}
			
			// assign the minimum viterbi distance
			vscore.m_pprScore[t][j] = mscore + m_pprB[j][m_pnSymbol[t]];
			// record the best path
			vscore.m_ppnState[t][j] = mi;
		}
	}

	// find the minimum negative log likelihood
	if (*pnLast < 0) {
		mscore = cMaxREAL;
		if (vscore.m_nKindOfModel == UNITED_MODEL) {
			for( i = 0 ; i < vscore.m_nSubModel ; i ++ ) {
				if (vscore.m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
					for( j=vscore.m_pSubVScore[i].m_nStartState;
						 j<=vscore.m_pSubVScore[i].m_nFinalState; j++) {
						if( mscore > vscore.m_pprScore[m_nTime-1][j] ) {
							mscore = vscore.m_pprScore[m_nTime-1][j];
							*pnLast = j;
						}
					}
				} else {
					SHORT nTmp = vscore.m_pSubVScore[i].m_nFinalState;

					if( mscore > vscore.m_pprScore[m_nTime-1][nTmp] ) {
						mscore = vscore.m_pprScore[m_nTime-1][nTmp];
						*pnLast = nTmp;
					}
				}
			}
		} else {
			for( i = 0 ; i < m_nState ; i ++ ) {
				if( mscore > vscore.m_pprScore[m_nTime-1][i] ) {
					mscore = vscore.m_pprScore[m_nTime-1][i];
					*pnLast = i;
				}
			}
		}
	}
	mscore = vscore.m_pprScore[m_nTime-1][*pnLast]; 
	vscore.m_nFinalState = *pnLast;
	vscore.m_rPenalty = 0.0;
	vscore.m_rLikelihood = mscore + vscore.m_rPenalty;
  
	return( mscore );
}
  
void CDHmm::BacktrackViterbi( const SHORT final, SHORT* pnPath, SHORT** ppBestPath )
{
	int	t = m_nTime-1;
	pnPath[t] = final;
	for( ; t > 0 ; t -- )
		pnPath[t-1] = ppBestPath[t][pnPath[t]];
}
  
void CDHmm::BacktrackViterbi( CVScore &vscore, SHORT nFinal )
{
	int	t = m_nTime-1;

	vscore.m_pnPath[t] = nFinal;
	for( ; t > 0 ; t -- )
		vscore.m_pnPath[t-1] = vscore.m_ppnState[t][vscore.m_pnPath[t]];
}

REAL CDHmm::Viterbi( SHORT* pnOutput, const SHORT nTime, SHORT* pnPath )
{
	SHORT**	ppBestPath;
	REAL*	pvd, lll;
	SHORT	nLast;

	if( !m_bLoged )
		NegLogProbability();

	ppBestPath = (SHORT**) MallocSHORT( nTime, m_nState );
	pvd  = (REAL *)MallocREAL( m_nState );
	if( !ppBestPath || !pvd ) {
		Warning( "(CDHmm::Viterbi) memory allocation failed" );
		if( ppBestPath ) Mfree( (void**) ppBestPath, nTime );
		if( pvd ) Mfree( (void *)pvd );
		return( 0.0 );
	}

	m_pnSymbol = pnOutput;
	m_nTime    = nTime;

	if (m_bThreshold)
		nLast = -1;
	else
		nLast = m_nState - 1;

	// Initialize
	InitViterbi( pvd, ppBestPath );

	// Recursion & find the minimum Negative Log Likelihood
	lll = RecursionViterbi( pvd, ppBestPath, &nLast );

	// Backtracking
	if( pnPath )
		BacktrackViterbi( nLast, pnPath, ppBestPath );

	Mfree( (void**) ppBestPath, m_nTime );
	Mfree( (void *) pvd );

	return( lll );
}

REAL CDHmm::Viterbi( SHORT* pnOutput, const SHORT nTime, CVScore &vscore )
{
	REAL	lll;
	SHORT	nLast;

	if( !m_bLoged )
		NegLogProbability();

	m_pnSymbol = pnOutput;
	m_nTime    = nTime;

	if (m_bThreshold)
		nLast = -1;
	else
		nLast = m_nState - 1;

	// Initialize
	InitViterbi( vscore );

	// Recursion & find the minimum Negative Log Likelihood
	lll = RecursionViterbi( vscore, 1, &nLast );

	// Backtracking
	BacktrackViterbi( vscore, nLast );

	return( lll );
}

void CDHmm::InitTraceViterbiScore( SHORT nModel )
{
	m_nFinal = 0;
	m_rMaxFinal = 0.0;

	if ( m_pnBest )
		Mfree( (void*)m_pnBest );
	m_pnBest = MallocSHORT( (int)MAX_OBSERVATION );

	if ( m_pnTime )
		Mfree( (void*)m_pnTime );
	m_pnTime = MallocSHORT( (int)MAX_OBSERVATION );

	if ( m_pprFinal )
		Mfree( (void**)m_pprFinal, MAX_OBSERVATION );
	m_pprFinal = (REAL**)MallocREAL( MAX_OBSERVATION, nModel );
}

void CDHmm::ResetTraceViterbiScore( SHORT nModel, SHORT* pnResetT )
{
	int i;

	m_rMaxFinal = 0.0;
	if (m_nFinal < 1 || *pnResetT >= m_pnTime[m_nFinal-1]) {
		m_nFinal = 0;
		return;
	}

	SHORT	nStartT = 0;
	for(int i=0; i<m_nFinal; i++) {
		if (m_pnTime[i] == *pnResetT) {
			nStartT = i;
			break;
		} else if (i > 0 && m_pnTime[i] > *pnResetT) {
			*pnResetT = m_pnTime[i-1];
			nStartT = i - 1;
			break;
		}
	}

	REAL	rSub = m_pprFinal[nStartT][0];
	for(i=1; i<nModel; i++)
		rSub = min( rSub, m_pprFinal[nStartT][i] );

	for(i=nStartT; i<m_nFinal; i++) {
		for(int j=0; j<nModel; j++) {
			m_pprFinal[i-nStartT][j] = m_pprFinal[i][j] - rSub;
			m_rMaxFinal = max( m_rMaxFinal, m_pprFinal[i-nStartT][j] );
		}
		m_pnBest[i-nStartT] = m_pnBest[i];
		m_pnTime[i-nStartT] = m_pnTime[i] - *pnResetT;
	}
	m_nFinal -= nStartT;
}


void CDHmm::TraceViterbiScore( CVScore& vscore )
{
	int		i;
	REAL	mscore;

	if ( m_nFinal >= MAX_OBSERVATION ) {
		for(i=0; i<MAX_OBSERVATION-1; i++) {
			for(int j=0; j<vscore.m_nSubModel; j++)
				m_pprFinal[i][j] = m_pprFinal[i+1][j];
			m_pnBest[i] = m_pnBest[i+1];
			m_pnTime[i] = m_pnTime[i+1];
		}
		m_nFinal--;
	}

	for(mscore = cMaxREAL, i=0; i<vscore.m_nSubModel; i++) {
		if (vscore.m_pSubVScore[i].m_nKindOfModel == THRESHOLD_MODEL) {
			m_pprFinal[m_nFinal][i] = cMaxREAL;
			vscore.GetBestThresholdState( i, &m_pprFinal[m_nFinal][i] );
		} else {
			m_pprFinal[m_nFinal][i] 
				= vscore.m_pprScore [m_nTime-1]
								[vscore.m_pSubVScore[i].m_nFinalState]; 
		}
		m_rMaxFinal = max( m_rMaxFinal, m_pprFinal[m_nFinal][i] );

		if (mscore > m_pprFinal[m_nFinal][i]) {
			m_pnBest[m_nFinal] = i;
			mscore = m_pprFinal[m_nFinal][i];
		}
	}
	m_pnTime[m_nFinal] = m_nTime; // -1;

	m_nFinal++;
}

void CDHmm::DestroyTraceViterbiScore()
{
	m_nFinal = 0;

	if ( m_pnBest )
		Mfree( (void*)m_pnBest );
	m_pnBest = NULL;

	if ( m_pnTime )
		Mfree( (void*)m_pnTime );
	m_pnTime = NULL;

	if ( m_pprFinal )
		Mfree( (void**)m_pprFinal, MAX_OBSERVATION );
	m_pprFinal = NULL;
}

#define META_X_AXIS		300
#define META_Y_AXIS		200
#define DISP_X_AXIS		600	//300
#define DISP_Y_AXIS		400	//200

static SHORT	X_AXIS, Y_AXIS;
	
#define MIN_DIST_Y		15
#define TOP_Y			30
#define GAP_XY			150
#define	TEXT_GAP		6
#define BAR_SIZE		3
#define BIG_BAR_SIZE	5
#define PEN_WIDTH		1

COLORREF rgbGraph[MODEL_THRESHOLD+1] 
				= { MY_RED, MY_GREEN, MY_BLUE, MY_PINK, MY_SKYBLUE, MY_GRAYRED,
					MY_PURPLE, MY_DARKGREEN, MY_LIGHTPURPLE, MY_ORANGE, MY_GRAY };

void DrawGraphLabel( CView* pView, CDC* pDC, SHORT nX, SHORT nY, REAL rLabel )
{
	POINT	ptLoc;
	char	szTmp[100];

	sprintf_s( szTmp, "-%.1f", rLabel );

	if (strlen(szTmp) > 8)
		return;

	ptLoc.x = nX;
	ptLoc.y = nY + TEXT_GAP;
	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LRCENTER | ALIGN_TOP, FALSE, FALSE, FSIZE_NORMAL, szTmp );
}

void CDHmm::DrawOneGraph( CMetaFileDC* pMetaDC, CView* pView, 
			SHORT nLeft, SHORT nTop, SHORT nCurModel, char* pszName, 
			RECOUT* pRecogOut, SHORT nRecogOut, SHORT nDrawT )
{
	POINT	ptLoc;
	SHORT	nX, nY, nStep;
	CPen	*oldPen, *newPen;
	COLORREF rgbOld;
	CDC*	pDC;
	int     i;
	
	if ( pMetaDC )
		pDC = pMetaDC;
	else
		pDC = pView->GetDC();

	rgbOld = pDC->SetTextColor( rgbGraph[nCurModel] );
	newPen = new CPen( PS_SOLID, PEN_WIDTH, rgbGraph[nCurModel] );
	oldPen = (CPen *)pDC->SelectObject( newPen );

	if (nDrawT <= 0)
		nStep = X_AXIS / m_pnTime[m_nFinal-1];
	else
		nStep = X_AXIS / min(nDrawT, m_pnTime[m_nFinal-1]);

	nX = nLeft;
	nY = nTop + (SHORT)(Y_AXIS * m_pprFinal[0][nCurModel] / m_rMaxFinal);
	pDC->MoveTo( nLeft, nTop );

	nY = nTop + (SHORT)(Y_AXIS * m_pprFinal[0][nCurModel] / m_rMaxFinal);
	if (m_pnTime[0] == 0) {
		pDC->MoveTo( nX, nY );
	} else {
		for(int k=0; k<m_pnTime[0]; k++)
			nX += nStep;
		pDC->LineTo( nX, nY );
	}

	if ( !pMetaDC )
		DrawGraphLabel( pView, pDC, nX, nY, m_pprFinal[0][nCurModel] );

	for(int i=1; i<m_nFinal; i++) {
		if (nDrawT > 0 && m_pnTime[i] >= nDrawT)
			break;

		for(int k=0; k<m_pnTime[i] - m_pnTime[i-1]; k++)
			nX += nStep;
		nY = nTop + (SHORT)(Y_AXIS * m_pprFinal[i][nCurModel] / m_rMaxFinal);
		pDC->LineTo( nX, nY );

		if ( !pMetaDC ) {
			for(int j=0; j<nRecogOut; j++) {
				if ( pRecogOut[j].nModel == nCurModel ) {
					if ( pRecogOut[j].nStartT == i || 
						 pRecogOut[j].nFirstEndT == i || pRecogOut[j].nEndT == i )
						DrawGraphLabel( pView, pDC, nX, nY, m_pprFinal[i][nCurModel] );
				}
			}
		}

		if ( m_pnBest[i] != m_pnBest[i-1] ) {
			if ( !pMetaDC )
				DrawGraphLabel( pView, pDC, nX, nY, m_pprFinal[i][nCurModel] );

			if ( m_pnBest[i] == nCurModel ) {
				ptLoc.x = GAP_XY + (int)(m_pnTime[i] * nStep);
				if ( (m_pnTime[i] % 5) == 0 ) {
					pDC->MoveTo( ptLoc.x, TOP_Y - BIG_BAR_SIZE );
					pDC->LineTo( ptLoc.x, TOP_Y + BIG_BAR_SIZE );
				} else {
					pDC->MoveTo( ptLoc.x, TOP_Y - BAR_SIZE );
					pDC->LineTo( ptLoc.x, TOP_Y + BAR_SIZE );
				}
				pDC->MoveTo( nX, nY );
			}
		}
	}

	ptLoc.x = nX + TEXT_GAP;
	ptLoc.y = nY;

	SHORT	nNewY = -1;
	for(i=0; i<nCurModel; i++) {
		nY = nTop + (SHORT)(Y_AXIS * m_pprFinal[m_nFinal-1][i] / m_rMaxFinal);
		if ( abs(nY-ptLoc.y) < MIN_DIST_Y ) {
			nNewY = nY;
			break;
		}
	}

	if (nNewY >= 0) {
		if (nNewY > ptLoc.y)
			ptLoc.y = nNewY - MIN_DIST_Y;
		else
			ptLoc.y = nNewY + MIN_DIST_Y;
	}

	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LEFT | ALIGN_TBCENTER, FALSE, FALSE, FSIZE_NORMAL, pszName );
	pDC->SetTextColor( rgbOld );

	pDC->SelectObject(oldPen);
	delete newPen;

	if (!pMetaDC)
		pView->ReleaseDC( pDC );
}

void CDHmm::CalcTraceBoundingRect( CRect* prc )
{
	X_AXIS = META_X_AXIS;
	Y_AXIS = META_Y_AXIS;
	prc->SetRect(GAP_XY - 50, 0, GAP_XY + X_AXIS + 70, TOP_Y + Y_AXIS + 30 );
}

void DrawMyYTitle( CDC* pDC, CView* pView, POINT ptLoc )
{
	int		width;
	CSize	size1, size2, size3;

	size1 = GetMyStringExtent( pView, "Log P(X|", FALSE );
	width = size1.cx;
	size2 = GetMyStringExtent( pView, "l", TRUE );
	width += size2.cx;
	size3 = GetMyStringExtent( pView, ")", FALSE );
	width += size3.cx;

	ptLoc.x -= width / 2;

	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LEFT | ALIGN_TOP, FALSE, FALSE, FSIZE_NORMAL, "Log P(X|" );
	ptLoc.x += size1.cx;
	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LEFT | ALIGN_TOP, FALSE, TRUE, FSIZE_NORMAL, "l" );
	ptLoc.x += size2.cx;
	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LEFT | ALIGN_TOP, FALSE, FALSE, FSIZE_NORMAL, ")" );
}



//#define USE_GARBAGE_MODEL
#define DUMP_GRAPH_DATA	1

void CDHmm::DrawTraceViterbiScore( CMetaFileDC* pMetaDC, CView* pView, 
								   MODEL* pModel, SHORT nModel, 
								   RECOUT* pRecogOut, SHORT nRecogOut, SHORT nDrawT )
{
	char	szStr[10];
	REAL	rValue;
	POINT	ptLoc;
	SHORT	nX, nY, nStep;
	CDC*	pDC;
	CPen	*oldPen, *newPen;
	int     i;

	if (m_nFinal < 1)
		return;

	nX = GAP_XY;
	nY = TOP_Y;

	if (pMetaDC) {
		pDC = pMetaDC;
		X_AXIS = META_X_AXIS;
		Y_AXIS = META_Y_AXIS;
	} else {
		pDC = pView->GetDC();
		X_AXIS = DISP_X_AXIS;
		Y_AXIS = DISP_Y_AXIS;
	}

	newPen = new CPen( PS_SOLID, PEN_WIDTH, MY_BLACK );
	oldPen = (CPen *)pDC->SelectObject( newPen );

	pDC->MoveTo( nX, nY );
	pDC->LineTo( nX, nY + Y_AXIS );
	pDC->MoveTo( nX, nY );
	pDC->LineTo( nX + X_AXIS, nY );

	ptLoc.x = GAP_XY - TEXT_GAP;
	rValue = 0.0;
	for( int i=0; i<11; i++ ) {
		ptLoc.y = TOP_Y + (int)( Y_AXIS * rValue / m_rMaxFinal ); 
		pDC->MoveTo( GAP_XY - BAR_SIZE, ptLoc.y );
		pDC->LineTo( GAP_XY + BAR_SIZE, ptLoc.y );
		sprintf_s( szStr, "-%.1f", rValue );
		DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
				ALIGN_RIGHT | ALIGN_TBCENTER, FALSE, FALSE, FSIZE_NORMAL, szStr );
		rValue += 50.0;
		if (rValue > m_rMaxFinal)
			break;
	}

	ptLoc.x = GAP_XY;
	ptLoc.y = TOP_Y + Y_AXIS + TEXT_GAP;
	DrawMyYTitle( pDC, pView, ptLoc );

	if (nDrawT <= 0)
		nStep = X_AXIS / m_pnTime[m_nFinal-1];
	else
		nStep = X_AXIS / min(nDrawT, m_pnTime[m_nFinal-1]);

	ptLoc.x = GAP_XY - TEXT_GAP;
	ptLoc.y = TOP_Y - TEXT_GAP; 
	for( i=0; i<m_pnTime[m_nFinal-1]+1; i++ ) {
		if (nDrawT > 0 && i >= nDrawT)
			break;

		ptLoc.x = GAP_XY + (int)(i * nStep);
		if ( (i % 5) == 0 ) {
			pDC->MoveTo( ptLoc.x, TOP_Y - BIG_BAR_SIZE );
			pDC->LineTo( ptLoc.x, TOP_Y + BIG_BAR_SIZE );
			sprintf_s( szStr, "%d", i );
			DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
					ALIGN_LRCENTER | ALIGN_BOTTOM, FALSE, FALSE, FSIZE_NORMAL, szStr );
		} else {
			pDC->MoveTo( ptLoc.x, TOP_Y - BAR_SIZE );
			pDC->LineTo( ptLoc.x, TOP_Y + BAR_SIZE );
		}
	}

	ptLoc.x = GAP_XY + X_AXIS + TEXT_GAP;
	ptLoc.y = TOP_Y;
	DrawMyText( pDC, pView, ptLoc, TRANSPARENT, 
			ALIGN_LEFT | ALIGN_TBCENTER, FALSE, FALSE, FSIZE_NORMAL, "Time" );

	pDC->SelectObject( oldPen );
	delete newPen;

	if ( !pMetaDC )
		pView->ReleaseDC( pDC );

#ifdef DUMP_GRAPH_DATA
	FILE* fp = NULL;
	if ( pMetaDC )
		fp = fopen(MakeMyFileName("Meta\\", "Graph", "Txt"), "w");
#endif

	for( i=0; i<nModel; i++ ) {
		char *pszName = GetModelFullName( i );

		DrawOneGraph( pMetaDC, pView, nX, nY, i, pszName, 
				pRecogOut, nRecogOut, nDrawT );

#ifdef DUMP_GRAPH_DATA
		if ( fp ) {
			fprintf(fp, "%s", pszName);
			fprintf(fp, "\t%.2f", 0.0);		
			for(int j=0; j<m_nFinal; j++) {
				if (nDrawT > 0 && m_pnTime[j] >= nDrawT)
					break;
				fprintf(fp, "\t-%.2f", m_pprFinal[j][i]);		
			}
			fprintf(fp, "\n");
		}
#endif
	}

#ifdef DUMP_GRAPH_DATA
	if ( fp )
		fclose( fp );
#endif
}

REAL CDHmm::UnitedViterbi( SHORT* pnOutput, const SHORT nTime, SHORT* pnPath, 
						  SHORT* pnFinal, SHORT nFinal )
{
	SHORT**	ppBestPath;
	REAL*	pvd, lll;

	if( !m_bLoged )
		NegLogProbability();

	ppBestPath = (SHORT**) MallocSHORT( nTime, m_nState );
	pvd  = (REAL *)MallocREAL( m_nState );

	if( !ppBestPath || !pvd ) {
		Warning( "(CDHmm::Viterbi) memory allocation failed" );
		if( ppBestPath ) Mfree( (void**) ppBestPath, nTime );
		if( pvd ) Mfree( (void *)pvd );
		return( 0.0 );
	}

	m_pnSymbol = pnOutput;
	m_nTime    = nTime;

	SHORT	nLast = -1;

	// Initialize
	InitViterbi( pvd, ppBestPath );

	// Recursion & find the minimum Negative Log Likelihood
	lll = RecursionViterbi( pvd, ppBestPath, &nLast, pnFinal, nFinal );

	// Backtracking
	if( pnPath )
		BacktrackViterbi( nLast, pnPath, ppBestPath );

	Mfree( (void**) ppBestPath, m_nTime );
	Mfree( (void *) pvd );

	return( lll );
}

REAL CDHmm::UnitedViterbi( SHORT* pnOutput, const SHORT nTime, 
						  SHORT nProcessedTime, CVScore &vscore )
{
	REAL	lll;

	if( !m_bLoged )
		NegLogProbability();

	m_pnSymbol = pnOutput;
	m_nTime    = nTime;

	SHORT nLast = -1;

	nProcessedTime--;

	// Initialize
	if (nProcessedTime <= 0) {
		InitViterbi( vscore );
		nProcessedTime = 1;
	}

	// Recursion & find the minimum Negative Log Likelihood
	lll = RecursionViterbi( vscore, nProcessedTime, &nLast );

	// Backtracking
	BacktrackViterbi( vscore, nLast );

	return( lll );
}

REAL CDHmm::ObsProbability( SHORT* pnOutput, const SHORT nTime )
{
	m_pnSymbol = pnOutput;
	NewTrainVar( nTime );
	Forward();
	DeleteTrainVar();
	return( LogLikelihood() );
}

////////////////////////////////////////////////////////////////////////////
// Read/Write
////////////////////////////////////////////////////////////////////////////

BOOL CDHmm::ReadModel( const char* pszFilename )
{
	FILE*	fp;
	int		nState, nSymbol;
	BOOL	bThreshold;

	if( !(fp = fopen( pszFilename, "r" )) ) {
		Warning( "(CDHmm::ReadModel) file <%s> not found", pszFilename );
		return( FALSE );
	}

	// read num states and num symbol
	if( !fscanf_s( fp, "# NumStates : %d\n", &nState ) ||
		!fscanf_s( fp, "# NumSymbols : %d\n", &nSymbol ) ||
		!fscanf_s( fp, "# IsGarbage : %d\n", &bThreshold ) ) {
		Warning( "(CDHmm::ReadModel) model file is corrupted" );
		return( FALSE );
	}

	New( (SHORT)nState, (SHORT)nSymbol );
	InitialStateProb();
	InitialOutputProb();

	if (bThreshold == 1)
		m_bThreshold = TRUE;

	SHORT	iState, jState;
	REAL	rTmp;
	char	szLine[MAX_BUFFER];

	// read the transition probability
	while( fgets( szLine, MAX_BUFFER-1, fp ) != NULL ){
		if( szLine[0] == '\n' || szLine[0] == '#' )
			continue;
		sscanf_s( szLine, "%d%d%g", &iState, &jState, &rTmp );
		if( iState < 0 || iState >= m_nState || jState < 0 || jState >= m_nState ) {
			Warning( "(CDHmm::ReadModel) model file isn't well defined" );
			Delete();
			fclose( fp );
			return( FALSE );
		}
		else
			m_pprA[iState][jState] = rTmp;
	}

	fclose( fp );
	return( TRUE );
}

BOOL CDHmm::Read( const char* pszFilename )
{
	FILE*	fp;
	SHORT   nState, nSymbol;
	BOOL	bThreshold;
	char	szMagic[MAGIC_SIZE];

	// file open and check the status of the fopen(...)
	if( !( fp = fopen( pszFilename, "rb" ) ) ) {
		Warning( "(CDHmm::ReadBinary) file <%s> not found", pszFilename );
		return( FALSE );
	}

	SetStatusMessage( "Reading model %s...", pszFilename );

	// Read Header and check the header is correct.
	int r = MyRead( szMagic, 1, MAGIC_SIZE, fp );

	if( r <= 0 || strcmp( szMagic, cszMagicStr ) ) {
		Warning( "(CDHmm::ReadBinary) feature file is corrupted" );
		fclose( fp );
		return( FALSE );
	}

	// read the num states and num symbols
	if( MyRead( &nState, sizeof( SHORT ), 1, fp ) &&
		MyRead( &nSymbol, sizeof( SHORT ), 1, fp ) &&
		MyRead( &bThreshold, sizeof( BOOL ), 1, fp ) ) {

		New( nState, nSymbol );
		if (bThreshold == 1)
			m_bThreshold = TRUE;

		int	 i;

		MyRead( m_prPi, sizeof(REAL), m_nState, fp );
		for( i = 0 ; i < m_nState ; i ++ )
			MyRead( m_pprA[i], sizeof(REAL), m_nState, fp );
		for( i = 0 ; i < m_nState ; i ++ )
			MyRead( m_pprB[i], sizeof(REAL), m_nSymbol, fp );
	} else {
		Warning( "(CDHmm::ReadBinary) feature file is corrupted" );
		fclose( fp );
		return( FALSE );
	}

	fclose( fp );
	return( TRUE );
}

BOOL CDHmm::ReadText( const char* pszFilename )
{
	FILE*	fp;
	int		nState, nSymbol;
	BOOL	bThreshold;

	if( !( fp = fopen( pszFilename, "r" )) ) {
		Warning( "(CDHmm::ReadText) file <%s> not found", pszFilename );
		return( FALSE );
	}

	SetStatusMessage( "Reading model %s...", pszFilename );

	// read num states and num symbol
	if( !fscanf_s( fp, "# NumStates : %d\n", &nState ) ||
		!fscanf_s( fp, "# NumSymbols : %d\n", &nSymbol ) ||
		!fscanf_s( fp, "# IsGarbage : %d\n", &bThreshold ) ) {
		Warning( "(CDHmm::ReadText) model file is corrupted" );
		return( FALSE );
	}

	New( (SHORT)nState, (SHORT)nSymbol );
	if (bThreshold == 1)
		m_bThreshold = TRUE;

	int	i, j;
	SHORT	t;

	// read the initial state probability
	for( i = 0 ; i < m_nState ; i ++ )
		fscanf_s( fp, "%d%g", &t, &m_prPi[i] );
  
	// read the transition probability
	for( i = 0 ; i < m_nState ; i ++ ) {
		for( j = 0 ; j < m_nState ; j ++ )
			fscanf_s( fp, "%d%d%g", &t, &t, &m_pprA[i][j] );
	}

	// read the output probability
	for( i = 0 ; i < m_nState ; i ++ ) {
		for( j = 0 ; j < m_nSymbol ; j ++ )
			fscanf_s( fp, "%d%d%g", &t, &t, &m_pprB[i][j] );
	}

	fclose( fp );
	return( TRUE );
}

BOOL CDHmm::Write( const char* pszFilename )
{
	FILE*	fp;

	if( !( fp = fopen( pszFilename, "wb" )) ) {
		Warning( "(CDHmm::WriteBinary) file <%s> not found", pszFilename );
		return( FALSE );
	}

	SetStatusMessage( "Writing model %s...", pszFilename );

	if( MyWrite( cszMagicStr, 1, MAGIC_SIZE, fp ) <= 0 ) {
		Warning( "(CDHmm::WriteBinary) write error" );
		return( FALSE );
	}

	// write the num states and num symbols
	MyWrite( &m_nState, sizeof( SHORT ), 1, fp );
	MyWrite( &m_nSymbol, sizeof( SHORT ), 1, fp );
	MyWrite( &m_bThreshold, sizeof( BOOL ), 1, fp );

	int i;
	// initial state probability
	MyWrite( m_prPi, sizeof(REAL), m_nState, fp );

	// transition probability
	for( i = 0 ; i < m_nState ; i ++ )
		MyWrite( m_pprA[i], sizeof(REAL), m_nState, fp );

	// output probability
	for( i = 0 ; i < m_nState ; i ++ )
		MyWrite( m_pprB[i], sizeof(REAL), m_nSymbol, fp );

	fclose( fp );
	return( TRUE );
}

BOOL CDHmm::WriteText( const char* pszFilename )
{
	FILE*	fp;

	if( !( fp = fopen( pszFilename, "w" )) ) {
		Warning( "(CDHmm::WriteText) file <%s> not found", pszFilename );
		return( FALSE );
	}

	SetStatusMessage( "Writing model %s...", pszFilename );

	// write the num states and num symbols
	fprintf( fp, "# NumStates : %d\n", m_nState );
	fprintf( fp, "# NumSymbols : %d\n", m_nSymbol );
	fprintf( fp, "# IsGarbage : %d\n", m_bThreshold );

	int i, j;
	// write the initial state probability
	for( i = 0 ; i < m_nState ; i ++ )
		fprintf( fp, "%d\t%g\n", i, m_prPi[i] );
	fputc( '\n', fp );

	// write the transition probability
	for( i = 0 ; i < m_nState ; i++ ) {
		for( j = 0 ; j < m_nState ; j++ )
			fprintf( fp, "%d\t%d\t%g\n", i, j, m_pprA[i][j] );
		fputc( '\n', fp );
	}

	// write the output probability
	for( i = 0 ; i < m_nState ; i++ ) {
		for( j = 0 ; j < m_nSymbol ; j++ )
			fprintf( fp, "%d\t%d\t%g\n", i, j, m_pprB[i][j] );
		fputc( '\n', fp );
	}

	fclose( fp );
	return( TRUE );
}


#define MAX_STATE_DIST		1000

typedef struct tagTmpHmm {
	SHORT	nParent;
	REAL	rSelf;
	REAL*	prB;
} TmpHmm;

REAL CalcRelativeEntropy( TmpHmm* pSrc, TmpHmm* pDest, SHORT nSymbol )
{
	REAL	rDist = 0.;

	for(int i=0; i<nSymbol; i++)
		rDist += ( pSrc->prB[i] * log(pSrc->prB[i] / pDest->prB[i])
				 + pDest->prB[i] * log(pDest->prB[i] / pSrc->prB[i]) ) / 2.;

	return rDist;
}

SHORT CDHmm::ReduceThresholdState()
{
	SHORT	i, j;
	SHORT	nReduce = m_nState - (m_nSymbol + m_nSymbol/2);
	TmpHmm*	pTHmm;

	if ( nReduce <= 0 )
		return m_nState;

	pTHmm = new TmpHmm[m_nState];

	for(i=0; i<m_nState; i++) {
		pTHmm[i].nParent = -1;
		pTHmm[i].prB = MallocREAL( m_nSymbol );
		pTHmm[i].rSelf = m_pprA[i][i];
		memcpy(pTHmm[i].prB, m_pprB[i], sizeof(REAL)*m_nSymbol); 
	}

	for(int k=0; k<nReduce; k++) {
		REAL	rDist, rMinDist = MAX_STATE_DIST;
		SHORT	nMinA=-1, nMinB=-1;

		for(i=0; i<m_nState; i++) {
			if (pTHmm[i].nParent >= 0)
				continue;
			for(j=i+1; j<m_nState; j++) {
				if (pTHmm[j].nParent >= 0)
					continue;
				rDist = CalcRelativeEntropy( &pTHmm[i], &pTHmm[j], m_nSymbol );

				if (rDist < rMinDist) {
					nMinA = i;
					nMinB = j;
					rMinDist = rDist;
				}
			}
		}

		if (nMinB < 0)
			break;

		pTHmm[nMinB].nParent = nMinA;

		for(i=0; i<m_nSymbol; i++) {
			pTHmm[nMinA].prB[i] = (pTHmm[nMinA].prB[i] + pTHmm[nMinB].prB[i]) / 2.;
			if ( pTHmm[nMinA].prB[i] < VERY_SMALL )
				pTHmm[nMinA].prB[i] = VERY_SMALL;
		}
		pTHmm[nMinA].rSelf = (pTHmm[nMinA].rSelf + pTHmm[nMinB].rSelf) / 2.;
	}

	SHORT  nNewState = m_nState - nReduce;
	REAL*  prPi     = (REAL*)  MallocREAL( nNewState );
	REAL** pprA     = (REAL**) MallocREAL( nNewState, nNewState );
	REAL** pprB     = (REAL**) MallocREAL( nNewState, m_nSymbol );

	int	nsize = sizeof(REAL) * nNewState;
	int ksize = sizeof(REAL) * m_nSymbol;
  
	memset( prPi, 0, nsize );
	for( i = 0 ; i < nNewState ; i ++ )
		memset( pprA[i], 0, nsize );
	for( i = 0 ; i < nNewState ; i ++ )
		memset( pprB[i], 0, ksize );

	for(i=0, j=0; i<m_nState; i++) {
		if (pTHmm[i].nParent < 0) {
			pprA[j][j] = pTHmm[i].rSelf;
			memcpy( pprB[j], pTHmm[i].prB, ksize );
			j++;
		}
	}

	Mfree( m_prPi );
	Mfree( (void**) m_pprA, m_nState );
	Mfree( (void**) m_pprB, m_nState );
	
	m_nState = nNewState;
	m_prPi = prPi;
	m_pprA = pprA;
	m_pprB = pprB;

	for(i=0; i<m_nState; i++)
		Mfree(pTHmm[i].prB);

	delete[] pTHmm;

	return nNewState;
}


void CDHmm::MakeThresholdModel( CDHmm* pHMM, SHORT nModel, 
							 SHORT nSymbol, REAL rThreshProb, BOOL bReduced )
{
	int		i, j, k;
	REAL	op;
	SHORT	iState, nNumState = 0;

	PrintProgressString( "Generating threshold model using other models..." );

#ifdef USE_GARBAGE_MODEL
	nNumState = nSymbol;

	New( nNumState, nSymbol );
	ClearVariable();
	m_bThreshold = TRUE;

	// Sum of prior probability is THRESHOLD_PROB, not 1.0
	// For giving some penalty to matching score of threshold model
	op = rThreshProb / nNumState;
	for(i=0; i<nNumState; i++)
		m_prPi[i] = op;

	for(i=0; i<nNumState; i++) {
		op = 1.0 / (REAL)nNumState;
		for(j=0; j<nNumState; j++)
			m_pprA[i][j] = op;

		op = 1.0 / (REAL)nSymbol;
		for(j=0; j<nSymbol; j++)
			m_pprB[i][j] = op;
	}
#else
	for(i=0; i<nModel; i++)
		nNumState += pHMM[i].m_nState - 2;	// Remove initial/final state

	New( nNumState, nSymbol );
	ClearVariable();
	m_bThreshold = TRUE;

	iState = 0;
	for(i=0; i<nModel; i++) {
		for(j=1; j<pHMM[i].m_nState-1; j++) {
			m_pprA[iState][iState] = pHMM[i].m_pprA[j][j];

			for(k=0; k<pHMM[i].m_nSymbol; k++)
				m_pprB[iState][k] = pHMM[i].m_pprB[j][k];

			iState++;
		}
	}

	if (bReduced)
		nNumState = ReduceThresholdState();

	// Sum of prior probability is THRESHOLD_PROB, not 1.0
	// For giving some penalty to matching score of threshold model
	op = rThreshProb / nNumState;
	for(i=0; i<nNumState; i++)
		m_prPi[i] = op;

	for(i=0; i<nNumState; i++) {
		op = (1.0 - m_pprA[i][i]) / (REAL)(nNumState-1.0);
		for(j=0; j<nNumState; j++) {
			if (i != j)
				m_pprA[i][j] = op;
		}
	}
#endif
}

void CDHmm::InitializeModel( SHORT nState, SHORT nSymbol )
{
	int		i, j;
	REAL	op;

	PrintProgressString( "Generating normal model..." );

	New( nState, nSymbol );
	ClearVariable();

	m_prPi[0] = 1.0;
	for(i=1; i<nState; i++)
		m_prPi[i] = 0.0;

	op = 1. / nSymbol;
	for(i=0; i<nState; i++) {
		if (i != nState-1) {
			m_pprA[i][i] = 0.5;
			m_pprA[i][i+1] = 0.5;
		}

		for(j=0; j<nSymbol; j++)
			m_pprB[i][j] = op;
	}
}

REAL CalcPenalty( SHORT nFinal, SHORT nNumState, SHORT nNumSymbol )
{
	if ( nFinal == (nNumState-1) ) 
		return 0.0;

	REAL	rResult = 0.0;
	REAL	rPenalty = -log( 0.5 / nNumSymbol );

	if ( nNumState > 10 )	// Etc Model
		return rPenalty;

	for(int i=nFinal; i<nNumState-1; i++)
		rResult += rPenalty;

	return rResult;
}

void CDHmm::SetThresholdModelTransition( CDHmm* pHMM, CDHmm* pETC, SHORT nModel,
					SHORT nCurStart, SHORT nCurFinal, REAL rThreshProb )
{
	REAL	tmp, op, opETC;
	SHORT	i, j, other;

	for(i=nCurStart; i<=nCurFinal; i++) {
		tmp = (1. - m_pprA[i][i]);
//#ifdef USE_GARBAGE_MODEL
//#	ifdef MAX_PERFORMANCE
//		rThreshProb = 1.0;
//		op = tmp * 1.0e-12 / nModel;
//#	endif
//#else
		op = tmp * (1. - rThreshProb) / nModel;
//#endif
		other = 0;
		for(j=0; j<nModel; j++) {
			m_pprA[i][other] = op;
			other += pHMM[j].m_nState;
		}

		opETC = tmp * rThreshProb / (pETC->m_nState-1.);
		for(j=nCurStart; j<=nCurFinal; j++) {
			if (i != j)
				m_pprA[i][j] = opETC;
		}
	}

	opETC = rThreshProb / (REAL)pETC->m_nState;
	for(i=nCurStart; i<=nCurFinal; i++)
		m_prPi[i] = opETC;
}

void CDHmm::SetModelTransition( CDHmm* pHMM, CDHmm* pETC, SHORT nModel, 
					SHORT nCurStart, SHORT nCurFinal, REAL rThreshProb )
{
	REAL	op, opETC;
	SHORT	i, other;


//#ifdef USE_GARBAGE_MODEL
//#	ifdef MAX_PERFORMANCE
//	rThreshProb = 1.0;
//	op = 1.0e-12 / nModel;
//#	endif
//#else
	op = (1. - rThreshProb) / nModel;
//#endif
	other = 0;
	for(i=0; i<nModel; i++) {
		m_pprA[nCurFinal][other] = op;
		other += pHMM[i].m_nState;
	}

	opETC = rThreshProb / (REAL)pETC->m_nState;
	for(i=0; i<pETC->m_nState; i++)
		m_pprA[nCurFinal][other+i] = opETC;

	m_prPi[nCurStart] = op;
}

void CDHmm::MakeUnitedModel( CDHmm* pHMM, CDHmm* pETC, SHORT nModel, 
							SHORT nSymbol, REAL rThreshProb )
{
	int		i, j, k;
	SHORT	iState, nStart, nNumState = 0;

	PrintProgressString( "Generating united model using other models..." );

	for(i=0; i<nModel; i++)
		nNumState += pHMM[i].m_nState;
	nNumState += pETC->m_nState;
//	nNumState += 1;			// Start node

	New( nNumState, nSymbol );
	ClearVariable();

	iState = 0;

	for(i=0; i<nModel; i++) {
		nStart = iState;
		for(j=0; j<pHMM[i].m_nState; j++) {
			for(k=0; k<pHMM[i].m_nState; k++)
				m_pprA[iState][nStart+k] = pHMM[i].m_pprA[j][k];
			for(k=0; k<pHMM[i].m_nSymbol; k++)
				m_pprB[iState][k] = pHMM[i].m_pprB[j][k];
			iState++;
		}

		SetModelTransition( pHMM, pETC, nModel, nStart, iState-1, rThreshProb );
	}

	nStart = iState;

	for(i=0; i<pETC->m_nState; i++) {
		for(k=0; k<pETC->m_nState; k++)
			m_pprA[iState][nStart+k] = pETC->m_pprA[i][k];
		for(k=0; k<pETC->m_nSymbol; k++)
			m_pprB[iState][k] = pETC->m_pprB[i][k];
		iState++;
	}
	SetThresholdModelTransition( pHMM, pETC, nModel, nStart, iState-1, rThreshProb );
}

REAL* CDHmm::CalcVQProbability()
{
	int		i, j;
	REAL*	pProb;

	if (m_pprB == NULL)
		return NULL;

	pProb = (REAL*) new REAL[m_nSymbol];

	for(i=0; i<m_nSymbol; i++)
		pProb[i] = 0.0;

	for(i=0; i<m_nSymbol; i++) {
		for(j=0; j<m_nState; j++)
			pProb[i] += m_pprB[j][i];
		pProb[i] /= (REAL)m_nState;
	}

	return pProb;
}
