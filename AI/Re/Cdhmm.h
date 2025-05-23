// Discrete HMM
#ifndef	_CDHMM_H
#define	_CDHMM_H

class CDHmm {
public:
	// constructor && destructor
	CDHmm( const char* pszFilename=NULL );
	~CDHmm();

  // attributes
public:
	SHORT	m_nState;	// the number of states						(S)
	SHORT	m_nSymbol;	// the number of observative symbols		(K)
	REAL*	m_prPi;		// state probability						[S]
	REAL**	m_pprA;		// transition matrix						[S][S]
	REAL**	m_pprB;		// observation probability					[S][K]

private:
	BOOL	m_bThreshold;	// Is threshold model?
	BOOL	m_bLoged;	// already do loagarithm on probabilitis
	SHORT	m_nTime;	// Time										(T)
	SHORT	m_nObs;		// the number of observation

	SHORT*	m_pnSymbol;	// output seuqence							[T]

	REAL*	m_prScale;	// scaling factor							[T]
	REAL**	m_pprAlpha;	// forward variable							[T][S]
	REAL**	m_pprBeta;	// backward variable						[T][S]
	REAL**	m_pprGamma;	// prob of being in state i at time t,		[T][S]

	// training with mulitple observation sequence
	REAL	m_rObsP;	// total observation probability
	REAL*	m_prGPi;	// initial state probabilty					[S]
	REAL*	m_prA_d;	// denominator of multi. obs. seq.			[S]
	REAL**	m_pprA_n;	//   nominator of							[S][S]
	REAL*	m_prB_d;	// denominator of multi. obs. seq.			[S]
	REAL**	m_pprB_n;	//   nominator of							[S][K]

	// For peak detector
	REAL	m_rMaxFinal;
	SHORT	m_nFinal;
	SHORT*	m_pnBest;	//											[X]
	SHORT*	m_pnTime;	//
	REAL**	m_pprFinal;	// history of likelihood at final state		[X][M]
						//	X = Maximum length of observation sequence
						//	M = Number of models including threshold model 
						//					( maybe 6 )

	char	m_szDataFile[MAX_BUFFER];	// Prefix of training/testing data files
										// o Format: FilePath\XXXX
										// o Real file names
										//	 - Training data
										//		 FilePath\XXXXNNNN.TRN
										//	 - Testing data
										//		 FilePath\XXXXNNNN.TST
										//   - NNNN: 0001 ~ 
	// operations
public:
	BOOL	LoadModel( const char* pszFilename );
	BOOL	NewModel( const SHORT nState, const SHORT nSymbol );
	void	SetABCounts( SHORT* pnOutput, const SHORT nOutput );
	REAL	Reestimation();
	REAL	Train( SHORT** ppnOutput, const SHORT nOutput, SHORT* pnOutput );
	REAL	Viterbi( SHORT* pnOutput, const SHORT nTime, SHORT* pnPath=NULL );
	REAL	Viterbi( SHORT* pnOutput, const SHORT nTime, 
					 class CVScore &vscore );
	void	InitTraceViterbiScore( SHORT nModel );
	void	ResetTraceViterbiScore( SHORT nModel, SHORT* pnResetT );
	void	TraceViterbiScore( class CVScore& vscore );
	void	DestroyTraceViterbiScore();
	void	CalcTraceBoundingRect( CRect* prc );
	void	DrawTraceViterbiScore( CMetaFileDC* pMetaDC, CView* pView, 
							MODEL* pModel, SHORT nModel,
							struct tagRECOUT* pRecogOut, 
							SHORT nRecogOut, SHORT nDrawT );
	REAL	UnitedViterbi( SHORT* pnOutput, const SHORT nTime,
						   SHORT* pnPath=NULL, SHORT* pnFinal=NULL, 
						   SHORT nFinal=0 );
	REAL	UnitedViterbi( SHORT* pnOutput, const SHORT nTime, 
						   SHORT nProcessedTime, class CVScore &vscore );
	REAL	ObsProbability( SHORT* pnOutput, const SHORT nTime );
	BOOL	Read( const char* pszFilename );
	BOOL	ReadText( const char* pszFilename );
	BOOL	Write( const char* pszFilename );
	BOOL	WriteText( const char* pszFilename );
	void	MakeThresholdModel( CDHmm* pHmm, SHORT nModel, 
							SHORT nSymbol, REAL rThreshProb, BOOL bReduced );
	void	InitializeModel( SHORT nState, SHORT nSymbol );
	void	SetThresholdModelTransition( CDHmm* pHMM, CDHmm* pETC, SHORT nModel,
							SHORT nCurStart, SHORT nCurFinal, REAL rThreshProb );
	void	SetModelTransition( CDHmm* pHMM, CDHmm* pETC, SHORT nModel, 
							SHORT nCurStart, SHORT nCurFinal, REAL rThreshProb );
	void	MakeUnitedModel( CDHmm* pHMM, CDHmm* pETC, SHORT nModel, 
							SHORT nSymbol, REAL rThreshProb );
	REAL*	CalcVQProbability();

private:
	void	New();
	void	NewTrainVar();
	void	NewSumVar();
	void	New( const SHORT nState, const SHORT nSymbol );
	void	NewTrainVar( const SHORT nTime );
	void	Delete();
	void	DeleteTrainVar();
	void	DeleteSumVar();
	void	ClearVariable();
	void	ClearTrainVar();
	void	ClearSumVar();
	void	Initialize();
	void	InitialStateProb();
	void	InitialTransitionProb();
	void	InitialOutputProb();
	void	RescaleAlpha( const SHORT nTime );
	REAL	Forward();
	void	RescaleBeta( const SHORT nTime );
	void	Backward();
	void	CalculateGamma();
	REAL	ExpectedTransition( const SHORT i );
	REAL	ExpectedTransition( const SHORT i, const SHORT j );
	REAL	ExpectedTimes( const SHORT nState, const SHORT Obs );
	void	RescaleMatrics();
	REAL	LogLikelihood();
	void	SumEstimation();
	REAL	NLog( const REAL rValue );
	void	NegLogProbability();
	void	InitViterbi( REAL* pvd, SHORT** ppBestPath );
	void	InitViterbi( class CVScore &vscore );
	REAL	RecursionViterbi( REAL* pVd, SHORT** ppBestPath, 
					SHORT* pnLast, SHORT* pnFinal=NULL, 
					SHORT nFinal=0 );
	REAL	RecursionViterbi( class CVScore &vscore, SHORT nProcessedTime, 
					SHORT* pnLast );
	void	BacktrackViterbi( const SHORT final, SHORT* pnPath, 
							  SHORT** ppBestPath );
	void	BacktrackViterbi( class CVScore &vscore, SHORT nFinal );
	void	DrawOneGraph( CMetaFileDC* pMetaDC, CView* pView, 
					SHORT nLeft, SHORT nTop, SHORT nCurModel, char* pszName, 
					struct tagRECOUT* pRecogOut, SHORT nRecogOut, SHORT nDrawT );
	BOOL	ReadModel( const char* pszFilename );
	SHORT	ReduceThresholdState();

};

REAL CalcPenalty( SHORT nFinal, SHORT nNumState, SHORT nNumSymbol );
#endif
