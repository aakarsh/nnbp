#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
#include "tracer.h"

#include <inttypes.h>
#include <math.h>
#define LOOPPREDICTOR

// we extensively looked at these parameters
#define NHIST  47
#define   MAXHIST  6551
#define   NHISTGEHL  209
#define  MAXHISTGEHL  1393
#define   NRHSP  80
#define   GWIDTH  60
#define   LWIDTH  60
#define   BWIDTH  60
#define   SWIDTH  60
#define   PWIDTH  42
//////////////////////////////


#define GPSTEP 6
#define LPSTEP 6
#define BPSTEP 6
#define PPSTEP 6
#define SPSTEP 6

#define YWIDTH 60
#define YPSTEP 6
#define TPSTEP 6
#define TWIDTH 60
#define QPSTEP 6
#define QWIDTH 60



#define MAXNHIST 47

#define LOGTAB 19
//#define SWIDTH 60
#define LOGB LOGTAB
#define LOGG LOGTAB
#define LOGSIZE (10)



#define LOGSIZEG (LOGSIZE)
#define LOGSIZEL (LOGSIZE)
#define LOGSIZEB (LOGSIZE)
#define LOGSIZES (LOGSIZE)
#define LOGSIZEP (LOGSIZE)
#define LOGSIZEY (LOGSIZE)
#define LOGSIZET (LOGSIZE)
#define LOGSIZEQ (LOGSIZE)


// The perceptron-inspired components
int8_t PERCYHA[(1 << LOGSIZEY)][10 * (1 << YPSTEP)];
int8_t PERC[(1 << LOGSIZEP)][10 * (1 << GPSTEP)];
int8_t PERCLOC[(1 << LOGSIZEL)][10 * (1 << LPSTEP)];
int8_t PERCBACK[(1 << LOGSIZEB)][10 * (1 << BPSTEP)];
int8_t PERCPATH[(1 << LOGSIZEP)][10 * (1 << PPSTEP)];
int8_t PERCSLOC[(1 << LOGSIZES)][10 * (1 << SPSTEP)];
int8_t PERCTLOC[(1 << LOGSIZET)][10 * (1 << TPSTEP)];
int8_t PERCQLOC[(1 << LOGSIZEQ)][10 * (1 << QPSTEP)];


// the three 16 local histories components
#define LOGLOCAL 4
#define NLOCAL (1<<LOGLOCAL)
#define INDLOCAL (PC & (NLOCAL-1))
long long L_shist[NLOCAL];
#define LNB 15
int Lm[LNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34, 39, 44, 50, 56, 63 };

int8_t LGEHLA[LNB][(1 << LOGTAB)];
int8_t *LGEHL[LNB];

#define  LOGSECLOCAL 4
#define NSECLOCAL (1<<LOGSECLOCAL)
#define NB 3
#define INDSLOCAL  (((PC ^ (PC >>5)) >> NB) & (NSECLOCAL-1))
long long S_slhist[NLOCAL];
#define SNB 15
int Sm[SNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34, 39, 44, 50, 56, 63 };

int8_t SGEHLA[SNB][(1 << LOGTAB)];
int8_t *SGEHL[SNB];

#define  LOGTLOCAL 4
#define NTLOCAL (1<<LOGTLOCAL)
#define INDTLOCAL  (((PC ^ (PC >>3)  ^(PC >> 6))) & (NTLOCAL-1))
long long T_slhist[NLOCAL];
#define TNB 15
int Tm[TNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34, 39, 44, 50, 56, 63 };

int8_t TGEHLA[TNB][(1 << LOGTAB)];
int8_t *TGEHL[TNB];


// 65536 different local histories

#define  LOGQLOCAL 15
#define NQLOCAL (1<<LOGQLOCAL)
#define INDQLOCAL  (((PC ^ (PC >>2) ^(PC>> 4) ^ (PC >> 8))) & (NQLOCAL-1))
long long Q_slhist[(1 << LOGQLOCAL)];
#define QNB 15
int Qm[QNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34, 39, 44, 50, 56, 63 };

int8_t QGEHLA[QNB][(1 << LOGTAB)];
int8_t *QGEHL[QNB];


//about the skeleton histories
#define YNB 15
int Ym[YNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34, 39, 44, 50, 56, 63 };

int8_t YGEHLA[YNB][(1 << LOGTAB)];
int8_t *YGEHL[YNB];
long long YHA;
UINT32 LastBR[8];

#define BNB 10
int Bm[BNB] = { 2, 4, 6, 9, 12, 16, 20, 24, 29, 34 };

int8_t BGEHLA[BNB][(1 << LOGTAB)];
int8_t *BGEHL[BNB];


//for the choser
#define CGNB 8
int CGm[CGNB] = { 2, 4, 6, 9, 12, 16, 21, 26 };

int8_t CGGEHLA[CGNB][(1 << LOGTAB)];
int8_t *CGGEHL[CGNB];
#define CLNB 8
int CLm[CLNB] = { 2, 4, 6, 9, 12, 16, 21, 26 };

int8_t CLGEHLA[CLNB][(1 << LOGTAB)];
int8_t *CLGEHL[CLNB];
int indexchose;
int LCHOSE;
int Cupdatethreshold;
#define LOGSIZECHOSE LOGTAB
#define MAXSIZECHOSE  (1<<LOGTAB)
#define SIZECHOSE  (1<<(LOGSIZECHOSE))
#define INDCHOSE (PC & (SIZECHOSE -1))
int8_t GCHOSE[MAXSIZECHOSE * 4];
// end choser



#define LOGSIZEUSEALT 15
#define MAXSIZEUSEALT  (1<<15)
int8_t use_alt_on_na[MAXSIZEUSEALT][2];
#define SIZEUSEALT  (1<<(LOGSIZEUSEALT))
#define INDUSEALT (PC & (SIZEUSEALT -1))


#define HYSTSHIFT 0		//no sharing is OK

#define LOGL (LOGB-8)
#define WIDTHNBITERLOOP LOGTAB	// we predict only loops with less than 1M iterations
#define LOOPTAG 15		//tag width in the loop predictor



#define INDUPD (PC & ((1 << LOGSIZE) - 1))	//index for updatethreshold
int PERCSUM;
int Pupdatethreshold[(1 << LOGSIZE)];
int updatethreshold;

//the GEHL predictor 
#ifndef LOGGEHL
// base 2 logarithm of number of entries  on each GEHL  component
#define LOGGEHL (LOGTAB+1)
#endif
#define MINSTEP 2
#define MINHISTGEHL 1
static int8_t GEHL[1 << LOGGEHL][NHISTGEHL + 1];	//GEHL tables
int mgehl[NHISTGEHL + 1];	//GEHL history lengths
int GEHLINDEX[NHISTGEHL + 1];
int8_t choseGT[(1 << LOGB)];

//The MACRHSP inspired predictor
#define LOGRHSP LOGGEHL
static int8_t RHSP[1 << LOGRHSP][NRHSP + 1];	//RHSP tables
int mrhsp[NRHSP + 1];	//RHSP history lengths
int RHSPINDEX[NRHSP + 1];
int SUMRHSP;

int m[NHIST + 1];

#define MINHIST (3)		// shortest history length

#define PHISTWIDTH 27		// width of the path history
#define TBITS 16		// minimum tag width
#define PERCWIDTH 8
#define CHOSEWIDTH 6
#define CONFWIDTH 6
#define UWIDTH 2
#define CWIDTH 5		// predictor counter width on the TAGE tagged tables
#define HISTBUFFERLENGTH (1<<18)	// we use a 256K entries history buffer to store the branch history



long long BHIST;
long long lastaddr;
long long P_phist;
long long GHIST;

int SUMGEHL;
bool pred_withoutloop;

#define LOGBIAS (LOGB-4)
int8_t Bias[(1 << (LOGBIAS + 1))];
#define INDBIAS (((PC<<1) ^ pred_inter) & ((1<<(LOGBIAS+1))-1))
bool LowConf;
bool HighConf;

int LSUM;
int8_t BIM;

// utility class for index computation
// this is the cyclic shift register for folding 
// a long global history into a smaller number of bits; see P. Michaud's PPM-like predictor at CBP-1
class folded_history
{
public:
//  unsigned intern;

  unsigned comp;
  int CLENGTH;
  int OLENGTH;
  int OUTPOINT;
    folded_history ()
  {
  }
  void init (int original_length, int compressed_length, int N)
  {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;

  }

  void update (uint8_t * h, int PT)
  {
    comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
    comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp = (comp) & ((1 << CLENGTH) - 1);
  }

};

#ifdef LOOPPREDICTOR
class lentry			//loop predictor entry
{
public:
  uint16_t NbIter;
  uint8_t confid;
  uint16_t CurrentIter;
  uint16_t TAG;
  uint8_t age;
  bool dir;



    lentry ()
  {
    confid = 0;
    CurrentIter = 0;
    NbIter = 0;
    TAG = 0;
    age = 0;
    dir = false;



  }
};
#endif

class bentry			// TAGE bimodal table entry  
{
public:
  int8_t hyst;
  int8_t pred;


    bentry ()
  {
    pred = 0;

    hyst = 1;
  }
};
class gentry			// TAGE global table entry
{
public:
  int8_t ctr;
  uint16_t tag;
  int8_t u;
    gentry ()
  {
    ctr = 0;
    tag = 0;
    u = 0;

  }
};




int TICK, LOGTICK;
//int BIGTICK;
		//control counter for the smooth resetting of useful counters




uint8_t ghist[HISTBUFFERLENGTH];



int ptghist;
long long phist;		//path history
folded_history ch_i[NHIST + 1];	//utility for computing TAGE indices
folded_history ch_t[2][NHIST + 1];	//utility for computing TAGE tags
folded_history chgehl_i[NHISTGEHL + 1];	//utility for computing GEHL indices

folded_history chrhsp_i[NRHSP + 1];	//utility for computing GEHL indices

//For the TAGE predictor
bentry *btable;			//bimodal TAGE table
gentry *gtable[NHIST + 1];	// tagged TAGE tables


int TB[NHIST + 1];		// tag width for the different tagged tables
int logg[NHIST + 1];		// log of number entries of the different tagged tables
int GI[NHIST + 1];		// indexes to the different tables are computed only once  
uint GTAG[NHIST + 1];	// tags for the different tables are computed only once  
int BI;				// index of the bimodal table
bool pred_taken;		// prediction
bool alttaken;			// alternate  TAGEprediction
bool tage_pred;			// TAGE prediction
bool LongestMatchPred;
int HitBank;			// longest matching bank
int AltBank;			// alternate matching bank
int Seed;			// for the pseudo-random number generator
bool pred_inter;



#ifdef LOOPPREDICTOR
lentry *ltable;			//loop predictor table
//variables for the loop predictor
bool predloop;			// loop predictor prediction
int LIB;
int LI;
int LHIT;			//hitting way in the loop predictor
int LTAG;			//tag on the loop predictor
bool LVALID;			// validity of the loop predictor prediction
int8_t WITHLOOP;		// counter to monitor whether or not loop prediction is beneficial

#endif






class PREDICTOR
{
public:
  PREDICTOR (void)
  {





#ifdef LOOPPREDICTOR
    LVALID = false;

#endif

    Seed = 0;
    TICK = 0;
    phist = 0;

    for (int i = 0; i < HISTBUFFERLENGTH; i++)
      ghist[0] = 0;
    ptghist = 0;
    m[1] = MINHIST;
    m[NHIST] = MAXHIST;
    for (int i = 2; i <= NHIST; i++)
      {
	m[i] = (int) (((double) MINHIST *
		       pow ((double) (MAXHIST) /
			    (double) MINHIST,
			    (double) (i -
				      1) / (double) ((NHIST - 1)))) + 0.5);
      }
    for (int i = 2; i <= NHIST; i++)
      if (m[i] <= m[i - 1] + 2)
	m[i] = m[i - 1] + 2;


    mgehl[0] = 0;

    mgehl[1] = MINHISTGEHL;
    mgehl[NHISTGEHL] = MAXHISTGEHL;
    for (int i = 2; i <= NHISTGEHL; i++)
      {
	mgehl[i] =
	  (int) (((double) MINHISTGEHL *
		  pow ((double) (MAXHISTGEHL) / (double) MINHISTGEHL,
		       (double) (i - 1) / (double) ((NHISTGEHL - 1)))) + 0.5);

      }
// just guarantee that all history lengths are distinct
    for (int i = 1; i <= NHISTGEHL; i++)
      if (mgehl[i] <= mgehl[i - 1] + MINSTEP)
	mgehl[i] = mgehl[i - 1] + MINSTEP;
    for (int i = 1; i <= NHISTGEHL; i++)
      {
	chgehl_i[i].init (mgehl[i], LOGGEHL, ((i & 1)) ? i : 1);

      }
// initialization of GEHL tables
    for (int j = 0; j < (1 << LOGGEHL); j++)
      for (int i = 0; i <= NHISTGEHL; i++)
	GEHL[j][i] = (i & 1) ? -4 : 3;

//RHSP initialization

    for (int i = 1; i <= NRHSP; i++)
      mrhsp[i] = 6 * i;

    for (int i = 1; i <= NRHSP; i++)
      {
	chrhsp_i[i].init (mrhsp[i], LOGRHSP, ((i & 1)) ? i : 1);

      }
// initialization of RHSP tables
    for (int j = 0; j < (1 << LOGRHSP); j++)
      for (int i = 0; i <= NRHSP; i++)
	RHSP[j][i] = (i & 1) ? -4 : 3;


    for (int i = 1; i <= NHIST; i++)
      {

	TB[i] = TBITS;
      }

// log2 of number entries in the tagged components
    for (int i = 0; i <= NHIST; i++)
      logg[i] = LOGG;



#ifdef LOOPPREDICTOR
    ltable = new lentry[1 << (LOGL)];
#endif

    for (int i = 0; i <= NHIST; i++)
      {
	gtable[i] = new gentry[1 << (logg[i])];

      }
//initialisation of the functions for index and tag computations

    for (int i = 1; i <= NHIST; i++)
      {
	ch_i[i].init (m[i], (logg[i]), i - 1);
	ch_t[0][i].init (ch_i[i].OLENGTH, TB[i], i);
	ch_t[1][i].init (ch_i[i].OLENGTH, TB[i] - 1, i + 2);
      }

    updatethreshold = 100;
    Cupdatethreshold = 11;

    for (int i = 0; i < (1 << LOGSIZE); i++)
      Pupdatethreshold[i] = 0;


    btable = new bentry[1 << LOGB];

    for (int i = 0; i < LNB; i++)
      LGEHL[i] = &LGEHLA[i][0];

    for (int i = 0; i < SNB; i++)
      SGEHL[i] = &SGEHLA[i][0];
    for (int i = 0; i < QNB; i++)
      QGEHL[i] = &QGEHLA[i][0];
    for (int i = 0; i < TNB; i++)
      TGEHL[i] = &TGEHLA[i][0];


    for (int i = 0; i < BNB; i++)
      BGEHL[i] = &BGEHLA[i][0];

    for (int i = 0; i < YNB; i++)
      YGEHL[i] = &YGEHLA[i][0];
    for (int i = 0; i < LNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      LGEHL[i][j] = -1;

	    }
	}

    for (int i = 0; i < SNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      SGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < QNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      QGEHL[i][j] = -1;

	    }
	}

    for (int i = 0; i < TNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      TGEHL[i][j] = -1;

	    }
	}

    for (int i = 0; i < BNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      BGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < YNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      YGEHL[i][j] = -1;

	    }
	}


//for the choser
    for (int i = 0; i < CGNB; i++)
      CGGEHL[i] = &CGGEHLA[i][0];
    for (int i = 0; i < CLNB; i++)
      CLGEHL[i] = &CLGEHLA[i][0];
    for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
      GCHOSE[j] = -1;


    for (int i = 0; i < CGNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      CGGEHL[i][j] = -1;

	    }
	}
    for (int i = 0; i < CLNB; i++)
      for (int j = 0; j < ((1 << LOGTAB) - 1); j++)
	{
	  if (j & 1)
	    {
	      CLGEHL[i][j] = -1;

	    }
	}

    for (int j = 0; j < (1 << (LOGBIAS + 1)); j++)
      Bias[j] = (j & 1) ? 23 : -24;


    for (int i = 0; i < (1 << LOGSIZES); i++)
      for (int j = 0; j < ((SWIDTH / SPSTEP)) * (1 << SPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERCSLOC[i][j] = -1;

	    }
	}
    for (int i = 0; i < (1 << LOGSIZEQ); i++)
      for (int j = 0; j < ((QWIDTH / QPSTEP)) * (1 << QPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERCQLOC[i][j] = -1;

	    }
	}

    for (int i = 0; i < (1 << LOGSIZEP); i++)
      for (int j = 0; j < ((GWIDTH / GPSTEP)) * (1 << GPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERC[i][j] = -1;
	    }
	}


    for (int i = 0; i < (1 << LOGSIZEL); i++)
      for (int j = 0; j < ((LWIDTH / LPSTEP)) * (1 << LPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERCLOC[i][j] = -1;


	    }
	}


    for (int i = 0; i < (1 << LOGSIZEB); i++)
      for (int j = 0; j < ((BWIDTH / BPSTEP)) * (1 << BPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERCBACK[i][j] = -1;


	    }
	}
    for (int i = 0; i < (1 << LOGSIZEP); i++)
      for (int j = 0; j < ((PWIDTH / PPSTEP)) * (1 << PPSTEP); j++)
	{
	  if (j & 1)
	    {


	      PERCPATH[i][j] = -1;
	    }
	}
    for (int i = 0; i < (1 << LOGSIZEY); i++)
      for (int j = 0; j < ((YWIDTH / YPSTEP)) * (1 << YPSTEP); j++)
	{
	  if (j & 1)
	    {
	      PERCYHA[i][j] = -1;


	    }
	}



  }

// index function for the GEHL tables
//FGEHL serves to mix path history
  int FGEHL (int A, int size, int bank)
  {
    int A1, A2;
    int SH = (bank % LOGGEHL);
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << LOGGEHL) - 1));
    A2 = (A >> LOGGEHL);
    A2 = ((A2 << SH) & ((1 << LOGGEHL) - 1)) + (A2 >> (LOGGEHL - SH));
    A = A1 ^ A2;
    A = ((A << SH) & ((1 << LOGGEHL) - 1)) + (A >> (LOGGEHL - SH));
    return (A);
  }
  int gehlindex (uint32_t PC, int bank)
  {
    int index;
    int M = (mgehl[bank] > 16) ? 16 : mgehl[bank];
    index =
      PC ^ (PC >> ((mgehl[bank] % LOGGEHL) + 1)) ^ chgehl_i[bank].comp ^
      FGEHL (phist, M, bank);
    return (index & ((1 << LOGGEHL) - 1));
  }
// index function for the RHSP tables
//FRHSP serves to mix path history
  int FRHSP (int A, int size, int bank)
  {
    int A1, A2;
    int SH = (bank % LOGRHSP);
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << LOGRHSP) - 1));
    A2 = (A >> LOGRHSP);
    A2 = ((A2 << SH) & ((1 << LOGRHSP) - 1)) + (A2 >> (LOGRHSP - SH));
    A = A1 ^ A2;
    A = ((A << SH) & ((1 << LOGRHSP) - 1)) + (A >> (LOGRHSP - SH));
    return (A);
  }
  int rhspindex (uint32_t PC, int bank)
  {
    int index;
    index = PC ^ (PC >> ((mrhsp[bank] % LOGRHSP) + 1)) ^ chrhsp_i[bank].comp;
    if (bank > 1)
      index ^= chrhsp_i[bank - 1].comp;
    if (bank > 3)
      index ^= chrhsp_i[bank / 3].comp;
    return (index & ((1 << LOGRHSP) - 1));
  }

  // index function for the bimodal table

  int bindex (uint32_t PC)
  {
    return ((PC) & ((1 << (LOGB)) - 1));
  }


// the index functions for the tagged tables uses path history as in the OGEHL predictor
//F serves to mix path history
  int F (int A, int size, int bank)
  {
    int A1, A2;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << logg[bank]) - 1));
    A2 = (A >> logg[bank]);
    A2 =
      ((A2 << bank) & ((1 << logg[bank]) - 1)) + (A2 >> (logg[bank] - bank));
    A = A1 ^ A2;
    A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));
    return (A);
  }
// gindex computes a full hash of PC, ghist and phist
  int gindex (unsigned int PC, int bank, int hist, folded_history * ch_i)
  {
    int index;
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    index =
      PC ^ (PC >> (abs (logg[bank] - bank) + 1))
      ^ ch_i[bank].comp ^ F (hist, M, bank);
    return (index & ((1 << (logg[bank])) - 1));
  }

  //  tag computation
  uint16_t gtag (unsigned int PC, int bank, folded_history * ch0,
		 folded_history * ch1)
  {
    int tag = PC ^ ch0[bank].comp ^ (ch1[bank].comp << 1);
    return (tag & ((1 << TB[bank]) - 1));
  }

  // up-down saturating counter
  void ctrupdate (int8_t & ctr, bool taken, int nbits)
  {
    if (taken)
      {
	if (ctr < ((1 << (nbits - 1)) - 1))
	  ctr++;
      }
    else
      {
	if (ctr > -(1 << (nbits - 1)))
	  ctr--;
      }
  }

  //THE LOOP PREDICTOR

#ifdef LOOPPREDICTOR
  int lindex (uint32_t PC)
  {
    return ((PC & ((1 << (LOGL - 2)) - 1)) << 2);
  }


//loop prediction: only used if high confidence
//skewed associative 4-way
//At fetch time: speculative
#define CONFLOOP 15
  bool getloop (uint32_t PC)
  {
    LHIT = -1;

    LI = lindex (PC);
    LIB = ((PC >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1));
    LTAG = (PC >> (LOGL - 2)) & ((1 << 2 * LOOPTAG) - 1);
    LTAG ^= (LTAG >> LOOPTAG);
    LTAG = (LTAG & ((1 << LOOPTAG) - 1));

    for (int i = 0; i < 4; i++)
      {
	int index = (LI ^ ((LIB >> i) << 2)) + i;

	if (ltable[index].TAG == LTAG)
	  {
	    LHIT = i;
	    LVALID = ((ltable[index].confid == CONFLOOP)
		      ||
		      ((ltable[index].confid * ltable[index].NbIter >
			128) & (ltable[index].confid >= 4)));
	    if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
	      return (!(ltable[index].dir));
	    else
	      return ((ltable[index].dir));
	  }
      }

    LVALID = false;
    return (false);

  }

  void loopupdate (uint32_t PC, bool Taken, bool ALLOC)
  {
    if (LHIT >= 0)
      {
	int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
//already a hit 
	if (LVALID)
	  {
	    if (Taken != predloop)
	      {
// free the entry
		ltable[index].NbIter = 0;
		ltable[index].age = 0;
		ltable[index].confid = 0;
		ltable[index].CurrentIter = 0;
		return;

	      }
	    else		//if ((predloop != pred_withoutloop) || ((MYRANDOM () & 7) == 0))
	    if (ltable[index].age < CONFLOOP)
	      ltable[index].age++;
	  }

	ltable[index].CurrentIter++;
	ltable[index].CurrentIter &= ((1 << WIDTHNBITERLOOP) - 1);
	//loop with more than 2** WIDTHNBITERLOOP iterations are not treated correctly; but who cares :-)
	if (ltable[index].CurrentIter > ltable[index].NbIter)
	  {
	    ltable[index].confid = 0;
	    ltable[index].NbIter = 0;
//treat like the 1st encounter of the loop 
	  }
	if (Taken != ltable[index].dir)
	  {
	    if (ltable[index].CurrentIter == ltable[index].NbIter)
	      {
		if (ltable[index].confid < CONFLOOP)
		  ltable[index].confid++;
		if (ltable[index].NbIter < 3)
		  //just do not predict when the loop count is 1 or 2     
		  {
// free the entry
		    ltable[index].dir = Taken;
		    ltable[index].NbIter = 0;
		    ltable[index].age = 0;
		    ltable[index].confid = 0;
		  }
	      }
	    else
	      {
		if (ltable[index].NbIter == 0)
		  {
// first complete nest;
		    ltable[index].confid = 0;
		    ltable[index].NbIter = ltable[index].CurrentIter;
		  }
		else
		  {
//not the same number of iterations as last time: free the entry
		    ltable[index].NbIter = 0;
		    ltable[index].confid = 0;
		  }
	      }
	    ltable[index].CurrentIter = 0;
	  }

      }
    else if (ALLOC)

      {
	uint32_t X = MYRANDOM () & 3;
	for (int i = 0; i < 4; i++)
	  {
	    int LHIT = (X + i) & 3;
	    int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
	    if (ltable[index].age == 0)
	      {
		ltable[index].dir = !Taken;
// most of mispredictions are on last iterations
		ltable[index].TAG = LTAG;
		ltable[index].NbIter = 0;
		ltable[index].age = 7;
		ltable[index].confid = 0;
		ltable[index].CurrentIter = 0;
		break;

	      }
	    else
	      ltable[index].age--;
	    break;
	  }
      }
  }
#endif
  int INDEX;

  bool getbim ()
  {

    BIM = (btable[BI].pred << 1) + btable[BI >> HYSTSHIFT].hyst;
    HighConf = (BIM == 0) || (BIM == 3);
    LowConf = !HighConf;
    return (btable[BI].pred > 0);
  }

  void baseupdate (bool Taken)
  {
    int inter = BIM;
    if (Taken)
      {
	if (inter < 3)
	  inter += 1;
      }
    else if (inter > 0)
      inter--;
    btable[BI].pred = inter >> 1;
    btable[BI >> HYSTSHIFT].hyst = (inter & 1);

  };

//just a simple pseudo random number generator: use available information
// to allocate entries  in the loop predictor
  int MYRANDOM ()
  {
    Seed++;
    Seed ^= phist;
    Seed = (Seed >> 21) + (Seed << 11);

    return (Seed);
  };


  //  TAGE PREDICTION: same code at fetch or retire time but the index and tags must recomputed
  void Tagepred (unsigned int PC)
  {
    HitBank = 0;
    AltBank = 0;

    for (int i = 1; i <= NHIST; i++)
      {
	GI[i] = gindex (PC, i, phist, ch_i);

	GTAG[i] = gtag (PC, i, ch_t[0], ch_t[1]);
      }
    GI[0] = PC & ((1 << LOGG) - 1);
    BI = PC & ((1 << LOGB) - 1);
//Look for the bank with longest matching history
    for (int i = NHIST; i > 0; i--)
      {
	if (gtable[i][GI[i]].tag == GTAG[i])
	  {
	    HitBank = i;
	    LongestMatchPred = (gtable[HitBank][GI[HitBank]].ctr >= 0);
	    break;
	  }
      }
//Look for the alternate bank
    for (int i = HitBank - 1; i > 0; i--)
      {
	if (gtable[i][GI[i]].tag == GTAG[i])

	  {

	    AltBank = i;
	    break;
	  }
      }
//computes the prediction and the alternate prediction

    if (HitBank > 0)
      {
	if (AltBank > 0)
	  alttaken = (gtable[AltBank][GI[AltBank]].ctr >= 0);
	else
	  alttaken = getbim ();

//if the entry is recognized as a newly allocated entry and 
//USE_ALT_ON_NA is positive  use the alternate prediction

	int index = INDUSEALT ^ LongestMatchPred;
	bool Huse_alt_on_na =
	  (use_alt_on_na[index][HitBank > (NHIST / 3)] >= 0);
	if ((!Huse_alt_on_na)
	    || (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) > 1))
	  tage_pred = LongestMatchPred;
	else
	  tage_pred = alttaken;

	HighConf =
	  (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) >=
	   (1 << CWIDTH) - 1);
	LowConf = (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1);
      }
    else
      {
	alttaken = getbim ();
	tage_pred = alttaken;
	LongestMatchPred = alttaken;

      }
  }



//compute the prediction
  bool GetPrediction (UINT32 PC)
  {
    Tagepred (PC);

    pred_taken = tage_pred;
    pred_inter = pred_taken;
    LSUM = 0;
    predict_gehl (PC);
    predict_rhsp (PC);
    LSUM += SUMGEHL;
    LSUM += SUMRHSP;
    int16_t ctr = Bias[INDBIAS];
    LSUM += 2 * (2 * ctr + 1);
    LSUM +=
      percpredict (PC, GHIST,
		   PERC[PC & ((1 << LOGSIZEG) - 1)], GPSTEP, GWIDTH);

    LSUM +=
      percpredict (PC, L_shist[INDLOCAL],
		   PERCLOC[PC & ((1 << LOGSIZEL) - 1)], LPSTEP, LWIDTH);

    LSUM +=
      percpredict (PC, BHIST,
		   PERCBACK[PC & ((1 << LOGSIZEB) - 1)], BPSTEP, BWIDTH);

    LSUM +=
      percpredict (PC, P_phist,
		   PERCPATH[PC & ((1 << LOGSIZEP) - 1)], PPSTEP, PWIDTH);

    LSUM +=
      percpredict (PC, S_slhist[INDSLOCAL],
		   PERCSLOC[PC & ((1 << LOGSIZES) - 1)], SPSTEP, SWIDTH);

    LSUM +=
      percpredict (PC, T_slhist[INDTLOCAL],
		   PERCTLOC[PC & ((1 << LOGSIZET) - 1)], TPSTEP, TWIDTH);

    LSUM +=
      percpredict (PC, Q_slhist[INDQLOCAL],
		   PERCQLOC[PC & ((1 << LOGSIZEQ) - 1)], QPSTEP, QWIDTH);

    LSUM +=
      percpredict (PC, YHA, PERCYHA[PC & ((1 << LOGSIZEY) - 1)], YPSTEP,
		   YWIDTH);
    LSUM += Gpredict (PC, L_shist[INDLOCAL], Lm, LGEHL, LNB);
    LSUM += Gpredict (PC, S_slhist[INDSLOCAL], Sm, SGEHL, SNB);
    LSUM += Gpredict (PC, T_slhist[INDTLOCAL], Tm, TGEHL, TNB);
    LSUM += Gpredict (PC, Q_slhist[INDQLOCAL], Qm, QGEHL, QNB);
    LSUM += Gpredict (PC, BHIST, Bm, BGEHL, BNB);
    LSUM += Gpredict (PC, YHA, Ym, YGEHL, BNB);

    if (pred_inter != (LSUM >= 0))
      {

	int CLASS = pred_inter;
	indexchose = ((INDCHOSE << 3) + CLASS) & ((1 << LOGTAB) - 1);
	LCHOSE = 2 * (GCHOSE[indexchose] + 1);
	LCHOSE += Gpredict (indexchose, GHIST, CGm, CGGEHL, CGNB);
	LCHOSE += Gpredict (indexchose, L_shist[INDLOCAL], CLm, CLGEHL, CLNB);
	if (LCHOSE >= 0)
	  {
	    pred_taken = (LSUM >= 0);
	  }

	else
	  pred_taken = pred_inter;
	//just in case



	if (LowConf)
	  {
	    pred_taken = (LSUM >= 0);
	  }

	if (HitBank == 0)
	  if (HighConf)
	    pred_taken = pred_inter;

      }

    pred_withoutloop = pred_taken;


#ifdef LOOPPREDICTOR


    predloop = getloop (PC);	// loop prediction
    pred_taken = ((WITHLOOP >= 0) && (LVALID)) ? predloop : pred_taken;
#endif





    return (pred_taken);




  }
  void predict_gehl (uint32_t PC)
  {

//index computation     
    for (int i = 1; i <= NHISTGEHL; i++)
      if (i < 5)
	GEHLINDEX[i] = gehlindex ((PC << 1) + tage_pred, i);
      else
	GEHLINDEX[i] = gehlindex (PC, i);

//    GEHLINDEX[0] = (PC & ((1 << LOGGEHL) - 1));

// SUMGEHL is centered
    SUMGEHL = 0;
    for (int i = 1; i <= NHISTGEHL; i++)
      SUMGEHL += 2 * GEHL[GEHLINDEX[i]][i] + 1;

  }
  void gehlupdate (uint32_t PC, bool taken)
  {
//update the GEHL  predictor tables
    for (int i = NHISTGEHL; i >= 1; i--)
      ctrupdate (GEHL[GEHLINDEX[i]][i], taken, PERCWIDTH);
  }



  void predict_rhsp (uint32_t PC)
  {

//index computation     
    for (int i = 1; i <= NRHSP; i++)
      RHSPINDEX[i] = rhspindex (PC, i);
    RHSPINDEX[0] = (PC & ((1 << LOGRHSP) - 1));

// SUMRHSP is centered
    SUMRHSP = 0;
    for (int i = 1; i <= NRHSP; i++)
      SUMRHSP += 2 * RHSP[RHSPINDEX[i]][i] + 1;

  }

  void rhspupdate (uint32_t PC, bool taken)
  {

    for (int i = NRHSP; i >= 1; i--)
      ctrupdate (RHSP[RHSPINDEX[i]][i], taken, PERCWIDTH);
  }


  void
    HistoryUpdate (uint32_t PC, uint8_t brtype, bool taken,
		   uint32_t target, long long &X, int &Y,
		   folded_history * H, folded_history * G,
		   folded_history * J, folded_history * K,
		   folded_history * L, long long &LH, long long &BH,
		   long long &lastaddr, long long &XH, long long &NH,
		   long long &QH, long long &TH, long long &YH,
		   long long &BHIST)
  {


// History skeletton
    bool V = false;

    for (int i = 0; i <= 7; i++)
      if (LastBR[i] == PC)
	V = true;

    for (int i = 7; i >= 1; i--)
      LastBR[i] = LastBR[i - 1];
    LastBR[0] = PC;
    if (!V)
      YH = (YH << 1) ^ (taken ^ ((PC >> 5) & 1));

    if ((PC < lastaddr - 16) || (abs (PC - lastaddr) >= 128))

      {
	BH = (BH << 1) ^ (PC & 15);

      }
    lastaddr = PC;

//Path history

    TH = (TH << 1) ^ (taken ^ ((PC >> 5) & 1));
    if (brtype == OPTYPE_BRANCH_COND)
      {
	/*local history  */


	QH = (QH << 1) + (taken);

	/*second local history + a little bit of path */
	LH = (LH << 1) + (taken);
	XH = (XH << 1) + (taken);
	XH ^= ((PC >> LOGSECLOCAL) & 15);
	NH = (NH << 1) + (taken);
	NH ^= ((PC >> LOGTLOCAL) & 15);
/*branch history*/
	BHIST = (BHIST << 1) + taken;
      }

    int T = ((target ^ (target >> 3) ^ PC) << 1) + taken;
    int PATH = PC;

    int8_t DIR = (T & 127);
    T >>= 1;
    int PATHBIT = (PATH & 127);
    PATH >>= 1;
//update  history
    Y--;
    int inter = (T ^ PC);
    inter = (inter ^ (inter >> 4)) & 1;

    ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;

    X = (X << 1) ^ PATHBIT;
    X = (X & ((1 << PHISTWIDTH) - 1));
//prepare next index and tag computations for user branchs 
    for (int i = 1; i <= NHIST; i++)
      {

	H[i].update (ghist, Y);
	G[i].update (ghist, Y);
	J[i].update (ghist, Y);


      }
    for (int i = 1; i <= NHISTGEHL; i++)
      {
	K[i].update (ghist, Y);
      }

    for (int i = 1; i <= NRHSP; i++)
      {
	L[i].update (ghist, Y);
      }



//END UPDATE  HISTORIES
  }






  // PREDICTOR UPDATE

  void
    UpdatePredictor (UINT32 PC, bool resolveDir, bool predDir,
		     UINT32 branchTarget)
  {

    bool taken = resolveDir;
    UINT32 target = branchTarget;

#ifdef LOOPPREDICTOR
    if (LVALID)
      {
	if (pred_withoutloop != predloop)
	  ctrupdate (WITHLOOP, (predloop == taken), 7);
      }

    loopupdate (PC, taken, (pred_withoutloop != taken));
#endif
    bool LPRED = (LSUM >= 0);
    bool CRES = ((LSUM >= 0) == taken);
    bool CPRED = (LCHOSE >= 0);

    if (pred_inter != LPRED)
      {
	if ((CPRED != CRES) || (abs (LCHOSE) < Cupdatethreshold))
	  {
	    if (CPRED != CRES)
	      Cupdatethreshold += 1;
	    else
	      Cupdatethreshold -= 1;
	    ctrupdate (GCHOSE[indexchose], CRES, CHOSEWIDTH);
	    Gupdate (indexchose, CRES, GHIST, CGm, CGGEHL, CGNB, CHOSEWIDTH);
	    Gupdate (indexchose, CRES, L_shist[INDLOCAL], CLm, CLGEHL, CLNB,
		     CHOSEWIDTH);
	  }
      }

//Statistic
    if ((LPRED != taken)
	|| ((abs (LSUM) < updatethreshold + Pupdatethreshold[INDUPD])))
      {
	if (LPRED != taken)
	  updatethreshold += 1;
	else
	  updatethreshold -= 1;
	if (LPRED != taken)
	  Pupdatethreshold[INDUPD] += 1;
	else
	  Pupdatethreshold[INDUPD] -= 1;
	gehlupdate (PC, taken);
	rhspupdate (PC, taken);
	ctrupdate (Bias[INDBIAS], taken, PERCWIDTH);
	updateperc (taken, PERC[PC & ((1 << LOGSIZEG) - 1)],
		    GHIST, GPSTEP, GWIDTH);
	updateperc (taken, PERCLOC[PC & ((1 << LOGSIZEL) - 1)],
		    L_shist[INDLOCAL], LPSTEP, LWIDTH);
	updateperc (taken, PERCBACK[PC & ((1 << LOGSIZEB) - 1)],
		    BHIST, BPSTEP, BWIDTH);

	updateperc (taken, PERCPATH[PC & ((1 << LOGSIZEP) - 1)],
		    P_phist, PPSTEP, PWIDTH);

	updateperc (taken, PERCSLOC[PC & ((1 << LOGSIZES) - 1)],
		    S_slhist[INDSLOCAL], SPSTEP, SWIDTH);
	updateperc (taken, PERCTLOC[PC & ((1 << LOGSIZES) - 1)],
		    T_slhist[INDTLOCAL], SPSTEP, SWIDTH);
	updateperc (taken, PERCQLOC[PC & ((1 << LOGSIZEQ) - 1)],
		    Q_slhist[INDQLOCAL], QPSTEP, QWIDTH);
	updateperc (taken, PERCYHA[PC & ((1 << LOGSIZEY) - 1)], YHA, YPSTEP,
		    YWIDTH);
	Gupdate (PC, taken, L_shist[INDLOCAL], Lm, LGEHL, LNB, PERCWIDTH);
	Gupdate (PC, taken, S_slhist[INDSLOCAL], Sm, SGEHL, SNB, PERCWIDTH);
	Gupdate (PC, taken, T_slhist[INDTLOCAL], Tm, TGEHL, TNB, PERCWIDTH);
	Gupdate (PC, taken, Q_slhist[INDQLOCAL], Qm, QGEHL, QNB, PERCWIDTH);

	Gupdate (PC, taken, BHIST, Bm, BGEHL, BNB, PERCWIDTH);
	Gupdate (PC, taken, YHA, Ym, YGEHL, YNB, PERCWIDTH);

      }






    bool ALLOC = ((tage_pred != taken) & (HitBank < NHIST));
    {
      // try to allocate a  new entries only if TAGE prediction was wrong
      if (HitBank > 0)
	{
// Manage the selection between longest matching and alternate matching
// for "pseudo"-newly allocated longest matching entry

	  bool PseudoNewAlloc =
	    (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) <= 1);
// an entry is considered as newly allocated if its prediction counter is weak
	  if (PseudoNewAlloc)
	    {
	      if (LongestMatchPred == taken)
		ALLOC = false;
// if it was delivering the correct prediction, no need to allocate a new entry
//even if the overall prediction was false
	      if (LongestMatchPred != alttaken)
		{
		  int index = INDUSEALT ^ LongestMatchPred;

		  ctrupdate (use_alt_on_na[index]
			     [HitBank > (NHIST / 3)], (alttaken == taken), 4);


		}

	    }
	}
    }




    if (ALLOC)
      {
/* for such a huge predictor allocating  several entries is better*/
#define NNN 24

	int T = NNN;

	int A = 1;
	if ((MYRANDOM () & 127) < 32)
	  A = 2;
	int Penalty = 0;
	int NA = 0;

	for (int i = HitBank + A; i <= NHIST; i += 1)
	  {

	    if (gtable[i][GI[i]].u == 0)
	      {
		gtable[i][GI[i]].tag = GTAG[i];
		gtable[i][GI[i]].ctr = (taken) ? 0 : -1;

		NA++;
		if (T <= 0)
		  break;
		i += 1;

		T -= 1;
	      }
	    else

	      {
		bool DEC = false;

		DEC = (TICK >= (MYRANDOM () & 127));
		if (DEC)
		  if (T == NNN)
		    if ((MYRANDOM () & 1))
		      gtable[i][GI[i]].u--;

		Penalty++;
	      }
	  }


	TICK += (2 * (Penalty - NA));

	if (TICK < 0)
	  TICK = 0;
	if (LOGB > 14)
	  TICK++;
	if (TICK >= 127)
	  TICK = 127;


      }




    if (HitBank > 0)
      {
	if (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)
	  if (LongestMatchPred != taken)

	    {			// acts as a protection 
	      if (AltBank > 0)
		{
		  ctrupdate (gtable[AltBank][GI[AltBank]].ctr, taken, CWIDTH);

		}
	      if (AltBank == 0)
		baseupdate (taken);


	    }


	ctrupdate (gtable[HitBank][GI[HitBank]].ctr, taken, CWIDTH);
//sign changes: no way it can have been useful
	if (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)

	  gtable[HitBank][GI[HitBank]].u = 0;

      }
    else
      baseupdate (taken);

    if (LongestMatchPred != alttaken)
      if (LongestMatchPred == taken)
	if (gtable[HitBank][GI[HitBank]].u < (1 << UWIDTH) - 1)
	  gtable[HitBank][GI[HitBank]].u++;






    HistoryUpdate (PC, OPTYPE_BRANCH_COND, taken, target, phist,
		   ptghist, ch_i, ch_t[0], ch_t[1], chgehl_i,
		   chrhsp_i, L_shist[INDLOCAL], BHIST, lastaddr,
		   S_slhist[INDSLOCAL], T_slhist[INDTLOCAL],
		   Q_slhist[INDQLOCAL], P_phist, YHA, GHIST);



  }

  int
    percpredict (int PC, long long BHIST, int8_t * line, int PSTEP, int WIDTH)
  {
    PERCSUM = 0;
    long long bhist = BHIST;
    int PT = 0;

    int index;

    for (int i = 0; i < WIDTH; i += PSTEP)
      {
	index = (bhist & ((1 << PSTEP) - 1));
	int8_t ctr = line[PT + index];

	PERCSUM += 2 * ctr + 1;

	bhist >>= PSTEP;
	PT += (1 << PSTEP);
      }
    return ((PERCSUM));

  }

  void
    updateperc (bool taken, int8_t * line, long long BHIST, int PSTEP,
		int WIDTH)
  {
    int PT = 0;
    long long bhist = BHIST;
    int index;

    for (int i = 0; i < WIDTH; i += PSTEP)
      {
	index = (bhist & ((1 << PSTEP) - 1));
	ctrupdate (line[PT + index], taken, PERCWIDTH);
	bhist >>= PSTEP;
	PT += (1 << PSTEP);
      }
  }


  int
    Gpredict (UINT32 PC, long long BHIST, int *length, int8_t ** tab, int NBR)
  {
    PERCSUM = 0;
    for (int i = 0; i < NBR; i++)
      {
	long long bhist = BHIST & ((long long) ((1 << length[i]) - 1));

	long long index =
	  (((long long) PC) ^ bhist ^ (bhist >> (LOGTAB - i)) ^
	   (bhist >> (40 - 2 * i)) ^ (bhist >> (60 - 3 * i))) & ((1 <<
								  LOGTAB) -
								 1);
	int8_t ctr = tab[i][index];
	PERCSUM += 2 * ctr + 1;


      }
    return ((PERCSUM));

  }

  void
    Gupdate (UINT32 PC, bool taken, long long BHIST, int *length,
	     int8_t ** tab, int NBR, int WIDTH)
  {


    for (int i = 0; i < NBR; i++)
      {
	long long bhist = BHIST & ((long long) ((1 << length[i]) - 1));

	long long index =
	  (((long long) PC) ^ bhist ^ (bhist >> (LOGTAB - i)) ^
	   (bhist >> (40 - 2 * i)) ^ (bhist >> (60 - 3 * i))) & ((1 <<
								  LOGTAB) -
								 1);

	ctrupdate (tab[i][index], taken, WIDTH);

      }
  }

  void TrackOtherInst (UINT32 PC, OpType opType, UINT32 branchTarget)
  {

    bool taken = true;
    int brtype = 0;


    switch (opType)
      {
      case OPTYPE_CALL_DIRECT:
      case OPTYPE_INDIRECT_BR_CALL:
      case OPTYPE_RET:
      case OPTYPE_BRANCH_UNCOND:
	HistoryUpdate (PC, brtype, taken, branchTarget, phist,
		       ptghist, ch_i,
		       ch_t[0], ch_t[1], chgehl_i, chrhsp_i,
		       L_shist[INDLOCAL], BHIST,
		       lastaddr, S_slhist[INDSLOCAL], T_slhist[INDTLOCAL],
		       Q_slhist[INDQLOCAL], P_phist, YHA, GHIST);

	break;
      default:;
      }
  }
};

/***********************************************************/
#endif
