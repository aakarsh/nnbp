#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
#include "tracer.h"
#include <vector>



/***
  Multi-dimensional history predictor.

  Our multi-dimensional history predictor is built on top of ISL-TAGE (by Andre Seznec).
 ***/



/*** Multi-dimensional history predictor ***/

// entry in multi-dimensional history table
typedef struct {
#define MULTIDIM_TAG_SIZE           18
    UINT32 pc;                      // 18-bit tag of PC
#define MULTIDIM_CONF_MIN           (-11)
#define MULTIDIM_CONF_MAX           4
#define LOG_MULTIDIM_CONF_SIZE      4
    int conf;                       // 4-bit confidence
#define MULTIDIM_LHIST_SIZE         101
#define LOG_MULTIDIM_LHIST_SIZE     7
    vector<bool> lhist;             // MULTIDIM_LHIST_SIZE local history bits
                                    // plus log2(MULTIDIM_LHIST_SIZE) bits per multidim_t to track current size of local history vector
#define MULTIDIM_LSAT_MIN           (-8)
#define MULTIDIM_LSAT_MAX           7
#define MULTIDIM_LSAT_THRES         8
#define MULTIDIM_LSAT_SIZE          16
#define LOG_MULTIDIM_NUM_LSAT       4
    int lsat[MULTIDIM_LSAT_SIZE];   // MULTIDIM_LSAT_SIZE 5-bit saturating counters
} multidim_t;

// multi-dimensional history table
#define MULTIDIMS_SIZE              5
vector<multidim_t> multidims;       // table of multidim_t, MULTIDIMS_SIZE entries
#define MULTIDIM_REPL_SIZE          3  // plus log2(MULTIDIMS_SIZE) bits per multidim_t to maintain ranking (for replacement)

// stat correlator threshold for allocating an entry in multidims
#define MULTIDIMS_STATCOR_THRES     2

// log2(MULTIM_LHIST_SIZE) bits to store number of iterations of current inner loop, taken from last hit in loop predictor 
int multidim_lastloop = 0;

// variables for propagating prediction info to the update function
int multidim_pred = 0;
UINT32 multidim_idx = 0;



/*** ISL-TAGE ***/

#include <inttypes.h>
#include <math.h>
//#define SHARINGTABLES		// let us share the physical among several logic predictor tables
#define INITHISTLENGTH		// uses the "best" history length we found
#define STATCOR			// Use the Statistical Corrector predictor
#define LOOPPREDICTOR		// Use the loop predictor

#define NHIST 12		// 15 tagged tables + 1 bimodal table
#define LOGB 13			// log of number of entries in bimodal predictor
#define HYSTSHIFT 2		// sharing an hysteris bit between 4 bimodal predictor entries
#define LOGG (LOGB-7)		// initial definition was with 2K entries per tagged tables
#define UBITS 3
//The Best  set of history lengths
//int m[NHIST + 1] =		{0, 3, 8, 12, 17, 33, 35, 67, 97, 138, 195, 330, 517, 1193, 1741, 1930};
int m[NHIST + 1] =        {0, 3, 8, 12, 17, 33, 35, 67, 97, 138, 195, 330, 517};

#ifndef INITHISLENGTH
#define MINHIST 8		// shortest history length
#define MAXHIST 2000		// longest history length
#endif

#define PHISTWIDTH 16		// width of the path history

#ifndef SHARINGTABLES
#define TBITS 6
#define MAXTBITS 15
#endif

#define CWIDTH 3		// predictor counter width on the tagged tables
#define HISTBUFFERLENGTH 4096	// we use a 4K entries history buffer to store the branch history

//Statistical corrector parameters
#define NSTAT 5
#define CSTAT 6
#define LOGStatCor (LOGG+1)	// the Statistical Corrector predictor features 5 * 1K entries
int8_t StatCor[1 << (LOGStatCor)];
int8_t USESTATCORTHRESHOLD;
int8_t CountStatCorThreshold;
#define MINUSESTATCORTHRESHOLD 5
#define MAXUSESTATCORTHRESHOLD 31	//just as a security, never reach this limit
#define UPDATESTATCORTHRESHOLD (21+  8*USESTATCORTHRESHOLD)

#define LOGL 6			//64 entries loop predictor 4-way skewed associative
#define WIDTHNBITERLOOP 10	// we predict only loops with less than 1K iterations
#define LOOPTAG 10		//tag width in the loop predictor

// utility class for index computation
// this is the cyclic shift register for folding 
// a long global history into a smaller number of bits; see P. Michaud's PPM-like predictor at CBP-1
class folded_history
{
public:
  unsigned comp;
  int CLENGTH;
  int OLENGTH;
  int OUTPOINT;

    folded_history ()
  {
  }

  void init (int original_length, int compressed_length)
  {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;
  }

  void update (uint8_t * h, int PT)
  {
    comp = (comp << 1) | h[PT & (HISTBUFFERLENGTH - 1)];
    comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp &= (1 << CLENGTH) - 1;
  }
};

#ifdef LOOPPREDICTOR
class lentry			//loop predictor entry
{
public:
  uint16_t NbIter;		//10 bits
  uint8_t confid;		// 3 bits
  uint16_t CurrentIter;		// 10 bits
  uint16_t CurrentIterSpec;	// 10 bits
  uint16_t TAG;			// 10 bits
  uint8_t age;			//3 bits
  bool dir;			// 1 bit

  //47 bits per entry    
    lentry ()
  {
    confid = 0;
    CurrentIter = 0;
    CurrentIterSpec = 0;

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
int8_t USE_ALT_ON_NA;		// "Use alternate prediction on newly allocated":  a 4-bit counter  to determine whether the newly allocated entries should be considered as  valid or not for delivering  the prediction
int TICK, LOGTICK;		//control counter for the smooth resetting of useful counters
int phist;			// use a path history as on  the OGEHL predictor
int Fetch_phist;		//path history
uint8_t ghist[HISTBUFFERLENGTH];
int Retire_ptghist;
int Retire_phist;		//path history
folded_history Retire_ch_i[NHIST + 1];	//utility for computing TAGE indices
folded_history Retire_ch_t[2][NHIST + 1];	//utility for computing TAGE tags

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

//For the TAGE predictor
bentry *btable;			//bimodal TAGE table
gentry *gtable[NHIST + 1];	// tagged TAGE tables
int TB[NHIST + 1];		// tag width for the different tagged tables
int logg[NHIST + 1];		// log of number entries of the different tagged tables

int GI[NHIST + 1];		// indexes to the different tables are computed only once  
int GTAG[NHIST + 1];		// tags for the different tables are computed only once  
int BI;				// index of the bimodal table

bool pred_taken;		// prediction
bool alttaken;			// alternate  TAGEprediction
bool tage_pred;			// TAGE prediction
bool LongestMatchPred;
int HitBank;			// longest matching bank
int AltBank;			// alternate matching bank

int Seed;			// for the pseudo-random number generator

class my_predictor
{
public:
  my_predictor (void)
  {
#ifdef LOOPPREDICTOR
    LVALID = false;
    WITHLOOP = -1;
#endif
    USESTATCORTHRESHOLD = MINUSESTATCORTHRESHOLD;
    USE_ALT_ON_NA = 0;
    Seed = 0;
    LOGTICK = 8;

    TICK = 0;
    Fetch_phist = 0;
    Retire_phist = 0;
    for (int i = 0; i < HISTBUFFERLENGTH; i++)
      ghist[0] = 0;
#ifndef INITHISTLENGTH
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
#endif

#ifndef SHARINGTABLES
    for (int i = 1; i <= NHIST; i++)
      TB[i] = TBITS + (i - 1);
    TB[1]++;
    for (int i = 1; i <= NHIST; i++)

      if (TB[i] > MAXTBITS)
	TB[i] = MAXTBITS;
// log2 of number entries in the tagged components
    for (int i = 1; i <= 3; i++)
      logg[i] = LOGG;
    for (int i = 4; i <= 6; i++)
      logg[i] = LOGG + 1;
    for (int i = 7; i <= 10; i++)
      logg[i] = LOGG;
    for (int i = 10; i <= NHIST; i++)
      logg[i] = LOGG - 1;
    for (int i = 1; i <= NHIST; i++)
      {
	gtable[i] = new gentry[1 << (logg[i])];
      }
#endif

#ifdef SHARINGTABLES
#define STEP1 3
#define STEP2 8
#define STEP3 12
    // log2 of number entries in the tagged components
    logg[1] = LOGG + 1;
    logg[STEP1] = LOGG + 3;
    logg[STEP2] = LOGG + 2;
    logg[STEP3] = LOGG - 1;
    for (int i = 2; i <= STEP1 - 1; i++)
      logg[i] = logg[1] - 3;	/* grouped together 4Kentries */
    for (int i = STEP1 + 1; i <= STEP2 - 1; i++)
      logg[i] = logg[STEP1] - 3;	/*grouped together 16K entries */
    for (int i = STEP2 + 1; i <= STEP3 - 1; i++)
      logg[i] = logg[STEP2] - 3;	/*grouped together 8K entries */
    for (int i = STEP3 + 1; i <= NHIST; i++)
      logg[i] = logg[STEP3] - 3;	/*grouped together 1Kentries */
    gtable[1] = new gentry[1 << logg[1]];
    gtable[STEP1] = new gentry[1 << logg[STEP1]];
    gtable[STEP2] = new gentry[1 << logg[STEP2]];
    gtable[STEP3] = new gentry[1 << logg[STEP3]];

//grouped tables have the same tag width
    for (int i = 1; i <= STEP1 - 1; i++)
      {
	gtable[i] = gtable[1];
	TB[i] = 8;
      }
    for (int i = STEP1; i <= STEP2 - 1; i++)
      {
	gtable[i] = gtable[STEP1];
	TB[i] = 11;
      }
    for (int i = STEP2; i <= STEP3 - 1; i++)
      {
	gtable[i] = gtable[STEP2];
	TB[i] = 13;
      }
    for (int i = STEP3; i <= NHIST; i++)
      {
	gtable[i] = gtable[STEP3];
	TB[i] = 14;
      }
#endif
//initialisation of the functions for index and tag computations

    for (int i = 1; i <= NHIST; i++)
      {
	Retire_ch_i[i].init (m[i], (logg[i]));
	Retire_ch_t[0][i].init (Retire_ch_i[i].OLENGTH, TB[i]);
	Retire_ch_t[1][i].init (Retire_ch_i[i].OLENGTH, TB[i] - 1);
      }

//allocation of the all predictor tables
#ifdef LOOPPREDICTOR
    ltable = new lentry[1 << (LOGL)];
#endif
    btable = new bentry[1 << LOGB];

//initialization of the Statistical Corrector predictor table
    for (int j = 0; j < (1 << LOGStatCor); j++)
      StatCor[j] = ((j & 1) == 0) ? -1 : 0;
#define PRINTCHAR
#ifdef PRINTCHAR
//for printing predictor characteristics
    int NBENTRY = 0;
    int STORAGESIZE = 0;
#ifndef SHARINGTABLES
    for (int i = 1; i <= NHIST; i++)
      {
	STORAGESIZE += (1 << logg[i]) * (CWIDTH + UBITS + TB[i]);
	NBENTRY += (1 << logg[i]);
      }
#endif
#ifdef SHARINGTABLES
    STORAGESIZE += (1 << (logg[1])) * (CWIDTH + UBITS + TB[1]);
    STORAGESIZE += (1 << (logg[STEP1])) * (CWIDTH + UBITS + TB[STEP1]);
    STORAGESIZE += (1 << (logg[STEP2])) * (CWIDTH + UBITS + TB[STEP2]);
    STORAGESIZE += (1 << (logg[STEP3])) * (CWIDTH + UBITS + TB[STEP3]);
    NBENTRY +=
      (1 << (logg[1])) + (1 << (logg[STEP1])) + (1 << (logg[STEP2])) +
      (1 << (logg[STEP3]));
#endif
    fprintf (stdout, "history lengths:");
    for (int i = 1; i <= NHIST; i++)
      {
	fprintf (stdout, "%d ", m[i]);
      }
    fprintf (stdout, "\n");
    STORAGESIZE += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT)) + 524*8;
    fprintf (stdout, "TAGE(+extra space) %d bytes, ", STORAGESIZE / 8);


#ifdef LOOPPREDICTOR
    STORAGESIZE += (1 << LOGL) * (3 * WIDTHNBITERLOOP + LOOPTAG + 3 + 3 + 1);
    fprintf (stdout, "LOOPPRED %d bytes, ",
	     ((1 << LOGL) * (3 * WIDTHNBITERLOOP + LOOPTAG + 3 + 3 + 1)) / 8);
#endif

#ifdef STATCOR
    STORAGESIZE += CSTAT * (1 << (LOGStatCor));
    fprintf (stdout, "Stat Cor %d bytes, ", CSTAT * (1 << (LOGStatCor)) / 8);
#endif
    int sizeMultidimEntry = (MULTIDIM_TAG_SIZE + LOG_MULTIDIM_CONF_SIZE + LOG_MULTIDIM_NUM_LSAT*MULTIDIM_LSAT_SIZE + MULTIDIM_REPL_SIZE + LOG_MULTIDIM_LHIST_SIZE + MULTIDIM_LHIST_SIZE);
    fprintf (stdout, "Wormhole %d bytes, %d per entry \n",  sizeMultidimEntry*MULTIDIMS_SIZE/ 8, sizeMultidimEntry/8);
    STORAGESIZE += sizeMultidimEntry*MULTIDIMS_SIZE;



    fprintf (stdout, "TOTAL STORAGESIZE= %d bytes\n", STORAGESIZE / 8);
#endif
  }

  // index function for the bimodal table

  int bindex (uint32_t pc)
  {
    return ((pc) & ((1 << (LOGB)) - 1));
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
// gindex computes a full hash of pc, ghist and phist
  int gindex (unsigned int pc, int bank, int hist, folded_history * ch_i)
  {
    int index;
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    index =
      pc ^ (pc >> (abs (logg[bank] - bank) + 1)) ^
      ch_i[bank].comp ^ F (hist, M, bank);
    return (index & ((1 << (logg[bank])) - 1));
  }

  //  tag computation
  uint16_t gtag (unsigned int pc, int bank, folded_history * ch0,
		 folded_history * ch1)
  {
    int tag = pc ^ ch0[bank].comp ^ (ch1[bank].comp << 1);
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
  bool getbim ()
  {
    return (btable[BI].pred > 0);
  }
// update  the bimodal predictor: a hysteresis bit is shared among 4 prediction bits
  void baseupdate (bool Taken)
  {
    int inter = (btable[BI].pred << 1) + btable[BI >> HYSTSHIFT].hyst;
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
    Seed ^= Fetch_phist;
    Seed = (Seed >> 21) + (Seed << 11);
    Seed += Retire_phist;

    return (Seed);
  };

//THE LOOP PREDICTOR
#ifdef LOOPPREDICTOR
  int lindex (uint32_t pc)
  {
    return ((pc & ((1 << (LOGL - 2)) - 1)) << 2);
  }

//loop prediction: only used if high confidence
//skewed associative 4-way
//at retire time non-speculative
  bool getloop (uint32_t pc)
  {
    LHIT = -1;

    LI = lindex (pc);
    LIB = ((pc >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1));
    LTAG = (pc >> (LOGL - 2)) & ((1 << 2 * LOOPTAG) - 1);
    LTAG ^= (LTAG >> LOOPTAG);
    LTAG = (LTAG & ((1 << LOOPTAG) - 1));

    for (int i = 0; i < 4; i++)
      {
	int index = (LI ^ ((LIB >> i) << 2)) + i;

	if (ltable[index].TAG == LTAG)
	  {
	    LHIT = i;
	    LVALID = (ltable[index].confid == 7);

        {   /*** Multi-dimensional history predictor ***/
            if (LVALID) {
                multidim_lastloop = 0;
            }
        }
        
	    if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
	      return (!(ltable[index].dir));
	    else
        {

            {   /*** Multi-dimensional history predictor ***/

                // if loop hit, record number of iterations in loop
                if (LVALID) {
                    if (ltable[index].NbIter <= MULTIDIM_LHIST_SIZE) {
                        multidim_lastloop = ltable[index].NbIter;
                    }
                }
            }
            
	        return ((ltable[index].dir));
        }
	  }
      }

    LVALID = false;
    return (false);
  }

  void loopupdate (uint32_t pc, bool Taken, bool ALLOC)
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
	    else if ((predloop != tage_pred) || ((MYRANDOM () & 7) == 0))
	      if (ltable[index].age < 7)
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
		if (ltable[index].confid < 7)
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

	if ((MYRANDOM () & 3) == 0)
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
  //  TAGE PREDICTION: same code at fetch or retire time but the index and tags must recomputed

  void Tagepred ()
  {
    HitBank = 0;
    AltBank = 0;
//Look for the bank with longest matching history
    for (int i = NHIST; i > 0; i--)
      {
	if (gtable[i][GI[i]].tag == GTAG[i])
	  {
	    HitBank = i;
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
	LongestMatchPred = (gtable[HitBank][GI[HitBank]].ctr >= 0);
//if the entry is recognized as a newly allocated entry and 
//USE_ALT_ON_NA is positive  use the alternate prediction
	if ((USE_ALT_ON_NA < 0)
	    || (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) > 1))
	  tage_pred = LongestMatchPred;
	else
	  tage_pred = alttaken;
      }
    else
      {
	alttaken = getbim ();
	tage_pred = alttaken;
	LongestMatchPred = alttaken;

      }
  }
//compute the prediction

  bool predict_brcond (unsigned int pc)
  {
    //if (brtype & IS_BR_CONDITIONAL)
      {
// computes the TAGE table addresses and the partial tags
	for (int i = 1; i <= NHIST; i++)
	  {
	    GI[i] = gindex (pc, i, Retire_phist, Retire_ch_i);
	    GTAG[i] = gtag (pc, i, Retire_ch_t[0], Retire_ch_t[1]);
	  }
#ifdef SHARINGTABLES
	for (int i = 2; i <= STEP1 - 1; i++)
	  GI[i] = ((GI[1] & 7) ^ (i - 1)) + (GI[i] << 3);
	for (int i = STEP1 + 1; i <= STEP2 - 1; i++)
	  GI[i] = ((GI[STEP1] & 7) ^ (i - STEP1)) + (GI[i] << 3);
	for (int i = STEP2 + 1; i <= STEP3 - 1; i++)
	  GI[i] = ((GI[STEP2] & 7) ^ (i - STEP2)) + (GI[i] << 3);
	for (int i = STEP3 + 1; i <= NHIST; i++)
	  GI[i] = ((GI[STEP3] & 7) ^ (i - STEP3)) + (GI[i] << 3);
#endif
	BI = pc & ((1 << LOGB) - 1);
	Tagepred ();

	pred_taken = tage_pred;
#ifdef STATCOR
//access the Statistical Corrector Predictor
//The Statistical Corrector is derived from GEHL
	if (HitBank >= 1)
	  {
	    int IndStatCor[NSTAT];
	    int8_t p[NSTAT];
	    int Sum = 0;

	    Sum += 8 * (2 * gtable[HitBank][GI[HitBank]].ctr + 1);
//biased the sum towards what is predicted by TAGE
//the stronger the prediction the stronger the bias
	    GI[0] = BI;
	    for (int i = 0; i < NSTAT; i++)
	      {
		IndStatCor[i] = GI[i];
		if (i != 0)
		  {
		    IndStatCor[i] <<= 3;
		    IndStatCor[i] ^= (pc ^ i) & 7;
		  }
		IndStatCor[i] <<= 1;
		IndStatCor[i] += tage_pred;
		IndStatCor[i] &= ((1 << (LOGStatCor)) - 1);
		p[i] = (StatCor[IndStatCor[i]]);
		Sum += (2 * p[i] + 1);

	      }
	    bool pred = (Sum >= 0);
	    if (abs (Sum) >= USESTATCORTHRESHOLD)
	      {
		pred_taken = pred;	//Use only if very confident 
	      }
	  }
#endif
#ifdef LOOPPREDICTOR
	predloop = getloop (pc);	// loop prediction
	pred_taken = ((WITHLOOP >= 0) && (LVALID)) ? predloop : pred_taken;
#endif
      }
      
      {   /*** Multi-dimensional history predictor ***/

          if (multidim_lastloop != 0) {
              unsigned int i;
              for (i = 0; i < multidims.size(); i++) {
                  if (multidims[i].pc == pc) {
                      break;
                  }
              }
              if (i < multidims.size()) {
                  if (multidim_lastloop > 1) {
                      if (multidims[i].lhist.size() >= ((unsigned int)multidim_lastloop + 1)) {

                          // generate index using 2D local history bits
                          multidim_idx = 0;
                          multidim_idx |= multidims[i].lhist[multidim_lastloop];
                          multidim_idx <<= 1;
                          multidim_idx |= multidims[i].lhist[multidim_lastloop-1];
                          multidim_idx <<= 1;
                          multidim_idx |= multidims[i].lhist[multidim_lastloop-2];
                          multidim_idx <<= 1;
                          multidim_idx |= multidims[i].lhist[0];
                          
                          multidim_idx = multidim_idx % MULTIDIM_LSAT_SIZE;

                          // get prediction from saturating counter
                          multidim_pred = multidims[i].lsat[multidim_idx];
                          
                          if (multidims[i].conf >= 0) {
                              if (abs((2*multidim_pred) + 1) >= MULTIDIM_LSAT_THRES) {
                                  return (multidim_pred >= 0);
                              }
                          }
                      }
                  }
              }
          }
      }
      
    return pred_taken;
  }

  void HistoryUpdate (uint32_t pc, bool taken,
		      uint32_t target, int &X, int &Y, folded_history * H,
		      folded_history * G, folded_history * J)
  {
    int maxt = 1;

    int T = ((target ^ (target >> 3) ^ pc) << 1) + taken;
    int PATH = pc;
    for (int t = 0; t < maxt; t++)
      {
	bool istaken = (T & 1);
	T >>= 1;
	bool PATHBIT = (PATH & 1);
	PATH >>= 1;
//update  history
	Y--;
	ghist[Y & (HISTBUFFERLENGTH - 1)] = istaken;
	X = (X << 1) + PATHBIT;
	X = (X & ((1 << PHISTWIDTH) - 1));
//prepare next index and tag computations for user branchs 
	for (int i = 1; i <= NHIST; i++)
	  {
	    H[i].update (ghist, Y);
	    G[i].update (ghist, Y);
	    J[i].update (ghist, Y);
	  }
      }
  }

  // PREDICTOR UPDATE

  void update_brcond (uint32_t pc, bool taken,
		      uint32_t target)
  {
    //if (brtype & IS_BR_CONDITIONAL)
      {
#ifdef STATCOR
	if (HitBank >= 1)
	  {
	    int IndStatCor[NSTAT];
	    int8_t p[NSTAT];
	    int Sum = 0;

	    Sum += 8 * (2 * gtable[HitBank][GI[HitBank]].ctr + 1);
	    GI[0] = BI;
	    for (int i = 0; i < NSTAT; i++)
	      {
		IndStatCor[i] = GI[i];
		if (i != 0)
		  {
		    IndStatCor[i] <<= 3;
		    IndStatCor[i] ^= (pc ^ i) & 7;
		  }
		IndStatCor[i] <<= 1;
		IndStatCor[i] += tage_pred;
		IndStatCor[i] &= ((1 << (LOGStatCor)) - 1);
		p[i] = (StatCor[IndStatCor[i]]);
		Sum += (2 * p[i] + 1);

	      }
	    bool pred = (Sum >= 0);
        
        {   /*** Multi-dimensional history predictor ***/

            // only allocate entries into multidims if this branch is difficult for TAGE
            if (abs ((Sum*2) + 1) <= MULTIDIMS_STATCOR_THRES) {
                unsigned int i;
                for (i = 0; i < multidims.size(); i++) {

                    // if entry already exists, move it up in the ranking
                    if (multidims[i].pc == pc) {
                        if (i > 0) {
                            multidim_t multidim = multidims[i];
                            multidims.erase(multidims.begin() + i);
                            i--;
                            multidims.insert(multidims.begin() + i, multidim);
                        }
                        break;
                    }
                }

                // if entry doesn't exist, push it to bottom of ranking
                if (i == multidims.size()) {
                    if (multidims.size() == MULTIDIMS_SIZE) {
                        multidims.pop_back();
                        i--;
                    }
                    multidim_t multidim;
                    multidim.pc = pc;
                    multidim.conf = 0;
                    for (int j = 0; j < MULTIDIM_LSAT_SIZE; j++) {
                        multidim.lsat[j] = 0;
                    }
                    multidims.push_back(multidim);
                }
            }
        }

	    if (abs (Sum) >= USESTATCORTHRESHOLD)
	      {
		pred_taken = pred;	//Use only if very confident 
	      }
	    if (tage_pred != pred)
	      if (abs (Sum) >= USESTATCORTHRESHOLD - 4)
		if (abs (Sum) <= USESTATCORTHRESHOLD - 2)
		  {
		    ctrupdate (CountStatCorThreshold, (pred == taken), 6);
		    if (CountStatCorThreshold == (1 << 5) - 1)
		      if (USESTATCORTHRESHOLD > MINUSESTATCORTHRESHOLD)
			{
			  CountStatCorThreshold = 0;
			  USESTATCORTHRESHOLD -= 2;
			}
		    if (CountStatCorThreshold == -(1 << 5))
		      if (USESTATCORTHRESHOLD < MAXUSESTATCORTHRESHOLD)
			{
			  CountStatCorThreshold = 0;
			  USESTATCORTHRESHOLD += 2;
			}
		  }
	    if ((pred != taken) || (abs (Sum) < UPDATESTATCORTHRESHOLD))
	      for (int i = 0; i < NSTAT; i++)
		{
		  ctrupdate (StatCor[IndStatCor[i]], taken, CSTAT);
		}
	  }
#endif

    {   /*** Multi-dimensional history predictor ***/
        unsigned int i;
        for (i = 0; i < multidims.size(); i++) {
            if (multidims[i].pc == pc) {
                if (multidim_lastloop > 1) {
                    if (multidims[i].lhist.size() >= ((unsigned int)multidim_lastloop + 1)) {

                        // update confidence, only if prediction is different from TAGE
                        if ((multidim_pred >= 0) != tage_pred) {
                            if ((multidim_pred >= 0) == taken) {
                                multidims[i].conf += (multidims[i].conf < MULTIDIM_CONF_MAX) ? 1 : 0;
                            } else {
                                multidims[i].conf += (multidims[i].conf > MULTIDIM_CONF_MIN) ? -1 : 0;
                            }
                        }

                        // update saturating counters
                        if (taken) {
                            multidims[i].lsat[multidim_idx] += (multidims[i].lsat[multidim_idx] < MULTIDIM_LSAT_MAX) ? 1 : 0;
                        } else {
                            multidims[i].lsat[multidim_idx] += (multidims[i].lsat[multidim_idx] > MULTIDIM_LSAT_MIN) ? -1 : 0;
                        }
                    }
                }

                // update local history bits
                multidims[i].lhist.insert(multidims[i].lhist.begin(), taken);
                if (multidims[i].lhist.size() > MULTIDIM_LHIST_SIZE) {
                    multidims[i].lhist.pop_back();
                }
            }
        }
    }

#ifdef LOOPPREDICTOR
	predloop = getloop (pc);	// effective loop prediction at retire  time

	if (LVALID)
	  if (tage_pred != predloop)
	    ctrupdate (WITHLOOP, (predloop == taken), 7);
	loopupdate (pc, taken, (tage_pred != taken));	//update the loop predictor
#endif
	{

	  bool ALLOC = ((tage_pred != taken) & (HitBank < NHIST));

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
		    ctrupdate (USE_ALT_ON_NA, (alttaken == taken), 4);
		}
	    }

//Allocate entries on mispredictions
	  if (ALLOC)
	    {

/* for such a huge predictor allocating  several entries is better*/
	      int T = 3;
	      for (int i = HitBank + 1; i <= NHIST; i += 1)
		{
		  if (gtable[i][GI[i]].u == 0)
		    {
		      gtable[i][GI[i]].tag = GTAG[i];
		      gtable[i][GI[i]].ctr = (taken) ? 0 : -1;
		      gtable[i][GI[i]].u = 0;
		      TICK--;
		      if (TICK < 0)
			TICK = 0;
		      if (T == 0)
			break;
		      i += 1;
		      T--;
		    }
		  else
		    TICK++;
		}
	    }
//manage the u  bit
	  if ((TICK >= (1 << LOGTICK)))
	    {
	      TICK = 0;
// reset the u bit
	      for (int i = 1; i <= NHIST; i++)
		for (int j = 0; j < (1 << logg[i]); j++)
        gtable[i][j].u = (gtable[i][j].u >0) ? gtable[i][j].u -1 : 0;
	    }

//update the prediction

	  if (HitBank > 0)
	    {
	      ctrupdate (gtable[HitBank][GI[HitBank]].ctr, taken, CWIDTH);
// acts as a protection 
	      if ((gtable[HitBank][GI[HitBank]].u == 0))
		{
		  if (AltBank > 0)
		    ctrupdate (gtable[AltBank][GI[AltBank]].ctr, taken,
			       CWIDTH);
		  if (AltBank == 0)
		    baseupdate (taken);
		}
	    }
	  else
	    baseupdate (taken);
// update the u counter
	  if (HitBank > 0)
	    if ( LongestMatchPred != alttaken)
	      {
		if (LongestMatchPred == taken)
		  {
		    if (gtable[HitBank][GI[HitBank]].u < 7)
		      gtable[HitBank][GI[HitBank]].u++;

		 } 
         }
 if (AltBank>0 && alttaken!=taken && gtable[AltBank][GI[AltBank]].u >0)
              gtable[AltBank][GI[AltBank]].u--;
	      
//END PREDICTOR UPDATE

	}

      }

//  UPDATE RETIRE HISTORY  
    HistoryUpdate (pc, taken, target, Retire_phist, Retire_ptghist,
		   Retire_ch_i, Retire_ch_t[0], Retire_ch_t[1]);
  }

};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

class PREDICTOR{

 public:

  // The interface to the four functions below CAN NOT be changed

  PREDICTOR(void);
  bool    GetPrediction(UINT32 PC);  
  void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
  void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget);

  // Contestants can define their own functions below

  my_predictor *mypred;
};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

PREDICTOR::PREDICTOR(void){
    mypred = new my_predictor();    
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool   PREDICTOR::GetPrediction(UINT32 PC){
    return mypred->predict_brcond (PC & 0x3ffff);
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  PREDICTOR::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
	Fetch_phist = (Fetch_phist << 1) + (PC & 1); // used by MYRANDOM
	Fetch_phist = (Fetch_phist & ((1 << PHISTWIDTH) - 1)); // used by MYRANDOM
    mypred->update_brcond (PC & 0x3ffff, resolveDir, branchTarget & 0x7f);
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void    PREDICTOR::TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget){

  // This function is called for instructions which are not
  // conditional branches, just in case someone decides to design
  // a predictor that uses information from such instructions.
  // We expect most contestants to leave this function untouched.

  return;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

/***********************************************************/
#endif

