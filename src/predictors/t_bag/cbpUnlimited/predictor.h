//Autor: Ibrahim Burak Karsli
//Tage code taken from http://www.irisa.fr/alf/index.php?option=com_content&view=article&id=83

/* 
Code has been successively derived from the tagged PPM predictor simulator from Pierre Michaud, the OGEHL predictor simulator from by André Seznec, the TAGE predictor simulator from  André Seznec and Pierre Michaud

*/

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
// #include "tracer.h"
#include <inttypes.h>
#include <math.h>

#define IUM			//Use the Immediate Update Mimicker
#define LSTAT			//use the Local Statistical Predictor


// log of number of entries in bimodal predictor
#define HYSTSHIFT 2		// sharing an hysteris bit between 4 bimodal predictor entries
#define PHISTWIDTH 16		// width of the path history
#define TBITS 16		// minimum tag width 
#define MAXTBITS 30		// maximum tag width 


#define HISTBUFFERLENGTH 131072	// we use a 128K entries history buffer to store the branch history

/*local history management*/

#define NLOCAL 4096		//Number of local histories
#define PCCLASS ((pc) & (NLOCAL-1))	//index  in local history table
#define NLSTAT 7		//Number of tables in the statistical predictor
#define CLSTAT 6		// width of the counters in the local statistical corrector (LSC)predictor
//#define LOGLSTAT  (LOGG-1)
#define LOGSPEC 6
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

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

class specentry
{
public:
  int32_t tag;
  bool pred;
    specentry ()
  {

  }
};
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





class tage{

  // The state is defined for Gshare, change for your design
 private:
bool usepc;
int CWIDTH;
int NHIST;
int LOGB;
int LOGG;
int MINHIST;
int MAXHIST;
int LOGLSTAT;
int tage_alloc;
int *m;
//int m[NHIST + 1];
long L_Fetch_ghist[NLOCAL];
long L_Retire_ghist[NLOCAL];
int8_t Confid[2 * NLOCAL];	// selector TAGE vs LSC
int lm[NLSTAT];	// local history lengths
int LGI[NLSTAT];		//for storing LSC index
//int8_t LStatCor[NLSTAT][(1 << LOGLSTAT)];
int8_t **LStatCor;
int LSUM;
int updatethreshold;
int8_t USE_ALT_ON_NA;		// "Use alternate prediction on newly allocated":  a 4-bit counter  to determine whether the newly allocated entries should be considered as  valid or not for delivering  the prediction
int TICK, LOGTICK;		//control counter for the smooth resetting of useful counters
int phist;			// use a path history as on  the OGEHL predictor
uint8_t ghist[HISTBUFFERLENGTH];
int Fetch_ptghist;
int Fetch_phist;		//path history
folded_history *Fetch_ch_i;
folded_history *Fetch_ch_t1;
folded_history *Fetch_ch_t2;
int Retire_ptghist;
int Retire_phist;		//path history
folded_history *Retire_ch_i;
folded_history *Retire_ch_t1;
folded_history *Retire_ch_t2;
//For the TAGE predictor
bentry *btable;			//bimodal TAGE table
gentry **gtable;	// tagged TAGE tables
int *TB;		// tag width for the different tagged tables
int *logg;		// log of number entries of the different tagged tables
int *GI;		// indexes to the different tables are computed only once  
int *GTAG;		// tags for the different tables are computed only once  

int BI;				// index of the bimodal table
bool pred_taken;		// prediction
bool alttaken;			// alternate  TAGEprediction
bool tage_pred;			// TAGE prediction
bool LongestMatchPred;
int HitBank;			// longest matching bank
int AltBank;			// alternate matching bank
int Seed;			// for the pseudo-random number generator
//For the IUM
int PtIumRetire;
int PtIumFetch;
specentry *IumPred;
int8_t countIum;
bool Ium, BefIum;
bool Trans_TagePred[(1 << 8)];
int Trans_HitBank[(1 << 8)];
int Trans_LSUM[(1 << 8)];
 public:

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
  int8_t BIM;

  bool getbim ()
  {
    BIM = (btable[BI].pred << 1) + btable[BI >> HYSTSHIFT].hyst;


    return (btable[BI].pred > 0);
  }
// update  the bimodal predictor: a hysteresis bit is shared among 4 prediction bits
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
    Seed ^= Fetch_phist;
    Seed = (Seed >> 21) + (Seed << 11);
    Seed += Retire_phist;


    return (Seed);
  };


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

//the IUM predictor
  bool PredIum (bool pred)
  {
#ifdef IUM


    int IumTag = (HitBank) + (GI[HitBank] << 4);
    if (HitBank == 0)
      IumTag = BI << 4;
    int Min =
      (PtIumRetire > PtIumFetch + 512) ? PtIumFetch + 512 : PtIumRetire;

    for (int i = PtIumFetch; i < Min; i++)
      {
	if (IumPred[i & ((1 << LOGSPEC) - 1)].tag == IumTag)
	  {

	    return IumPred[i & ((1 << LOGSPEC) - 1)].pred;
	  }


      }
#endif
    return pred;

  }

  void UpdateIum (bool Taken)
  {
#ifdef IUM
    int IumTag = (HitBank) + (GI[HitBank] << 4);
    if (HitBank == 0)
      IumTag = BI << 4;

    PtIumFetch--;
    IumPred[PtIumFetch & ((1 << LOGSPEC) - 1)].tag = IumTag;
    IumPred[PtIumFetch & ((1 << LOGSPEC) - 1)].pred = Taken;
#endif
    Trans_TagePred[PtIumFetch & 255] = (countIum >= 0) ? Ium : tage_pred;
    if (Ium != BefIum)
      {
	ctrupdate (countIum, (Ium == Taken), 5);

      }
    Trans_HitBank[PtIumFetch & 255] = HitBank;
    Trans_LSUM[PtIumFetch & 255] = LSUM;

  }
//compute the prediction
  // The interface to the four functions below CAN NOT be changed

  tage(int counter, int numtable, int tablesize, int min, int max, int talloc = 3, bool use_pc = true)
    {
      usepc = use_pc;
      CWIDTH = counter;
      NHIST = numtable;
      LOGB = tablesize;
      LOGG = tablesize-4;
      m = new int[NHIST+1];
      MINHIST = min;
      MAXHIST = max;
      LOGLSTAT = LOGG-1;
      tage_alloc = talloc;
      LStatCor = new int8_t* [NLSTAT];
      for(int i = 0; i < NLSTAT; i++)
	LStatCor[i] = new int8_t[(1 << LOGLSTAT)];
      
      
      
      Fetch_ch_i = new folded_history[NHIST+1];
      Fetch_ch_t1 = new folded_history[NHIST+1];
      Fetch_ch_t2 = new folded_history[NHIST+1];
      Retire_ch_i = new folded_history[NHIST+1];
      Retire_ch_t1 = new folded_history[NHIST+1];
      Retire_ch_t2 = new folded_history[NHIST+1];

      gtable = new gentry*[NHIST+1];
      TB = new int[NHIST+1];
      logg = new int[NHIST+1];
      GI = new int[NHIST+1];
      GTAG = new int[NHIST+1];
      
      
      lm[0] = 0;
      lm[1] = 4;
      lm[2] = 10;
      lm[3] = 17;
      lm[4] = 31;
      lm[5] = 45;
      lm[6] = 63;
      
      updatethreshold = 16 << 5;
      PtIumFetch = 0;
      PtIumRetire = 0;
      USE_ALT_ON_NA = 0;
      Seed = 0;
      LOGTICK = 8;
      
      TICK = 0;
      Fetch_phist = 0;
      Retire_phist = 0;
      for (int i = 0; i < HISTBUFFERLENGTH; i++)
	ghist[0] = 0;
      Fetch_ptghist = 0;
      
      
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
      
      for (int i = 1; i <= NHIST; i++)
	{
	  TB[i] = TBITS + (i - 1);
	  if (TB[i] > MAXTBITS)
	    TB[i] = MAXTBITS;
	}
      
      
      if(NHIST == 38){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 2;
	for (int i = 1; i <= 22; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
      }else if(NHIST == 32){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 2;
	for (int i = 1; i <= 28; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
      }else if(NHIST == 30){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
    }else if(NHIST == 29){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 8; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	
      }else if(NHIST == 28){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 7; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	
      }else if(NHIST == 27){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 6; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	
      }else if(NHIST == 26){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 5; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	
      }else if(NHIST == 25){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 4; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	
      }else if(NHIST == 24){
	
	for (int i = 1; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG-1;
	for (int i = 3; i <= 8; i++)
	  if(NHIST>=i) logg[i] = LOGG;
      
      }else if(NHIST == 23){
	
	for (int i = 1; i <= 3; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 4; i <= 5; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 6; i <= 7; i++)
	  if(NHIST>=i)	logg[i] = LOGG;
	for (int i = 8; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
      }else if(NHIST == 22){
	
	for (int i = 1; i <= 3; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 4; i <= 5; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 6; i <= 8; i++)
	  if(NHIST>=i)	logg[i] = LOGG;
	for (int i = 9; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
      }else if(NHIST == 21){
	
	for (int i = 1; i <= 3; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 4; i <= 5; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 6; i <= 9; i++)
	  if(NHIST>=i)	logg[i] = LOGG;
	for (int i = 10; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
	
      }else if(NHIST == 20){
	
	for (int i = 1; i <= 3; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 4; i <= 5; i++)
	  if(NHIST>=i) logg[i] = LOGG + 1;
	for (int i = 6; i <= 6; i++)
	  if(NHIST>=i)	logg[i] = LOGG;
	for (int i = 7; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
      }else{ //15
	
	for (int i = 1; i <= 3; i++)
	  if(NHIST>=i) logg[i] = LOGG;
	for (int i = 4; i <= 6; i++)
	  if(NHIST>=i) logg[i] = LOGG + 1;
	for (int i = 7; i <= 10; i++)
	  if(NHIST>=i)	logg[i] = LOGG;
	for (int i = 10; i <= NHIST; i++)
	  if(NHIST>=i) logg[i] = LOGG - 1;
      }
      
      
      for (int i = 1; i <= NHIST; i++)
	{
	  gtable[i] = new gentry[1 << (logg[i])];
	}
      //initialisation of the functions for index and tag computations
      
      for (int i = 1; i <= NHIST; i++)
	{
	  Fetch_ch_i[i].init (m[i], (logg[i]));
	  Fetch_ch_t1[i].init (Fetch_ch_i[i].OLENGTH, TB[i]);
	  Fetch_ch_t2[i].init (Fetch_ch_i[i].OLENGTH, TB[i] - 1);
	}
      
      for (int i = 1; i <= NHIST; i++)
	{
	  Retire_ch_i[i].init (m[i], (logg[i]));
	  Retire_ch_t1[i].init (Retire_ch_i[i].OLENGTH, TB[i]);
	  Retire_ch_t2[i].init (Retire_ch_i[i].OLENGTH, TB[i] - 1);
	}
      
    
      btable = new bentry[1 << LOGB];
      IumPred = new specentry[1 << LOGSPEC];
      //initialization of the Statistical Corrector predictor table
      for (int i = 0; i < NLSTAT; i++)
	for (int j = 0; j < (1 << LOGLSTAT); j++)
	  LStatCor[i][j] = ((j & 1) == 0) ? -4 : 3;
      
    
#define PRINTCHAR
#ifdef PRINTCHAR
      //for printing predictor characteristics
      int NBENTRY = 0;
      int STORAGESIZE = 0;
      for (int i = 1; i <= NHIST; i++)
	{
	  STORAGESIZE += (1 << logg[i]) * (CWIDTH + 1 + TB[i]);
	  NBENTRY += (1 << logg[i]);
	}
      fprintf (stdout, "history lengths:");
      for (int i = 1; i <= NHIST; i++)
	{
	  fprintf (stdout, "%d ", m[i]);
	}
      fprintf (stdout, "\n");
      STORAGESIZE += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));
      fprintf (stdout, "TAGE %d bytes, ", STORAGESIZE / 8);
#ifdef LSTAT
      STORAGESIZE += CLSTAT * NLSTAT * (1 << (LOGLSTAT));
      fprintf (stdout, "local hist Stat Cor %d bytes, ",
	       CLSTAT * NLSTAT * (1 << (LOGLSTAT)) / 8);
      
#endif
#ifdef IUM
      STORAGESIZE += (1 << LOGSPEC) * 20;
      fprintf (stdout, "IUM  %d bytes, ", ((1 << LOGSPEC) * 20) / 8);
      // Entry width on the speculative table is 20 bits (19 bits to identify the table entry that gives the prediction and 1 prediction bit
#endif
      fprintf (stdout, "TOTAL STORAGESIZE= %d bytes\n", STORAGESIZE / 8);
#endif
    }
  bool    GetPrediction(UINT32 PC)
  {
    uint32_t pc = PC & 0xffffff;
    BefIum = false;
    Ium = false;
    
    // computes the TAGE table addresses and the partial tags
    for (int i = 1; i <= NHIST; i++)
      {
	GI[i] = gindex (usepc?pc:0, i, Fetch_phist, Fetch_ch_i);
	GTAG[i] = gtag (usepc?pc:0, i, Fetch_ch_t1, Fetch_ch_t2);
      }
    
    BI = pc & ((1 << LOGB) - 1);
    Tagepred ();
    pred_taken = tage_pred;
    BefIum = pred_taken;
    Ium = PredIum (pred_taken);
    if (countIum >= 0)
      pred_taken = Ium;
#ifdef LSTAT
    if (HitBank > 0)
      {
	LSUM = 0;
	LSUM = NLSTAT * (2 * gtable[HitBank][GI[HitBank]].ctr + 1);
	
	
	int lhist = L_Fetch_ghist[PCCLASS];
	LGI[0] = pc;
	
	for (int i = 1; i < NLSTAT; i++)
	  {
	    if (lm[i] < 32)
	      LGI[i] = (lhist & ((1 << lm[i]) - 1)) ^ (pc >> i);
	    else
	      LGI[i] = lhist ^ (pc >> i);
	    
	    int L = 32;
	    int A = 4 + i;
	    while (L > (LOGLSTAT - 1))
	      {
		
		LGI[i] ^= (LGI[i] >> A);
		L -= A;
		
	      }
	  }
	for (int i = 0; i < NLSTAT; i++)
	  {
	    LGI[i] <<= 1;
	    LGI[i] = (LGI[i] + pred_taken) & ((1 << LOGLSTAT) - 1);
	    LSUM += (2 * LStatCor[i][LGI[i]] + 1);
	  }
	if (Confid[(PCCLASS << 1) + pred_taken] >= 0)
	  pred_taken = (LSUM >= 0);
	
      }
#endif
    
    
    
    return pred_taken;
  }
  void FetchHistoryUpdate (uint32_t pc, OpType opType, bool taken,
			   uint32_t target)
  {

    if (opType == OPTYPE_BRANCH_COND)
      {
	UpdateIum (taken);
      }
    
    HistoryUpdate (pc, opType, taken, target, Fetch_phist, Fetch_ptghist,
		   Fetch_ch_i, Fetch_ch_t1, Fetch_ch_t2,
		   L_Fetch_ghist[PCCLASS]);
    
  }
  
  void HistoryUpdate (uint32_t pc, OpType opType, bool taken,
		      uint32_t target, int &X, int &Y, folded_history * H,
		      folded_history * G, folded_history * J, long &LH)
  {
    //special treatment for indirects and returnd: inherited from the indirect branch predictor submission
    int maxt = 1;
    
    if (opType == OPTYPE_CALL_DIRECT)
      maxt = 5;
    if ((opType == OPTYPE_INDIRECT_BR_CALL)) 
      maxt=4;
    if ((opType == OPTYPE_RET)) 
      maxt = 3;
    
    
    if (opType == OPTYPE_BRANCH_COND)
      {
	LH = (LH << 1) + taken;
      }
    
    
    
    int T = ((target ^ (target >> 3) ^ pc) << 1) + taken;
    int PATH = pc;
    for (int t = 0; t < maxt; t++)
      {
	bool aTAKEN = (T & 1);
	T >>= 1;
	bool PATHBIT = (PATH & 1);
	PATH >>= 1;
	//update  history
	Y--;
	ghist[Y & (HISTBUFFERLENGTH - 1)] = aTAKEN;
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

    //END UPDATE  HISTORIES
  }
  void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget, int updloop) 
  {
    static int blastrand = 0;
    OpType opType = OPTYPE_BRANCH_COND;
    uint32_t pc = PC & 0xffffff;
    bool taken = resolveDir;
    uint32_t target = branchTarget & 0x7f;
    FetchHistoryUpdate (pc, opType, taken, target);
    
    
    int8_t U[NHIST + 1];
    int8_t CTR[NHIST + 1];
    
    PtIumRetire--;
    tage_pred = Trans_TagePred[PtIumRetire & 255];
    HitBank = Trans_HitBank[PtIumRetire & 255];
    LSUM = Trans_LSUM[PtIumRetire & 255];
    
    /**/
    for(int myn = 0; myn <= updloop; myn++){
      
      if(myn == 0){
	//Recompute the indices and tags with the Retire history.
	for (int i = 1; i <= NHIST; i++)
	  {
	    
	    GI[i] = gindex (usepc?pc:0, i, Retire_phist, Retire_ch_i);
	    
	    GTAG[i] = gtag (usepc?pc:0, i, Retire_ch_t1, Retire_ch_t2);
	  }
	
	

	pred_taken = tage_pred;
	
	
#ifdef LSTAT
	
	if (HitBank > 0)
	  {
	    
	    int lhist = L_Retire_ghist[PCCLASS];
	    LGI[0] = pc;
	    
	    for (int i = 1; i < NLSTAT; i++)
	      {
		if (lm[i] < 32)
		  LGI[i] = (lhist & ((1 << lm[i]) - 1)) ^ (pc >> i);
		else
		  LGI[i] = lhist ^ (pc >> i);
		int L = 32;
		int A = 4 + i;
		while (L > (LOGLSTAT - 1))
		  {
		    
		    LGI[i] ^= (LGI[i] >> A);

		    
		    L -= A;
		    // A++;
		    
		  }
	      }
	    for (int i = 0; i < NLSTAT; i++)
	      {
		LGI[i] <<= 1;
		LGI[i] = (LGI[i] + pred_taken) & ((1 << LOGLSTAT) - 1);
		
	      }
	    
	    
	    bool LPRED = (LSUM >= 0);
	    if (pred_taken != LPRED)
	      if (pred_taken == taken)
		{
		  if (Confid[(PCCLASS << 1) + pred_taken] > -64)
		    Confid[(PCCLASS << 1) + pred_taken]--;
		}
	      else
		{
		  if (Confid[(PCCLASS << 1) + pred_taken] < 63)
		    Confid[(PCCLASS << 1) + pred_taken]++;
		  
		}

	    
	    if ((LPRED != taken) || (abs (LSUM) < (updatethreshold >> 5)))
	      {
		if (LPRED != taken)
		  updatethreshold++;
		else
		  updatethreshold--;
		
		for (int i = 0; i < NLSTAT; i++)
		  {
		    ctrupdate (LStatCor[i][LGI[i]], taken, CLSTAT);
		  }
		
	      }
	    
	  }
	
#endif
	BI = pc & ((1 << LOGB) - 1);
	Tagepred ();
	
	for (int i = 1; i <= NHIST; i++)
	  {
	    U[i] = gtable[i][GI[i]].u;
	    CTR[i] = gtable[i][GI[i]].ctr;
	  }
      } // end myn = 0
      
      bool ALLOC = ((tage_pred != taken) & (HitBank < NHIST));
      {
	if(myn == 1){
	  // try to allocate a  new entries only if TAGE prediction was wrong
	  if (HitBank > 0)
	    {
	      // Manage the selection between longest matching and alternate matching
	      // for "pseudo"-newly allocated longest matching entry
	      
	      bool PseudoNewAlloc = (abs (2 * CTR[HitBank] + 1) <= 1);
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
	      int T = tage_alloc;
	      
	      int A = 1;
	      if (HitBank == 0)
		if ((MYRANDOM () & 127) < 32)
		  A = 2;
	      
	      for (int i = HitBank + A; i <= NHIST; i += 1)
		{
		  if (U[i] == 0)
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
		    {
		      TICK++;
		      if (LOGB > 14) 			TICK++;
		      srand(blastrand);
		      blastrand = random(); 
		      if ((blastrand & 63)==0) gtable[i][GI[i]].u--;
                  
		    }
		  
		}
	    }
	} // end myn = 1
	
	if(myn == 0){
	  //manage the u  bit
	  if ((TICK >= (1 << LOGTICK)))
	    {
	      TICK = 0;
	      // reset the u bit
	      for (int i = 1; i <= NHIST; i++)
		for (int j = 0; j < (1 << logg[i]); j++)
		  gtable[i][j].u >>= 1;
	      
	    }
	} // end myn = 0
	
      }

      if(myn!= 0){
	//update the prediction
	
	if (HitBank > 0)
	  {
	    ctrupdate (CTR[HitBank], taken, CWIDTH);
	    gtable[HitBank][GI[HitBank]].ctr = CTR[HitBank];
	    
	    
	    // acts as a protection 
	    if (!U[HitBank])
	      {
		if (AltBank > 0)
		  {
		    ctrupdate (CTR[AltBank], taken, CWIDTH);
		    gtable[AltBank][GI[AltBank]].ctr = CTR[AltBank];
		  }
		
		
		if (AltBank == 0)
		  baseupdate (taken);
	      }
	  }
	else
	  baseupdate (taken);

      } // end myn!= 0

      if(myn == 0){
	// update the u counter
	if (HitBank > 0)
	  if (LongestMatchPred != alttaken)
	    {
	      if (LongestMatchPred == taken)
		{
		  if (gtable[HitBank][GI[HitBank]].u < 1)
		    gtable[HitBank][GI[HitBank]].u++;
		  
		}
	    }
      }
      //END PREDICTOR UPDATE
      
    }

//  UPDATE RETIRE HISTORY  
    HistoryUpdate (pc, opType, taken, target, Retire_phist, Retire_ptghist,
		   Retire_ch_i, Retire_ch_t1, Retire_ch_t2,
		   L_Retire_ghist[PCCLASS]);
  }

  void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget){

    uint32_t pc = PC & 0xffffff;
    bool taken = 1;
    uint32_t target = branchTarget & 0x7f;
    FetchHistoryUpdate (pc, opType, taken, target);
    
    //  UPDATE RETIRE HISTORY  
    HistoryUpdate (pc, opType, taken, target, Retire_phist, Retire_ptghist,
		   Retire_ch_i, Retire_ch_t1, Retire_ch_t2,
		   L_Retire_ghist[PCCLASS]);
}

  // Contestants can define their own functions below

};



/***********************************************************/

inline int rangf(){
  static int lastrand = 0;
  srand(lastrand);
  lastrand = rand();
  int r =  lastrand%5;
  if(r == 0) return 0;
  if(r == 1) return 2;
  return 1;
}

inline int poisson(double A) {
  static int lastrand = 0;
  int k = 0;
  int maxK = 2;
  while (1) {
    srand(lastrand);
    lastrand = rand();
    double U_k= lastrand / (RAND_MAX + 1.0);
    A *= U_k;
    if (k >= maxK || A < exp(-1.0)) {
      break;
    }
    k++;
  }
  return k;
}



#define tagecount 32
#define oobe_hist_length 16

class PREDICTOR{
 public:
  
  tage  *brpred[tagecount];
  int predDira[tagecount];
  bool oobe_hist[tagecount][oobe_hist_length];
  int oobe_hist_inpoint[tagecount];
  int oobe[tagecount];
  
  PREDICTOR(void){
    
    for(int o=0; o<tagecount;o++){
      
      for(int p=0; p<oobe_hist_length; p++){
	oobe_hist[o][p] = 0;
      }
      
      oobe[o] = 0;
      oobe_hist_inpoint[o] = oobe_hist_length - 1;
    }
    
    //tage(int counter, int numtable, int tablesize, int min, int max, int talloc)
    brpred[0] = new tage(3, 24, 22, 9, 2000, 38, true);
    brpred[1] = new tage(3, 32, 22, 7, 30000, 38, true);
    brpred[2] = new tage(3, 30, 22, 9, 10000, 38, true);
    brpred[3] = new tage(3, 29, 22, 6, 5000, 38, true);
    brpred[4] = new tage(3, 28, 22, 8, 4000, 38, true);
    brpred[5] = new tage(3, 27, 22, 10, 3000, 38, true);
    brpred[6] = new tage(3, 25, 22, 6, 2500, 38, true);
    brpred[7] = new tage(3, 38, 22, 5, 100000, 38, true);

    brpred[8] = new tage(3, 38, 22, 12, 100000, 38, false);
    brpred[9] = new tage(3, 32, 22, 10, 30000, 38, false);
    brpred[10] = new tage(3, 30, 22, 9, 10000, 38, false);
    brpred[11] = new tage(3, 29, 22, 11, 5000, 38, false);
    brpred[12] = new tage(3, 28, 22, 10, 4000, 38, false);
    brpred[13] = new tage(3, 27, 22, 13, 3000, 38, false);
    brpred[14] = new tage(3, 25, 22, 11, 2500, 38, false);
    brpred[15] = new tage(3, 24, 22, 12, 2000,38, false);
    
    brpred[16] = new tage(3, 23, 22, 4, 2000, 38, true);
    brpred[17] = new tage(3, 23, 22, 5, 1800, 38, true);
    brpred[18] = new tage(3, 22, 22, 3, 1600, 38, true);
    brpred[19] = new tage(3, 22, 22, 8, 1500, 38, true);
    brpred[20] = new tage(3, 21, 22, 9, 1400, 38, true);
    brpred[21] = new tage(3, 21, 22, 10, 1300, 38, true);
    brpred[22] = new tage(3, 20, 22, 6, 1200, 38, true);
    brpred[23] = new tage(3, 20, 22, 7, 1000, 38, true);

    brpred[24] = new tage(3, 20, 22, 9, 100000, 38, false);
    brpred[25] = new tage(3, 20, 22, 10, 85000, 38, false);
    brpred[26] = new tage(3, 20, 22, 11, 70000, 38, false);
    brpred[27] = new tage(3, 20, 22, 13, 55000, 38, false);
    brpred[28] = new tage(3, 20, 22, 12, 40000, 38, false);
    brpred[29] = new tage(3, 20, 22, 8, 25000, 38, false);
    brpred[30] = new tage(3, 20, 22, 10, 10000, 38, false);
    brpred[31] = new tage(3, 20, 22, 7, 8000, 38, false);

  }
  
  bool    GetPrediction(UINT32 PC){
    int predDirint = 0;
    for(int i = 0; i < tagecount; i++){
      int weight;
      predDira[i] = brpred[i]->GetPrediction(PC);
      weight = (predDira[i]==0)?(oobe[i]-oobe_hist_length):(oobe_hist_length-oobe[i]);
      predDirint += weight;
    }
    return predDirint >=0;
  }


  void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
    for(int i = 0; i < tagecount; i++){
      oobe[i] = oobe[i] - oobe_hist[i][(oobe_hist_inpoint[i] + oobe_hist_length - 1)%oobe_hist_length] + (predDira[i]!=resolveDir); 
      oobe_hist[i][oobe_hist_inpoint[i]] = (predDira[i]!=resolveDir);
      if(oobe_hist_inpoint[i] == 0)
	oobe_hist_inpoint[i] = oobe_hist_length - 1;
      else
	oobe_hist_inpoint[i]--;
      
      brpred[i]->UpdatePredictor(PC, resolveDir, predDir, branchTarget, rangf());
    }
  }
  
  
  void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget){
    
    if ((opType < 8) && (opType != 6) && (opType >=3)){ 
      for(int i = 0; i < tagecount; i++){
	brpred[i]->TrackOtherInst(PC, opType, branchTarget);
      }
    }
    
  }
};

#endif

