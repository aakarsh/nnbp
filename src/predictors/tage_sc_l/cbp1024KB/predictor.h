/**
 * uncomment to get a realistic predictor within the 256 Kbits limit
 * with only 12 1024 entries tagged tables in the TAGE predictor
 * and a global history and single local history GEHL statistical corrector:
 * total misprediction numbers 2.430 MPKI
 */
//#define REALISTIC

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

//To get the predictor storage budget on stderr  uncomment the next line
#define PRINTSIZE

#include "utils.h"


#include <inttypes.h>
#include <math.h>

#define LOOPPREDICTOR	        // Loop predictor enable
#define NNN 1			// Number of entries allocated on a TAGE misprediction

// Number of tagged tables

/**
 * Hard coded value for the number of tag tables .
 */
#ifndef REALISTIC
#define NHIST 17
#else
#define NHIST 13
#endif

#define HYSTSHIFT 2 // bimodal hysteresis shared by 4 entries

/**
 * Number of entries in bi-modal predictor = 2^LOG_BIMODAL_NENTRIES = (* 2 16384)
 * Base predictor  PC indexed 2-bit counter bi-modal table
 */
#define LOG_BIMODAL_NENTRIES 17 // log of number of entries in bimodal predictor



#define PERCWIDTH 8 //Statistical coorector maximum counter width

//The statistical corrector components

//global branch GEHL
#ifdef REALISTIC
#define LOGGNB 10
#else
#define LOGGNB 9
#endif
#define GNB 4
int Gm[GNB] = {16,11, 6,3};
int8_t GGEHLA[GNB][(1 << LOGGNB)];
int8_t *GGEHL[GNB];

//large local history
#ifdef REALISTIC
#define LOGLNB 10
#define LNB 4
int Lm[LNB] = {16,11,6,3};
int8_t LGEHLA[LNB][(1 << LOGLNB)];
int8_t *LGEHL[LNB];
#else
#define LOGLNB 10
#define LNB 3
int Lm[LNB] = {11,6,3};
int8_t LGEHLA[LNB][(1 << LOGLNB)];
int8_t *LGEHL[LNB];
#endif
#define  LOGLOCAL 8
#define NLOCAL (1<<LOGLOCAL)
#define INDLOCAL (PC & (NLOCAL-1))
long long L_shist[NLOCAL];


// small local history
#define LOGSNB 9
#define SNB 4

int Sm[SNB] =  {16,11, 6, 3};
int8_t SGEHLA[SNB][(1 << LOGSNB)];
int8_t *SGEHL[SNB];
#define LOGSECLOCAL 4
#define NSECLOCAL (1<<LOGSECLOCAL)	//Number of second local histories
#define INDSLOCAL  (((PC ^ (PC >>5))) & (NSECLOCAL-1))
long long S_slhist[NSECLOCAL];


#define LOGTNB 9
#define TNB 3
int Tm[TNB] =  {11, 6, 3};
int8_t TGEHLA[TNB][(1 << LOGTNB)];
int8_t *TGEHL[TNB];
#define INDTLOCAL  (((PC ^ (PC >>3))) & (NSECLOCAL-1))	// differen hash for thehistory
long long T_slhist[NSECLOCAL];

//return-stack associated history component
/**
 * Return Stack Associated History
 */
#define PNB 4
#define LOGPNB 9
int Pm[PNB] ={16,11,6,3};
int8_t PGEHLA[PNB][(1 << LOGPNB)];
int8_t *PGEHL[PNB];
long long HSTACK[16];
int pthstack;


//parameters of the loop predictor

/** Size of loop predictor in use */
#define LOG_LOOP_PRED_NENTRIES 9
#define WIDTHNBITERLOOP 12	// we predict only loops with less than 1K iterations
#define LOOPTAG 17		//tag width in the loop predictor


//update threshold for the statistical corrector
#ifdef REALISTIC
#define LOGSIZEUP 0
#else
#define LOGSIZEUP 5
#endif
int Pupdatethreshold[(1 << LOGSIZEUP)];	//size is fixed by LOGSIZEUP
#define INDUPD (PC & ((1 << LOGSIZEUP) - 1))

// The three counters used to choose between TAGE ang SC on High Conf TAGE/Low Conf SC
int8_t  FirstH;
int8_t  SecondH;
int8_t  ThirdH;

#define CONFWIDTH 7	//for the counters in the choser

#define PHISTWIDTH 32		// Width of the path history used in TAGE

/**
 * Useful counter are incremented whenever the prediction was correct
 * and the alternate predictor was incorrect.
 */
#define UWIDTH 2   // u counter width on TAGE

#define CWIDTH 3   // predictor counter width on the TAGE tagged tables

#define HISTBUFFERLENGTH 8192	// we use a 4K entries history buffer to store the branch history


//the counter(s) to chose between longest match and alternate prediction on TAGE when weak counters

#ifdef REALISTIC
#define LOGSIZEUSEALT 0
#else
#define LOGSIZEUSEALT 10
#endif
#define SIZEUSEALT  (1<<(LOGSIZEUSEALT))
#define INDUSEALT (PC & (SIZEUSEALT -1))

int8_t use_alt_on_na[SIZEUSEALT][2];


long long GHIST;

//The two BIAS tables in the SC component
#define LOGBIAS 13
int8_t Bias[(1<<(LOGBIAS+1))];
#define INDBIAS (((PC<<1) + pred_inter) & ((1<<(LOGBIAS+1)) -1))
int8_t BiasSK[(1<<(LOGBIAS+1))];
#define INDBIASSK ((((PC^(PC>>LOGBIAS))<<1) + pred_inter) & ((1<<(LOGBIAS+1)) -1))

bool HighConf;
int LSUM;
int8_t BIM;


/**
 *  Utility class for index computation this is the cyclic shift
 *  register for folding a long global history into a smaller number
 *  of bits; see P. Michaud's PPM-like predictor at CBP-1 
 */
class folded_history
{
public:

  /** Computed resgister which keeps the result of folding registers as updates go through   */
  unsigned comp;

  /** Compressed Length */
  int CLENGTH;

  /** Original Length */
  int OLENGTH;

  /** Remainder that doesnt fit inside some factor of compressed length  */
  int OUTPOINT;

  folded_history (){ }


  void init (int original_length, int compressed_length, int N)
  {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;

  }

  /**
   * Update comp with new history length
   */
  void update (uint8_t * h, int PT)
  {
    comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
    comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp = (comp) & ((1 << CLENGTH) - 1);
  }

};

#ifdef LOOPPREDICTOR


/**
//loop predictor entry
 */
class lentry {
public:
  uint16_t NbIter;		// 10 bits  - iteration count
  uint8_t confid;		// 4bits    -
  uint16_t CurrentIter;		// 10 bits  -

  uint16_t TAG;			// 10 bits  - Partial tag on 10 bits
  uint8_t age;			// 4 bits   - Replace when age is 0
  bool dir;			// 1 bit    - direction

  //39 bits per entry

    lentry () {
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
  /**
   * singed 8 bit counter
   */
  int8_t hyst;

  /**
   * Start off assuming pred not taken.
   */
  int8_t pred;

  bentry () {
    pred = 0;
    hyst = 1;
  }

};

/**
 * Global Table entry
 */
class gentry			// TAGE global table entry
{
public:
  /**
   * 8-bit signed integer counter
   */
  int8_t ctr;

  /**
   * Tag field in global entry.
   */
  uint tag;

  /**
   * u - useful counter
   */
  int8_t u;

  /**
   * Counter, Tag and  useful start off empty
   */
  gentry () {
    ctr = 0;
    tag = 0;
    u = 0;
  }

};


// for the reset of the u counter
int TICK;


uint8_t ghist[HISTBUFFERLENGTH];
int ptghist;

// Path history
long long phist;

// Utility for computing TAGE indices
folded_history ch_i[NHIST + 1];

// Utility for computing TAGE tags
folded_history ch_t[2][NHIST + 1];

// For the TAGE predictor
/**
 * bimodal TAGE table
 */
bentry *btable;

/**
 * Tagged TAGE tables
 */
gentry *gtable[NHIST + 1];	


#ifdef REALISTIC

/**
 * History lengths for different tables.
 */
int history_lengths[NHIST+1]={0,8,12,18,27,40,60,90,135,203,305,459,690,890,1024};


/**
 * Tag width for different tables
 */
int tag_width[NHIST + 1]  ={0,8, 9, 9,10,10,11,11,12,12,13,13,15,16,16};

/**
 * Log of number of entries in different tables
 */
int log_nentry_tables[NHIST + 1]={0,32,32,32,32,32,32,32,32,32,32,32,32,32,64};

#else

/**
 * History lenghts for different tag tables.
 */
int history_lengths[NHIST+1]={0,7,13,18,25,35,55,69,105,155,230,354,479,642,1012,1347,1347,2048};

/**
 * Width of tags for different tables
 */
int tag_width[NHIST + 1]={3,12,12,13,13,13,14,14,15,15,15,16,17,19,19,19,19,22 };



/**
 * Log of number of entries in different tables
 */
int log_nentry_tables[NHIST + 1]={0,11,11,11,10,10,10,12,12,12,9,9,9,8,7,7,7,10};

#endif

/**
 * Contains the index computation for current PC into each of the history tables.
 */
int GI[NHIST + 1];		// indexes to the different tables are computed only once

/**
 * Contains the tag computation for the current PC into each of the history tables.
 */
uint GTAG[NHIST + 1];	// tags for the different tables are computed only once

/**
 * Contains the index into the bi-modal table for the current PC
 */
int BI;				// index of the bimodal table

bool pred_taken;		// prediction
bool alttaken;			// alternate TAGEprediction

/**
 * Global variable which stores the prediction going to be made by
 * TAGE for current value of PC
 */
bool tage_pred;			// TAGE prediction

bool LongestMatchPred;

/**
 * Contains the bank number for the longest bank where with matching
 * history.
 */
int HitBank;


/**
 * Contains the bank number for next longest bank with matching tag.
 */
int AltBank;			// alternate matching bank

int Seed;			// for the pseudo-random number generator

bool pred_inter;


#ifdef LOOPPREDICTOR
lentry *ltable;			//loop predictor table
//variables for the loop predictor
bool predloop;			// loop predictor prediction
int LIB;
int LI;
int LHIT;			// hitting way in the loop predictor
int LTAG;			// tag on the loop predictor
bool LVALID;			// validity of the loop predictor prediction
int8_t WITHLOOP;		// counter to monitor whether or not loop prediction is beneficial
#endif


/**
 *
 * Given a table bank number it computes the size of entry for that bank.
 *
 * An entry's width is broken down to :
 * - size of its predictor counter 
 * - the size fo the tag used for the entry
 * - the size of useful counter
 *
 */
static inline int entry_width(int bank_number)
{
    int predictor_counter_width = CWIDTH;
    int useful_counter_width = UWIDTH;
    int entry_tag_width = tag_width[bank_number];

    // Width of entry in this table
    return (predictor_counter_width + useful_counter_width + entry_tag_width);
}

/**
 * Entries inside different banks have different sizes as given in in
 * the llog array.
 */
static inline int num_entries(int bank_number)
{
  return  (1 << (log_nentry_tables[bank_number]));
}

/**
 * Each table's entry size and number of entries is hand tuned.
 */
static inline int tage_table_size(int bank_number)
{
  return num_entries(bank_number) * entry_width(bank_number);
}

/**
 * Computes sum of all tage tables
 */
static inline int tage_table_size_all()
{
  
  int tage_size = 0;
  
  for (int i = 1; i <= NHIST; i += 1) {
    tage_size += tage_table_size(i);
  }
  
  return tage_size;
}


/**
 * Maximum history length used by the the buffer.
 */
static int inline max_history_length() {
  return history_lengths[NHIST];
}


/**
 * Computes and prints the size of the bi-modal table.
 */
int predictor_size () {

  int storage_size = 0;

  storage_size += tage_table_size_all();
  fprintf(stderr, "%d\n",SIZEUSEALT);
  storage_size += 2 * (SIZEUSEALT) * 4;
  storage_size += (1 << LOG_BIMODAL_NENTRIES ) + (1 << (LOG_BIMODAL_NENTRIES - HYSTSHIFT));
  storage_size += max_history_length();
  storage_size += PHISTWIDTH;
  storage_size += 10 ; // the TICK counter

  fprintf(stderr, "(TAGE %d bytes)\n", storage_size);

  // No idea how the loop predictor is supposed to work.
  int inter = 0;

#ifdef LOOPPREDICTOR
  inter = (1 << LOG_LOOP_PRED_NENTRIES) * (2 * WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1);
  fprintf (stderr, "(LOOP %d)\n", inter);
  storage_size += inter;
#endif

  inter = 8 * (1 << LOGSIZEUP) ; //the update threshold counters
  inter += (PERCWIDTH) * 4 * (1 << (LOGBIAS));
  inter += (GNB-2) * (1 << (LOGGNB)) * (PERCWIDTH - 1) + (1 << (LOGGNB-1))*(2*PERCWIDTH-1);
  inter += (LNB-2) * (1 << (LOGLNB)) * (PERCWIDTH - 1) + (1 << (LOGLNB-1))*(2*PERCWIDTH-1);

#ifndef REALISTIC
  inter += (SNB-2) * (1 << (LOGSNB)) * (PERCWIDTH - 1) + (1 << (LOGSNB-1))*(2*PERCWIDTH-1);
  inter += (TNB-2) * (1 << (LOGTNB)) * (PERCWIDTH - 1) + (1 << (LOGTNB-1))*(2*PERCWIDTH-1);
  inter += (PNB-2) * (1 << (LOGPNB)) * (PERCWIDTH - 1) + (1 << (LOGPNB-1))*(2*PERCWIDTH-1);
  inter += 16*16; // the history stack
  inter += 4;     // the history stack pointer
  inter += 16;    // global histories for SC
  inter += NSECLOCAL * (Sm[0]+Tm[0]);
#endif

  inter += NLOCAL * Lm[0];
  inter += 3 * CONFWIDTH;    // the 3 counters in the choser

  storage_size += inter;

  fprintf (stderr, "(SC %d)\n", inter);
  fprintf (stdout, "(TOTAL %d)\n ", storage_size);


  return (storage_size);
}




class PREDICTOR
{
public:
  PREDICTOR (void) {

    reinit ();

#ifdef PRINTSIZE
    predictor_size ();
#endif

  }


  void reinit () {

    // hate this way of programming
#ifdef LOOPPREDICTOR
    ltable = new lentry[1 << (LOG_LOOP_PRED_NENTRIES)];
#endif

    /**
     * For each history length create a nentries
     * (table_i) =   2^log_nentry_tables[i]
     */
    for (int i = 1; i <= NHIST; i++) {
      gtable[i] = new gentry[1 << (log_nentry_tables[i])];
    }

    /**
     * Initialized the bimodal table with 2^LOG_BIMODAL_NENTRIES entries
     */
    btable = new bentry[1 << LOG_BIMODAL_NENTRIES ];

    for (int i = 1; i <= NHIST; i++) {

      ch_i[i].init (history_lengths[i] /* original length */,
                    (log_nentry_tables[i]), /** foldted length */
                    i - 1);
	ch_t[0][i].init (ch_i[i].OLENGTH, tag_width[i], i);
	ch_t[1][i].init (ch_i[i].OLENGTH, tag_width[i] - 1, i + 2);
    }

#ifdef LOOPPREDICTOR
    LVALID = false;
    WITHLOOP = -1;
#endif
    Seed = 0;

    TICK = 0;
    phist = 0;
    Seed = 0;

    for (int i = 0; i < HISTBUFFERLENGTH; i++)
      ghist[0] = 0;

    ptghist = 0;

    for (int i = 0; i < (1 << LOGSIZEUP); i++)
      Pupdatethreshold[i] = 35;

    for (int i = 0; i < GNB; i++)
      GGEHL[i] = &GGEHLA[i][0];
    for (int i = 0; i < LNB; i++)
      LGEHL[i] = &LGEHLA[i][0];

    for (int i = 0; i < GNB; i++)
      for (int j = 0; j < ((1 << LOGGNB) - 1); j++) {
	  if (!(j & 1)) {
	      GGEHL[i][j] = -1;
          }
      }

    for (int i = 0; i < LNB; i++)
      for (int j = 0; j < ((1 << LOGLNB) - 1); j++) {
	  if (!(j & 1)) {
	      LGEHL[i][j] = -1;
          }
      }

#ifndef REALISTIC
for (int i = 0; i < SNB; i++)
      SGEHL[i] = &SGEHLA[i][0];
    for (int i = 0; i < TNB; i++)
      TGEHL[i] = &TGEHLA[i][0];
    for (int i = 0; i < PNB; i++)
      PGEHL[i] = &PGEHLA[i][0];
    for (int i = 0; i < SNB; i++)
      for (int j = 0; j < ((1 << LOGSNB) - 1); j++) {
	  if (!(j & 1)) {
	      SGEHL[i][j] = -1;
          }
      }

    for (int i = 0; i < TNB; i++)
      for (int j = 0; j < ((1 << LOGTNB) - 1); j++) {
	  if (!(j & 1)) {
	      TGEHL[i][j] = -1;
          }
      }

    for (int i = 0; i < PNB; i++)
      for (int j = 0; j < ((1 << LOGPNB) - 1); j++) {
        if (!(j & 1)) {
          PGEHL[i][j] = -1;
        }
      }
#endif

    /**
     * Initialize the base bi-modal table with 2^LOG_BIMODAL_NENTRIES
     * entries with hysteresis 1 and prediction 0
     */
    for (int i = 0; i < (1 << LOG_BIMODAL_NENTRIES ); i++)
      {
	btable[i].pred = 0;
	btable[i].hyst = 1;
      }


    for (int j = 0; j < (1 << (LOGBIAS + 1)); j++)
      Bias[j] = (j & 1) ? 15 : -16;

    for (int j = 0; j < (1 << (LOGBIAS + 1)); j++)
      BiasSK[j] = (j & 1) ? 15 : -16;


    for (int i = 0; i < NLOCAL; i++)
      {
	L_shist[i] = 0;
      }


    for (int i = 0; i < NSECLOCAL; i++) {
	S_slhist[i] = 0;
    }

    GHIST = 0;

    /**
     * Aternate predictors initialized to zero
     */
    for (int i = 0; i < SIZEUSEALT; i++) {
	use_alt_on_na[i][0] = 0;
	use_alt_on_na[i][1] = 0;
    }

    TICK = 0;
    ptghist = 0;
    phist = 0;

  }



  /**
   * Simply select the last LOG_BIMODAL_NENTRIES number of bits of PC
   * - index function for the bimodal table
   */

  int bindex(uint32_t PC)
  {
    return ((PC) & ((1 << (LOG_BIMODAL_NENTRIES )) - 1));
  }


// the index functions for the tagged tables uses path history as in the OGEHL predictor
//F serves to mix path history: not very important impact
  /**
   * F is a magical mix of path history ?
   *
   * A    - history register
   * size - size of the path history
   * bank - bank number
   */
  int F (long long A, int size, int bank)
  {
    int A1, A2;

    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << log_nentry_tables[bank]) - 1));
    A2 = (A >> log_nentry_tables[bank]);
    A2 = ((A2 << bank) & ((1 << log_nentry_tables[bank]) - 1)) + (A2 >> (log_nentry_tables[bank] - bank));
    A = A1 ^ A2;
    A = ((A << bank) & ((1 << log_nentry_tables[bank]) - 1)) + (A >> (log_nentry_tables[bank] - bank));
    return (A);
  }

  // gindex computes a full hash of PC, ghist and phist

  /**
   * Start with PC. Compute index for bank?
   */
  int gindex (unsigned int PC, int bank, long long hist, folded_history * ch_i)
  {

    int index;

    // if size of bank is greater than history width then use full history width
    // else use bank width

    /**
     * Cut off path history at pattern history width.
     */
    int M = (history_lengths[bank] > PHISTWIDTH) ? PHISTWIDTH : history_lengths[bank];

    // PC XOR
    index = PC ^ (PC >> (abs(log_nentry_tables[bank] - bank) + 1)) ^ ch_i[bank].comp ^ F (hist, M, bank);

    return (index & ((1 << (log_nentry_tables[bank])) - 1));

  }

  /**
   * Takes two folded histories and the bank number we need the tag
   * for and returns tag for comparison
   */
  uint16_t gtag (unsigned int PC, int bank, folded_history * ch0, folded_history * ch1)
  {
    int tag = PC ^ ch0[bank].comp ^ (ch1[bank].comp << 1);
    /** Ensure only tag width amount*/
    return (tag & ((1 << tag_width[bank]) - 1));    
  }


  /**
   * Update saturating counter.
   * 1. increment if taken , decrement if not taken
   * 2. If counter has reached limit for bit field length then leave ctr as is.
   */
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

    
#ifdef LOOPPREDICTOR

  /**
   * Select the LAST LOGL bits from  PC
   */
  int lindex (uint32_t PC)
  {
    // LOGL - Size of loop predictor
    return ((PC & ((1 << (LOG_LOOP_PRED_NENTRIES - 2)) - 1)) << 2);
  }


// loop prediction: only used if high confidence
// skewed associative 4-way
// At fetch time: speculative

#define CONFLOOP 15

  /**
   *
   */
  static inline int compute_loop_tag(uint32_t PC)
  {
    int pc_loop_tag = (PC >> (LOG_LOOP_PRED_NENTRIES - 2)) & ((1 << 2 * LOOPTAG) - 1);
    pc_loop_tag ^= (pc_loop_tag >> LOOPTAG);
    pc_loop_tag = (pc_loop_tag & ((1 << LOOPTAG) - 1));
    return pc_loop_tag;
  }

  /**
   * Predict whether loop is taken or not for a given value of PC.
   */
  bool getloop(uint32_t PC)
  {

    LHIT = -1;     // hitting way in the predictor
    LI = lindex (PC);
    LIB  = ((PC >> (LOG_LOOP_PRED_NENTRIES - 2)) & ((1 << (LOG_LOOP_PRED_NENTRIES - 2)) - 1));
    LTAG = compute_loop_tag(PC);

    // Checking inside of a four way associative index - i is the block

    for (int i = 0; i < 4; i++) {

	int index = (LI ^ ((LIB >> i) << 2)) + i;

	if (ltable[index].TAG == LTAG) {

          // store the hitting way in the predictor
          LHIT = i;

          // set validity of loop
          LVALID = ((ltable[index].confid == CONFLOOP)
                    || (ltable[index].confid * ltable[index].NbIter > 128));

          if (ltable[index].CurrentIter + 1 == ltable[index].NbIter) {
            return (!(ltable[index].dir));
          } else {
            // If the loop iteration number is reached return dir
            return ((ltable[index].dir));
          }
        }
    }

    LVALID = false;
    return (false);

  }


  /**
   * Cleara loop entry so that it will not be used.
   */
  static inline void loop_entry_free(lentry* ltable,int index)
  {
    ltable[index].NbIter = 0;
    ltable[index].age = 0;
    ltable[index].confid = 0;
    ltable[index].CurrentIter = 0;
  }


  /**
   * If no hit and prediction wrong then allocate an entry.
   */
  void loopupdate (uint32_t PC, bool Taken, bool ALLOC) {

    if (LHIT >= 0) { // if there was a hit

      int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;

      // Already a hit
      if (LVALID) {
        // free the entry
        if (Taken != predloop) {
          loop_entry_free(ltable,index);
          return;
        } else if ((predloop != tage_pred) || ((next_pseudo_random() & 7) == 0))
          if (ltable[index].age < CONFLOOP)
            ltable[index].age++;
      }

      ltable[index].CurrentIter++;
      ltable[index].CurrentIter &= ((1 << WIDTHNBITERLOOP) - 1);

      //loop with more than 2** WIDTHNBITERLOOP iterations are not treated correctly; but who cares :-)

      if (ltable[index].CurrentIter > ltable[index].NbIter) {
        ltable[index].confid = 0;
        ltable[index].NbIter = 0;
        //treat like the 1st encounter of the loop
      }

      if (Taken != ltable[index].dir)
	  {
	    if (ltable[index].CurrentIter == ltable[index].NbIter)
	      {
                /**
                 * If the loop was correctly predicted and loop
                 * confidence has not be saturated increment the
                 * confidence.
                 */
		if (ltable[index].confid < CONFLOOP)
		  ltable[index].confid++;

                // Skip over loops that are too small to know if they are loops
                // just do not predict when the loop count is 1 or 2
                // clear short loop information
		if (ltable[index].NbIter < 3){
                    // free the entry
		    ltable[index].dir = Taken;
		    ltable[index].NbIter = 0;
		    ltable[index].age = 0;
		    ltable[index].confid = 0;
                }
	      }
	    else {
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
    // if no hit and wrong prediction allocate an entry in loop predictor.
    // randomly turn off entry?
    else if (ALLOC) {
	uint32_t X = next_pseudo_random () & 3;

	if ((next_pseudo_random () & 3) == 0)
	  for (int i = 0; i < 4; i++)
	    {
	      int LHIT = (X + i) & 3;
	      int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;

              // When age goes to zero
	      if (ltable[index].age == 0)
		{
		  ltable[index].dir = !Taken;
                  // most of mispredictions are on last iterations
                  // Change to confidence 0 age 7 entry
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

  /**
   * Returns the prediction from bimodal table
   */
  bool getbim ()
  {
    BIM = (btable[BI].pred << 1) + (btable[BI >> HYSTSHIFT].hyst);
    HighConf = (BIM == 0) || (BIM == 3);
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

  /**
   * XOR - Seed with path history.
   */
  int next_pseudo_random ()
  {
    Seed++;
    Seed ^= phist;
    Seed = (Seed >> 21) + (Seed << 11);
    return (Seed);
  };


  //  TAGE PREDICTION: Same code at fetch or retire time but the index and tags must recomputed

  void Tagepred (UINT32 PC)
  {
    
    HitBank = 0;
    AltBank = 0;

    /**
     * For each table compute the tag and index
     */
    for (int i = 1; i <= NHIST; i++) {
      GI[i]   = gindex (PC, i, phist, ch_i);
      GTAG[i] = gtag (PC, i, ch_t[0], ch_t[1]);
    }

    /**
     * Index into bi-modal table is simple last LOG_BIMODAL_NENTRIES bits of PC
     */
    BI = PC & ((1 << LOG_BIMODAL_NENTRIES ) - 1);


    //Look for the bank with longest matching history
    /**
     * Start with longest history.
     *
     */
    for (int i = NHIST; i > 0; i--) // Check longest history first
      {

        /**
         * Tag at index GI[i] matches the TAG computed using folded
         * history and PC for this history size
         */
	if (gtable[i][GI[i]].tag == GTAG[i]) {
            // Record the table which was a hit
	    HitBank = i;

            // Record the fact that ctr at this table was positive
	    LongestMatchPred = (gtable[HitBank][GI[HitBank]].ctr >= 0);

	    break;
	  }
      }

    // Look for the alternate bank
    /**
     * Alternate hit bank is the next longest matching history.
     */
    for (int i = HitBank - 1; i > 0; i--) {
      if (gtable[i][GI[i]].tag == GTAG[i]) {
        AltBank = i;
        break;
      }
    }

    // Computes the prediction and the alternate prediction

    if (HitBank > 0) // was a hit
      {
	if (AltBank > 0) // had alternate hit
	  alttaken = (gtable[AltBank][GI[AltBank]].ctr >= 0);
	else
	  alttaken = getbim ();

        // if the entry is recognized as a newly allocated entry and
        // USE_ALT_ON_NA is positive  use the alternate prediction

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

      }
    else
      {
	alttaken = getbim ();
	tage_pred = alttaken;
	LongestMatchPred = alttaken;
      }


  }


  /**
   * Prints the prediction for given program counter.
   */
  bool GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN) 
  {
    // Computes the TAGE table addresses and the partial tags

    Tagepred(PC);
    pred_taken = tage_pred;

#ifdef LOOPPREDICTOR
    predloop = getloop (PC);	// loop prediction
    pred_taken = ((WITHLOOP >= 0) && (LVALID)) ? predloop : pred_taken;
#endif

    pred_inter = pred_taken;

    //Compute the SC prediction

    // begin to bias the sum towards TAGE predicted direction

    LSUM = 1;
    LSUM += 2 * (GNB + PNB);

#ifndef REALISTIC
    LSUM += 2*(SNB+ LNB+TNB);
#endif

    if (!pred_inter)
      LSUM = -LSUM;

    //integrate BIAS prediction
    int8_t ctr = Bias[INDBIAS];
    LSUM += (2 * ctr + 1);
    ctr = BiasSK[INDBIASSK];
    LSUM += (2 * ctr + 1);

    //integrate the GEHL predictions
    LSUM += Gpredict ((PC<<1)+pred_inter, GHIST, Gm, GGEHL, GNB, LOGGNB);
    LSUM += Gpredict (PC, L_shist[INDLOCAL], Lm, LGEHL, LNB, LOGLNB);

#ifndef REALISTIC
    LSUM += Gpredict (PC, S_slhist[INDSLOCAL], Sm, SGEHL, SNB, LOGSNB);
    LSUM += Gpredict (PC, T_slhist[INDTLOCAL], Tm, TGEHL, TNB, LOGTNB);
    LSUM += Gpredict (PC, HSTACK[pthstack], Pm, PGEHL, PNB, LOGPNB);
#endif

    bool SCPRED = (LSUM >= 0);

    // chose between the SC output and the TAGE + loop  output

    if (pred_inter != SCPRED) {
      //Choser uses TAGE confidence and |LSUM|
      pred_taken = SCPRED;
      if (HighConf) {
	    if ((abs (LSUM) < Pupdatethreshold[INDUPD] / 3))
	      pred_taken = (FirstH < 0) ? SCPRED : pred_inter;
	    else if ((abs (LSUM) < 2 * Pupdatethreshold[INDUPD] / 3))
	      pred_taken = (SecondH < 0) ? SCPRED : pred_inter;
	    else if ((abs (LSUM) < Pupdatethreshold[INDUPD]))
	      pred_taken = (ThirdH < 0) ? SCPRED : pred_inter;
          }
      }

    return pred_taken;
  }


  void HistoryUpdate (uint32_t PC, uint16_t brtype, bool taken,
		      uint32_t target, long long &X, int &Y,
		      folded_history * H, folded_history * G,
		      folded_history * J, long long &LH,
                      long long &SH, long long &TH, long long &PH,
		      long long &GBRHIST)
  {

    //special treatment for unconditional branchs;
    int maxt;

    //if (brtype == OPTYPE_BRANCH_COND)
    if(brtype == OPTYPE_JMP_INDIRECT_COND || brtype == OPTYPE_JMP_DIRECT_COND)
      maxt = 1;
    else
      maxt = 4;

    // the return stack associated history

    PH = (PH << 1) ^ (target ^(target >> 5) ^ taken);

    //if (brtype == OPTYPE_BRANCH_COND) {

    if (brtype == OPTYPE_JMP_INDIRECT_COND ||  brtype == OPTYPE_JMP_DIRECT_COND) {
            GBRHIST = (GBRHIST << 1) +   taken;
            LH = (LH << 1) + (taken);
            SH = (SH << 1) + (taken);
            SH ^= (PC & 15);
            TH=  (TH << 1) + (taken);
    }

    if (brtype ==  OPTYPE_RET_UNCOND || brtype == OPTYPE_RET_COND ) {
      pthstack= (pthstack-1) & 15;
    }

    //    if (brtype ==  OPTYPE_CALL_DIRECT){
    if (brtype ==  OPTYPE_CALL_DIRECT_UNCOND || brtype ==  OPTYPE_CALL_DIRECT_COND ){
      int index= (pthstack+1) & 15; HSTACK[index]= HSTACK[pthstack];
      pthstack= index;
    }

    int T = ((PC) << 1) + taken;
    int PATH = PC;

    for (int t = 0; t < maxt; t++)
      {
	bool DIR = (T & 1);
	T >>= 1;
	int PATHBIT = (PATH & 127);
	PATH >>= 1;

        //update  history

	Y--;
	ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
	X = (X << 1) ^ PATHBIT;
        for (int i = 1; i <= NHIST; i++)
	  {

	    H[i].update(ghist, Y);
	    G[i].update(ghist, Y);
	    J[i].update(ghist, Y);

	  }
      }


    //END UPDATE  HISTORIES
  }

// PREDICTOR UPDATE

  void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN)
  {

#ifdef LOOPPREDICTOR

    if (LVALID) {
      if (pred_taken != predloop)
        ctrupdate (WITHLOOP, (predloop == resolveDir), 7);
    }

    // optionally update or allocte entry in loop predictor
    loopupdate (PC, resolveDir, (pred_taken != resolveDir));

#endif

    bool  SCPRED = (LSUM >= 0);
    if (pred_inter != SCPRED)
	{
     	if ((abs (LSUM) < Pupdatethreshold[INDUPD]))
          if ((HighConf)) {

	      if ((abs (LSUM) < Pupdatethreshold[INDUPD] / 3))
		ctrupdate (FirstH, (pred_inter == resolveDir), CONFWIDTH);
	      else if ((abs (LSUM) < 2 * Pupdatethreshold[INDUPD] / 3))
		ctrupdate (SecondH, (pred_inter == resolveDir), CONFWIDTH);
	      else if ((abs (LSUM) < Pupdatethreshold[INDUPD]))
		ctrupdate (ThirdH, (pred_inter == resolveDir), CONFWIDTH);
	    }
	}

      if ((SCPRED != resolveDir)
	  || ((abs (LSUM) < Pupdatethreshold[INDUPD])))
	{
	  {
	    if (SCPRED != resolveDir)
	      Pupdatethreshold[INDUPD] += 1;
	    else
	      Pupdatethreshold[INDUPD] -= 1;

	    if (Pupdatethreshold[INDUPD] >= 256)
	      Pupdatethreshold[INDUPD] = 255;
	    if (Pupdatethreshold[INDUPD] < 0)
	      Pupdatethreshold[INDUPD] = 0;
	  }

	  ctrupdate (Bias[INDBIAS], resolveDir, PERCWIDTH);
          ctrupdate (BiasSK[INDBIASSK], resolveDir, PERCWIDTH);

	  Gupdate ((PC<<1)+pred_inter, resolveDir, GHIST, Gm, GGEHL, GNB, LOGGNB);
	  Gupdate (PC, resolveDir, L_shist[INDLOCAL], Lm, LGEHL,LNB, LOGLNB);

#ifndef REALISTIC
Gupdate (PC, resolveDir, S_slhist[INDSLOCAL], Sm, SGEHL, SNB, LOGSNB);
Gupdate (PC, resolveDir,T_slhist[INDTLOCAL], Tm, TGEHL, TNB, LOGTNB);
	  Gupdate (PC, resolveDir, HSTACK[pthstack], Pm, PGEHL, PNB, LOGPNB);
#endif

	}


      //TAGE UPDATE
      bool ALLOC = ((tage_pred != resolveDir) & (HitBank < NHIST));

      if (pred_taken == resolveDir)
	if ((next_pseudo_random () & 31) != 0)
	  ALLOC = false;
      // Do not allocate too often if the overall prediction is correct

	if (HitBank > 0)	  {

          // Manage the selection between longest matching and alternate matching
          // for "ps`eudo"-newly allocated longest matching entry

	    bool PseudoNewAlloc =
	      (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) <= 1);

            // an entry is considered as newly allocated if its prediction counter is weak

	    if (PseudoNewAlloc) {

		if (LongestMatchPred == resolveDir)
		  ALLOC = false;

                // if it was delivering the correct prediction, no need to allocate a new entry
                // even if the overall prediction was false

		if (LongestMatchPred != alttaken) {
		    int index = (INDUSEALT) ^ LongestMatchPred;
		    ctrupdate (use_alt_on_na[index]
			       [HitBank > (NHIST / 3)],
			       (alttaken == resolveDir), 4);

		  }

	      }
	  }


     if (ALLOC) {

	  int T = NNN;
	  int A = 1;

	  if ((next_pseudo_random () & 127) < 32)
	    A = 2;

	  int Penalty = 0;
	  int NA = 0;

          for (int i = HitBank + A; i <= NHIST; i += 1) {
	      if (gtable[i][GI[i]].u == 0)
		{
		  gtable[i][GI[i]].tag = GTAG[i];
		  gtable[i][GI[i]].ctr = (resolveDir) ? 0 : -1;
		  NA++;
		  if (T <= 0) {
		      break;
                  }
                  i +=  1;
		  T -= 1;
		}
	      else {
		  Penalty++;
              }
          }

	  TICK += (Penalty -NA );

          //just the best formula for the Championship
	  if (TICK < 0)
	    TICK = 0;
 	  if (TICK > 1023) {

	      for (int i = 1; i <= NHIST; i++)
		for (int j = 0; j <= (1 << log_nentry_tables[i]) - 1; j++)

// substracting 1 to a whole array is not that realistic
#ifdef REALISTIC
                     gtable[i][j].u >>= 1;
#else
                     if (gtable[i][j].u)  gtable[i][j].u--;
#endif
 	      TICK = 0;

              }


	}
//update predictions
      if (HitBank > 0)
	{
	  if (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)
	    if (LongestMatchPred != resolveDir)

	      {			// acts as a protection
		if (AltBank > 0)
		  {
		    ctrupdate (gtable[AltBank][GI[AltBank]].ctr,
			       resolveDir, CWIDTH);

		  }
		if (AltBank == 0)
		  baseupdate (resolveDir);
	      }
	  ctrupdate (gtable[HitBank][GI[HitBank]].ctr, resolveDir, CWIDTH);
          
          //sign changes: no way it can have been useful          
	  if (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)
	    gtable[HitBank][GI[HitBank]].u = 0;

	}
      else
	baseupdate (resolveDir);

      if (LongestMatchPred != alttaken)
	if (LongestMatchPred == resolveDir)
        {if (gtable[HitBank][GI[HitBank]].u < (1 << UWIDTH) - 1)
                  gtable[HitBank][GI[HitBank]].u++;
        }
//END TAGE UPDATE

      HistoryUpdate (PC, opType /* OPTYPE_BRANCH_COND */, resolveDir, branchTarget,  phist,
		       ptghist, ch_i, ch_t[0],
		       ch_t[1], L_shist[INDLOCAL],
		       S_slhist[INDSLOCAL], T_slhist[INDTLOCAL], HSTACK[pthstack], GHIST);

//END PREDICTOR UPDATE


  }


  /**
   *
   */
  int Gpredict (UINT32 PC, long long BHIST, int *length, int8_t ** tab, int NBR, int logs)
  {

    //calcul de la somme, commence par  le biais du branchement
    int PERCSUM = 0;
    for (int i = 0; i < NBR; i++)
      {
	long long bhist = BHIST & ((long long) ((1 << length[i]) - 1));

	long long index = (((long long) PC) ^ bhist ^ (bhist >> (8 - i)) ^
                           (bhist >> (16 - 2 * i)) ^ (bhist >> (24 - 3 * i)) ^ (bhist >>
                                                                                (32 -
                                                                                 3 *
								 i)) ^ (bhist
									>> (40
									    -
									    4
									    *
									    i)))
             & ((1 << (logs-(i >=(NBR-2)))) - 1);

	int16_t ctr = tab[i][index];
	PERCSUM += (2 * ctr + 1);
      }
    return ((PERCSUM));
  }



  void Gupdate (UINT32 PC, bool taken, long long BHIST, int *length,
		int8_t ** tab, int NBR, int logs)
  {


    for (int i = 0; i < NBR; i++)
      {
	long long bhist = BHIST & ((long long) ((1 << length[i]) - 1));
	long long index =
	  (((long long) PC) ^ bhist ^ (bhist >> (8 - i)) ^
	   (bhist >> (16 - 2 * i)) ^ (bhist >> (24 - 3 * i)) ^ (bhist >>
								(32 -
								 3 *
								 i)) ^ (bhist
									>> (40
									    -
									    4
									    *
									    i)))
  & ((1 << (logs-(i >=(NBR-2)))) - 1);

        ctrupdate (tab[i][index], taken, PERCWIDTH -(i<(NBR-1)));


      }
  }

  void TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget) {

    bool taken = true;

    switch (opType) {

    case OPTYPE_RET_UNCOND:
    case OPTYPE_JMP_DIRECT_UNCOND:
    case OPTYPE_JMP_INDIRECT_UNCOND:
    case OPTYPE_CALL_DIRECT_UNCOND:
    case OPTYPE_CALL_INDIRECT_UNCOND:
    case OPTYPE_RET_COND:
    case OPTYPE_JMP_DIRECT_COND:
    case OPTYPE_JMP_INDIRECT_COND:
    case OPTYPE_CALL_DIRECT_COND:
    case OPTYPE_CALL_INDIRECT_COND:

      HistoryUpdate(PC, opType, taken, branchTarget, phist,
                    ptghist, ch_i,
                    ch_t[0], ch_t[1],
                    L_shist[INDLOCAL],
                    S_slhist[INDSLOCAL],
                    T_slhist[INDTLOCAL],
                    HSTACK[pthstack],
                    GHIST);
      break;


    default:;
    }

  }

};



#endif
