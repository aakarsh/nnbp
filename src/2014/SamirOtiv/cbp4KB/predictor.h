#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
#include "tracer.h"



//Unlimited
//#define PATTROWS 8192
#define PATTROWS 32
#define LOCALHISTORYLENGTH 8
#define MAXPLEN 4
#define LOCAL_SEL_RANGE 15

#define TAGWIDTH 7

#define USEFUL_RANGE LOCALHISTORYLENGTH
#define RESET_RANGE 16

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

class PREDICTOR{
private:
    char select[PATTROWS][2];
    
    bool  TagePredict(UINT32 PC);
    void  TageUpdate(UINT32 PC, bool taken, bool predDir, UINT32 branchTarget);
    void TageInit();
    
    
    
    
    class patternpredictor {
        UINT32 pattern_sample[PATTROWS][2];
        UINT32 tag[PATTROWS][2];
        UINT32 useful[PATTROWS][2];        
        
        UINT32 resetcounter;
        
        UINT32 tagfn (UINT32 PC);
        
        void ResetUseful();
        
        
    public:
        int GetTableID (UINT32 PC);
        void InitPatternPredictor ();
        void UpdatePatternPredictor (UINT32 PC, bool taken);
        bool GetPatternPrediction (UINT32 PC);
        bool IsPatternPredReady (UINT32 PC);
        void PrintAll ();
        
        bool TagMatches (UINT32 PC);
    };
    
    public:
    patternpredictor local;
    
    
    
    //STATS
    long long unsigned int s_asked, s_correct, l_asked, l_correct;
    int last;
    
    
public:
    
    // The interface to the four functions below CAN NOT be changed
    
    PREDICTOR(void);
    bool    GetPrediction(UINT32 PC);
    void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
    void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget);
    
    // Contestants can define their own functions below
    
    void PrintStats();
    
};



/***********************************************************/
#endif








#ifdef _INCLUDETAGE_
/*
 CODE DERIVED & PORTED FROM THE 64K ISL-TAGE PREDICTOR BY ANDREW SEZNEC, FROM CBP 3.
 
 */
// uop type encoding
#define IS_BR_CONDITIONAL (1 << 0)   // conditional branch
#define IS_BR_INDIRECT    (1 << 1)   // indirect branch
#define IS_BR_CALL        (1 << 2)   // call
#define IS_BR_RETURN      (1 << 3)   // return
#define IS_BR_OTHER       (1 << 4)   // other branches
#define IS_LOAD           (1 << 5)   // load uop
#define IS_STORE          (1 << 6)   // store uop
#define IS_FP             (1 << 7)   // floating point uop
#define IS_WFLAGS         (1 << 8)   // uop writes flags
#define IS_RFLAGS         (1 << 9)   // uop reads flags
#define IS_BOM            (1 << 14)  // uop is the beginning uop of an instruction
#define IS_EOM            (1 << 15)  // uop is the end uop of an instruction

/*
 Code has been succesively derived from the tagged PPM predictor simulator from Pierre Michaud, the OGEHL predictor simulator from by AndrÈ Seznec, the TAGE predictor simulator from  AndrÈ Seznec and Pierre Michaud
 
 */

#include <inttypes.h>
#include <math.h>
//#define SHARINGTABLES		// let us share the physical among several logic predictor tables
//#define INITHISTLENGTH		// uses the "best" history length we found
#define STATCOR				// Use the Statistical Corrector predictor
//#define IUM					//Use the Immediate Update Mimicker
#define LOOPPREDICTOR		// Use the loop predictor

#define NN 15

#define NHIST 7			// 7 TAGGED COMPONENTS
#define LOGB (NN - 4)				// log of number of entries in bimodal predictor
//#define HYSTSHIFT 2			// sharing an hysteris bit between 4 bimodal predictor entries
#define HYSTSHIFT 1			// sharing an hysteris bit between 4 bimodal predictor entries

#define LOGG (NN - 7)		// initial definition was with 2K entries per tagged tables

//The Best  set of history lengths
//int m[NHIST + 1] =		{0, 3, 3, 8, 8, 12,12, 17,17, 33,33, 35,35, 67,67, 97,97, 138,138, 195,195, 330,330, 517,517, 1193,1193, 1741,1741, 1930,1930};
//int m[16] =		{0, 3, 8, 12, 17, 35, 67, 97, 195, 517, 1193, 1741};

int m[] =		{0, 3, 8, 12, 17, 33, 35, 67, 97, 138, 195, 330, 517, 1193, 1741, 1930};
//int taggedtableoffset[] =     {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1};
//int taggedtableoffset[] =     {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2};
int taggedtableoffset[] =     {0, 0, 0, 0, 0, 0, 0, 0};

int tabletagwidths[] = {11, 11, 10, 11, 10, 11, 10 ,11};

//int sharedtableoffset[] = {1, 3, 2, -1};
int sharedtableoffset[] = {2, 3, 1, -2};

//int sharedtabletags[] = {8, 11, 13, 14};
int sharedtabletags[] = {8, 11, 13, 14};

#ifndef INITHISLENGTH
#define MINHIST 5			// shortest history length
#define MAXHIST 126		// longest history length
#endif

#define PHISTWIDTH 16		// width of the path history


#define UWIDTH 2


#ifndef SHARINGTABLES
#define TBITS 6
#define MAXTBITS 15
#endif



#define CWIDTH 3		// predictor counter width on the tagged tables
#define HISTBUFFERLENGTH 2048	// we use a 4K entries history buffer to store the branch history


//Statistical corrector parameters
#define NSTAT 5
#define CSTAT 5
#define LOGStatCor 7	// the Statistical Corrector predictor features 5 * 1K entries
int8_t StatCor[1 << (LOGStatCor)];
int8_t USESTATCORTHRESHOLD;
int8_t CountStatCorThreshold;
#define MINUSESTATCORTHRESHOLD 5
#define MAXUSESTATCORTHRESHOLD 31	//just as a security, never reach this limit
#define UPDATESTATCORTHRESHOLD (21+  8*USESTATCORTHRESHOLD)




#define LOGL 3			//64 entries loop predictor 4-way skewed associative
#define WIDTHNBITERLOOP 10	// we predict only loops with less than 1K iterations
//#define LOOPTAG 10		//tag width in the loop predictor
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
int8_t USE_ALT_ON_NA;		// "Use alternate prediction on newly allocated":  a 4-bit counter  to determine whether the newly allocated entries should be considered as  valid or not for delivering  the prediction
int TICK, LOGTICK;		//control counter for the smooth resetting of useful counters
int phist;			// use a path history as on  the OGEHL predictor
uint8_t ghist[HISTBUFFERLENGTH];
int Fetch_ptghist;
int Fetch_ptghistos;
int Fetch_phist;		//path history
int Fetch_phistos;		//path os history
folded_history Fetch_ch_i[NHIST + 1];	//utility for computing TAGE indices
folded_history Fetch_ch_t[2][NHIST + 1];	//utility for computing TAGE tags
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
int TB[NHIST + 1];  		// tag width for the different tagged tables
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


//For the IUM
#define LOGSPEC 6
int PtIumRetire;
int PtIumFetch;
specentry *IumPred;



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
        Fetch_ptghistos = 0;
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
            TB[i] = tabletagwidths[i];
        
        // log2 of number entries in the tagged components
        
        
        for (int i = 1; i <= 3; i++)
            logg[i] = LOGG + taggedtableoffset[i];
        for (int i = 4; i <= 6; i++)
            logg[i] = LOGG + taggedtableoffset[i];
        for (int i = 7; i <= 10; i++)
            logg[i] = LOGG + taggedtableoffset[i];
        for (int i = 10; i <= NHIST; i++)
            logg[i] = LOGG + taggedtableoffset[i];
        for (int i = 1; i <= NHIST; i++)
        {
            gtable[i] = new gentry[1 << (logg[i])];
        }
#endif
        
        
#ifdef SHARINGTABLES
#define STEP1 3
#define STEP2 8
#define STEP3 14
        // log2 of number entries in the tagged components
        logg[1] = LOGG + sharedtableoffset[0];
        logg[STEP1] = LOGG + sharedtableoffset[1];
        logg[STEP2] = LOGG + sharedtableoffset[2];
        logg[STEP3] = LOGG + sharedtableoffset[3];
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
            TB[i] = sharedtabletags[0];
        }
        for (int i = STEP1; i <= STEP2 - 1; i++)
        {
            gtable[i] = gtable[STEP1];
            TB[i] = sharedtabletags[1];
        }
        for (int i = STEP2; i <= STEP3 - 1; i++)
        {
            gtable[i] = gtable[STEP2];
            TB[i] = sharedtabletags[2];
        }
        for (int i = STEP3; i <= NHIST; i++)
        {
            gtable[i] = gtable[STEP3];
            TB[i] = sharedtabletags[3];
        }
        
        
#endif
        //initialisation of the functions for index and tag computations
        
        for (int i = 1; i <= NHIST; i++)
        {
            Fetch_ch_i[i].init (m[i], (logg[i]));
            Fetch_ch_t[0][i].init (Fetch_ch_i[i].OLENGTH, TB[i]);
            Fetch_ch_t[1][i].init (Fetch_ch_i[i].OLENGTH, TB[i] - 1);
        }
        
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
        IumPred = new specentry[1 << LOGSPEC];
        
        
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
            STORAGESIZE += (1 << logg[i]) * (CWIDTH + UWIDTH + TB[i]);
            NBENTRY += (1 << logg[i]);
        }
#endif
#ifdef SHARINGTABLES
        STORAGESIZE += (1 << (logg[1])) * (CWIDTH + 1 + TB[1]);
        STORAGESIZE += (1 << (logg[STEP1])) * (CWIDTH + 1 + TB[STEP1]);
        STORAGESIZE += (1 << (logg[STEP2])) * (CWIDTH + 1 + TB[STEP2]);
        STORAGESIZE += (1 << (logg[STEP3])) * (CWIDTH + 1 + TB[STEP3]);
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
        STORAGESIZE += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));
        fprintf (stdout, "TAGE %d bytes, ", STORAGESIZE / 8);
        
        
#ifdef LOOPPREDICTOR
        STORAGESIZE += (1 << LOGL) * (3 * WIDTHNBITERLOOP + LOOPTAG + 3 + 3 + 1);
        fprintf (stdout, "LOOPPRED %d bytes, ",
                 ((1 << LOGL) * (3 * WIDTHNBITERLOOP + LOOPTAG + 3 + 3 + 1)) / 8);
#endif
        
#ifdef STATCOR
        STORAGESIZE += CSTAT * (1 << (LOGStatCor));
        fprintf (stdout, "Stat Cor %d bytes, ", CSTAT * (1 << (LOGStatCor)) / 8);
#endif
#ifdef IUM
        STORAGESIZE += (1 << LOGSPEC) * 20;
        fprintf (stdout, "IUM  %d bytes, ", ((1 << LOGSPEC) * 20) / 8);
        // Entry width on the speculative table is 20 bits (19 bits to identify the table entry that gives the prediction and 1 prediction bit
#endif
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
    //At fetch time: speculative
    bool getloopspec (uint32_t pc)
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
                if (ltable[index].CurrentIterSpec + 1 == ltable[index].NbIter)
                    return (!(ltable[index].dir));
                else
                    return ((ltable[index].dir));
                
            }
        }
        
        LVALID = false;
        return (false);
        
    }
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
                if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
                    return (!(ltable[index].dir));
                else
                    return ((ltable[index].dir));
            }
        }
        
        LVALID = false;
        return (false);
        
    }
    void speculativeloopupdate (uint32_t pc, bool Taken)
    {
        //all indexes have been computed during prediction computations
        if (LHIT != -1)
        {
            int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
            if (Taken != ltable[index].dir)
                ltable[index].CurrentIterSpec = 0;
            else
            {
                ltable[index].CurrentIterSpec++;
                ltable[index].CurrentIterSpec &= ((1 << WIDTHNBITERLOOP) - 1);
                //loop with more than 2** WIDTHNBITERLOOP are not correctly predicted
            }
        }
        
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
    //the IUM predictor
    bool PredIum (bool pred)
    {
        int IumTag = (HitBank) + (GI[HitBank] << 4);
        if (HitBank == 0)
            IumTag = BI << 4;
        int Min = (PtIumRetire > PtIumFetch + 8) ? PtIumFetch + 8 : PtIumRetire;
        
        for (int i = PtIumFetch; i < Min; i++)
        {
            if (IumPred[i & ((1 << LOGSPEC) - 1)].tag == IumTag)
                return IumPred[i & ((1 << LOGSPEC) - 1)].pred;
            
            
        }
        return pred;
        
    }
    
    void UpdateIum (bool Taken)
    {
        int IumTag = (HitBank) + (GI[HitBank] << 4);
        if (HitBank == 0)
            IumTag = BI << 4;
        PtIumFetch--;
        IumPred[PtIumFetch & ((1 << LOGSPEC) - 1)].tag = IumTag;
        IumPred[PtIumFetch & ((1 << LOGSPEC) - 1)].pred = Taken;
        
    }
    //compute the prediction
    
    bool predict_brcond (unsigned int pc, uint16_t brtype)
    {
        if (brtype & IS_BR_CONDITIONAL)
        {
            // computes the TAGE table addresses and the partial tags
            for (int i = 1; i <= NHIST; i++)
            {
                GI[i] = gindex (pc, i, Fetch_phist, Fetch_ch_i);
                GTAG[i] = gtag (pc, i, Fetch_ch_t[0], Fetch_ch_t[1]);
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
            predloop = getloopspec (pc);	// loop prediction
            pred_taken = ((WITHLOOP >= 0) && (LVALID)) ? predloop : pred_taken;
#endif
#ifdef IUM
            bool Ium = PredIum (pred_taken);
            pred_taken = Ium;	// correct with inflight information if needed
            
#endif
        }
        return pred_taken;
    }
    
    
    //  UPDATE  FETCH HISTORIES   + spec update of the loop predictor + Update of the IUM
    void FetchHistoryUpdate (uint32_t pc, uint16_t brtype, bool taken,
                             uint32_t target)
    {
        
        if (brtype & IS_BR_CONDITIONAL)
        {
#ifdef LOOPPREDICTOR
            speculativeloopupdate (pc, taken);
#endif
#ifdef IUM
            UpdateIum (taken);
#endif
        }
        
        HistoryUpdate (pc, brtype, taken, target, Fetch_phist, Fetch_ptghist,
                       Fetch_ch_i, Fetch_ch_t[0], Fetch_ch_t[1]);
        
    }
    
    void HistoryUpdate (uint32_t pc, uint16_t brtype, bool taken,
                        uint32_t target, int &X, int &Y, folded_history * H,
                        folded_history * G, folded_history * J)
    {
        //special treatment for indirects and returnd: inhereited from the indirect branch predictor submission
        int maxt = (brtype & IS_BR_INDIRECT) ? 4 : 1;
        if (brtype & IS_BR_CALL)
            maxt = 5;
        
        
        
        int T = ((target ^ (target >> 3) ^ pc) << 1) + taken;
        int PATH = pc;
        for (int t = 0; t < maxt; t++)
        {
            bool TAKEN_VAR = (T & 1);
            T >>= 1;
            bool PATHBIT = (PATH & 1);
            PATH >>= 1;
            //update  history
            Y--;
            ghist[Y & (HISTBUFFERLENGTH - 1)] = TAKEN_VAR;
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
    
    // PREDICTOR UPDATE
    
    void update_brcond (uint32_t pc, uint16_t brtype, bool taken,
                        uint32_t target)
    {
        
        if (brtype & IS_BR_CONDITIONAL)
        {
            PtIumRetire--;
            // Recompute the prediction
            
            //Note that on a real hardware processor, one would avoid this recomputation to save an extra read port on the branch predictor. One  would  have to propagate informations needed for update with the instruction: HitBank, updated through the STATCOR or not, etc.
            
            //Recompute the prediction (the indices and tags) with the Retire history.
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
                            gtable[i][j].u >>= 1;
                    
                }
                
                //update the prediction
                
                if (HitBank > 0)
                {
                    ctrupdate (gtable[HitBank][GI[HitBank]].ctr, taken, CWIDTH);
                    // acts as a protection
                    if (gtable[HitBank][GI[HitBank]].u == 0)
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
                    if (LongestMatchPred != alttaken)
                    {
                        if (LongestMatchPred == taken)
                        {
                            if (gtable[HitBank][GI[HitBank]].u < ((1 << UWIDTH) -1))
                                gtable[HitBank][GI[HitBank]].u++;
                            
                        }
                    }
                //END PREDICTOR UPDATE
                
                
            }
            
        }
        
        //  UPDATE RETIRE HISTORY
        HistoryUpdate (pc, brtype, taken, target, Retire_phist, Retire_ptghist,
                       Retire_ch_i, Retire_ch_t[0], Retire_ch_t[1]);
    }
    
};








#endif