#include "predictor.h"
#include "stdint.h"

#define REMOVE_DUP_METHOD_1
#define FOLDED_GHR_HASH 
#define RECENCY_STACK
#define LOOPPREDICTOR

#define GHL 4096     // maximum history length our predictor attempts tp correalte with
#define LOG_BST 24
#define  WT_REG 22
#define  WL_1 16
#define  WT_RS 26
#define  WL_2 72
#define  REMOVE_DUP_DIST 32
#define  NFOLDED_HIST 512
#define  TRAIN_TH 107

#define PHISTWIDTH 16   

#define LOGL 6			//64 entries loop predictor 4-way skewed associative
#define WIDTHNBITERLOOP 10	// we predict only loops with less than 1K iterations
#define LOOPTAG 10		//tag width in the loop predictor

/*----- histLen array is used in conjuntion with filtered history as defined below in the code and is used to boost accuracy and provide multiple instances of a branch present in the past global history to the prediction function if required ------*/
#define NHIST 15
int histLen[NHIST] = {64, 80, 96, 112, 128, 160, 192, 256, 320, 416, 512, 768, 1024, 1536, 2048};

#define PA_SHIFT_HASH 2 

/* --- weight tables --- */
int8_t     **weight_m;          //1-dimensional bias weight table
INT32     *weight_b;         //2-dimensional conventional perceptron predictor weight table
int8_t     *weight_rs;         //1-dimensional weight table as proposed in the writeup


/* --- Shift registers --- */
bool HTrain; // A register indicate if training is needed 
bool *GHR;  // Unfiltered history shift register used for prediction and updates

bool *isBrStable;// A shift register/circular buffer indicating if a branch in PA was stable the time it was inserted into that

UINT32 *HA;	// Path address register containing the address of past branches without any filtering

UINT32 *HAConsd;	// Set containing some path addresses used later in the code to remove duplicate instances of a branch in the prediction computation

UINT32 *HistConsd;	// Set containing the absolute distances of branches, used later in the code to remove duplicate instances of a branch in the prediction computation

/*-----Folloing three variables are used to capture filtered histories that appeared at least at depth of EXTRA_PHIST_START_DIST in the global history;used for boosting the accuracy -----*/
bool *filteredGHR;// outcome of the branch in filtered history
UINT32 *filteredHA;	// address of the brancg in filtered history 
UINT32 *filteredHADist;	// absolute distance/depth of the branch in the global history


UINT32 *folded_hist;// folded history array; n-th bit in this array computes folded history for the history bits from the n-th positon in the global unfiltered history to the current branch; 
/*-----Folloing four variables are used to avoid the recomputation of perceptron table index during predictor retire/update, so they are merely used for speeding up the simulation, hence some space allocated for them are not included in the storage budget shown in the writeup -----*/
UINT32 *OUTPOINT;
UINT32 *idxConsd;
bool *dirConsd;
UINT32 corrFoundCurCycle = 0;

/*-----Folloing three variables are the three fields in Recency Stack (RS) defined in the writeup -----*/
bool *nonDupFilteredGHR1;           //used for RS[].H to contain the latest outcome of a branch

UINT32 *nonDupFilteredHA1;          //used for RS[].A to contain branch address/tag (hashed down to required number of bits)

UINT32 *nonDupFilteredHADist1;      //used for RS[].P to contain absolute distance of the latest occurrence of a branch in global history


INT32 GHRHeadPtr = 0;//pointer to the start of the circular buffer GHR
INT32 filteredGHRHeadPtr = 0;//pointer to the start of the circular buffer filteredGHR, filteredGHRHA and filteredGHRDist

INT32 threshold; // dynamic threshold value as in O-GEHL
INT32 TC; //threshold counter as in O-GEHL

UINT32 numFetchedCondBr = 0;

int Fetch_phist;		//path history
int Retire_phist;		//path history
int Seed;			// for the pseudo-random number generator

class bst_entry      //Branch Status table entry
{
   /* state 00 means -> Not Found, 
    * 01 means -> Taken, 
    * 02 means -> Not Taken, 
    * 03 means -> Non-biased */
   public:
      int8_t state;
      bst_entry ()
      {
         state = 0;
      }
};

#ifdef LOOPPREDICTOR
class lentry			//loop predictor entry
{
public:
  UINT32 NbIter;		//10 bits
  UINT32 confid;		// 3 bits
  UINT32 CurrentIter;		// 10 bits
  UINT32 TAG;			// 10 bits
  UINT32 age;			//3 bits
  bool dir;			// 1 bit


  //37 bits per entry    
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

bst_entry *bst_table;			//branch status table
bool prcpt_pred;			// prcpt prediction

#ifdef LOOPPREDICTOR
lentry *ltable;			//loop predictor table
//variables for the loop predictor
bool predloop;			// loop predictor prediction
int LIB;
int LI;
int LHIT;			//hitting way in the loop predictor
int LTAG;			//tag on the loop predictor
bool LVALID;			// validity of the loop predictor prediction
INT32 WITHLOOP;		// counter to monitor whether or not loop prediction is beneficial

#endif

PREDICTOR::PREDICTOR(void) {
  
   weight_m = new int8_t*[1<<(WT_REG)];
   for(int i = 0; i < (1<<(WT_REG)); ++i) {
      weight_m[i] = new int8_t[WL_1];
   }
   weight_b = new INT32[1<<(WT_REG)];
   weight_rs = new int8_t[(1<<WT_RS)];

   GHR = new bool[GHL];
   isBrStable = new bool[GHL];
   filteredGHR = new bool[GHL];
   HA = new UINT32[GHL];	
   HAConsd = new UINT32[GHL];	
   HistConsd = new UINT32[GHL];	
   filteredHA = new UINT32[GHL];	
   filteredHADist = new UINT32[GHL];	

   folded_hist = new UINT32[NFOLDED_HIST];	
   OUTPOINT = new UINT32[GHL];
   idxConsd = new UINT32[WL_1+WL_2];
   dirConsd = new bool[WL_1+WL_2];

   nonDupFilteredGHR1 = new bool[WL_2];
   nonDupFilteredHA1 = new UINT32[WL_2];
   nonDupFilteredHADist1 = new UINT32[WL_2];

   bst_table = new bst_entry[1 << LOG_BST];
   for (int idx = 0; idx < (1 << LOG_BST); idx++){
      bst_table[idx].state = 0;
   }

   numFetchedCondBr = 0;

   for (int i = 0; i < (1<<WT_REG); i++) {
      for(int j = 0; j < WL_1; j++)
      {
         weight_m[i][j] = 0;
      }
      weight_b[i] = 0;
   }

   threshold = TRAIN_TH;
   TC = 0;

   for (int k = 0; k < GHL; k++) {
      GHR[k] = false;
      isBrStable[k] = true;
      filteredGHR[k] = false;
      HA[k] = 0;
      HAConsd[k] = 0;
      HistConsd[k] = 0;
      filteredHA[k] = 0;
      filteredHADist[k] = 0;
   }

   for (int k = 0; k < WL_2; k++) {
      nonDupFilteredGHR1[k] = false;
      nonDupFilteredHA1[k] = 0;
      nonDupFilteredHADist1[k] = 0;
   }

   for (int i = 0; i < (1<<WT_RS); i++) {
      weight_rs[i] = 0;
   }

   for (int i = 0; i < NFOLDED_HIST; i++) {
      folded_hist[i] = 0;
   }
   for (int i = 0; i < WL_1; i++) {
      OUTPOINT[i] = (i+1) % WT_REG;
   }
   for (int i = WL_1; i < GHL; i++) {
      OUTPOINT[i] = (i+1) % WT_RS;
   }

#ifdef LOOPPREDICTOR
   LVALID = false;
   WITHLOOP = -1;
   ltable = new lentry[1 << (LOGL)];
#endif
   Fetch_phist = 0;
   Retire_phist = 0;
   Seed = 0;
}

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

  // up-down saturating counter
  void ctrupdate (INT32 & ctr, bool taken, int nbits)
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
  int lindex (UINT32 pc)
  {
    return ((pc & ((1 << (LOGL - 2)) - 1)) << 2);
  }


//loop prediction: only used if high confidence
//skewed associative 4-way
  bool getloop (UINT32 pc)
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

  void loopupdate (UINT32 pc, bool origDir, bool ALLOC)
  {
     if (LHIT >= 0)
     {
        int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
        //already a hit 
        if (LVALID)
        {
           if (origDir != predloop)
           {
              // free the entry
              ltable[index].NbIter = 0;
              ltable[index].age = 0;
              ltable[index].confid = 0;
              ltable[index].CurrentIter = 0;
              return;
           }
           else if ((predloop != prcpt_pred) || ((MYRANDOM () & 7) == 0))
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
        if (origDir != ltable[index].dir)
        {
           if (ltable[index].CurrentIter == ltable[index].NbIter)
           {
              if (ltable[index].confid < 7)
                 ltable[index].confid++;
              if (ltable[index].NbIter < 3)
                 //just do not predict when the loop count is 1 or 2     
              {
                 // free the entry
                 ltable[index].dir = origDir;
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
        UINT32 X = MYRANDOM () & 3;
        if ((MYRANDOM () & 3) == 0)
           for (int i = 0; i < 4; i++)
           {
              int LHIT = (X + i) & 3;
              int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
              if (ltable[index].age == 0)
              {
                 ltable[index].dir = !origDir;
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

/*----- Hash function that calculating index of weight tables -----*/
UINT32 gen_widx(UINT32 cur_pc, UINT32 path_pc, UINT32 wt_size) 
{
    cur_pc = (cur_pc ) ^ (cur_pc / (1<<wt_size));
    path_pc = path_pc >> PA_SHIFT_HASH;
    path_pc = (path_pc) ^ (path_pc / (1<<wt_size));
    UINT32 widx = cur_pc ^ (path_pc);
    widx = widx % (1<<wt_size);
    return widx;
}
UINT32 gen_bias_widx(UINT32 cur_pc, UINT32 wt_size) 
{
   cur_pc = (cur_pc ) ^ (cur_pc / (1<<wt_size));
   UINT32 widx = cur_pc;
   widx = widx % (1<<wt_size);
   return widx;
}
bool PREDICTOR::GetPrediction(UINT32 PC){

   INT32 accum = 0;

   accum = 0;
   corrFoundCurCycle = 0;
   bool pred;
   int corrConsd = 0;
   //int histPtr = 0;
   //int filteredHistPtr = 0;

   if (bst_table[PC & ((1 << LOG_BST) - 1)].state != 3) {
      pred = (bst_table[PC & ((1 << LOG_BST) - 1)].state == 1)?true:false;
   }
   else {
      int non_dup_consd = 0;
      int distantHist = 0;

      /*----- Accumulate bias weight -----*/
      UINT32 bias_widx = gen_bias_widx(PC, WT_REG);
      accum = accum + weight_b[bias_widx];

      /*----- Accumulate weights from the conventional perceptron predictor weight table for 11 recent unfiltered branches -----*/
      for(int j=0, histPtr=GHRHeadPtr; j <WL_1; j++) {
         UINT32 widx = gen_widx(PC, HA[histPtr], WT_REG);
#ifdef FOLDED_GHR_HASH
         widx = widx ^ folded_hist[j];
         widx = widx % (1<<WT_REG);
#endif
         if( GHR[histPtr] == 1)		// If history is Taken
            accum += weight_m[widx][j];  // Then add Taken weight
         else 				// If history is Not-Taken
            accum -= weight_m[widx][j]; // Then add Not-Taken weight

         idxConsd[corrFoundCurCycle] = widx;
         dirConsd[corrFoundCurCycle] = GHR[histPtr];
         corrFoundCurCycle++;

         HistConsd[j] = j+1;

         histPtr++;
         if (histPtr == GHL) histPtr = 0;
      }

      /*----- Accumulate weights from weight_rs table for Non_biased branches that are at absolute distance 12 to distance 32 in the past global history-----*/
      /*----- This is done to boost accuracy ------*/
      for(int j=WL_1, histPtr=(GHRHeadPtr+WL_1)%GHL;j<REMOVE_DUP_DIST; j++) {
         if ((HA[histPtr] != 0) && (isBrStable[histPtr] == false)){
            UINT32 widx = 0;
            UINT32 distance;
#ifdef RECENCY_STACK
            widx = gen_widx(PC, HA[histPtr], WT_RS); // compute index to access either weight table 
            distance = (j+1) ^ ((j+1) / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            widx = widx ^ folded_hist[j];
            widx = widx % (1<<WT_RS);
#endif
            if(GHR[histPtr] == 1)		// If history is Taken
               accum += weight_rs[widx];
            else 				// If history is Not-Taken
               accum -= weight_rs[widx];
#endif
            HistConsd[WL_1+distantHist] = j+1;

            idxConsd[corrFoundCurCycle] = widx;
            dirConsd[corrFoundCurCycle] = GHR[histPtr];
            corrFoundCurCycle++;

            distantHist++;
         }
         histPtr++;
         if (histPtr == GHL) histPtr = 0;

      }

      /*----- Accumulate weights from weight_rs table for Non_biased branches (present in the Recency Stack) that are at absolute distance above 32 and below 1024 in the past global history-----*/
      for(int j=0;((distantHist < WL_2) && (j<WL_2)); j++, corrConsd++) {
         int pathPCDist = numFetchedCondBr - nonDupFilteredHADist1[j];
         if ((nonDupFilteredHA1[j] != 0) && /*(pathPCDist <= GHL)*/(pathPCDist < 1024)){
            UINT32 widx = 0;
            UINT32 distance;

#ifdef RECENCY_STACK
            widx = gen_widx(PC, nonDupFilteredHA1[j], WT_RS); // compute index to access either weight table 
            distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            if ((pathPCDist-1) < NFOLDED_HIST) {
               widx = widx ^ folded_hist[pathPCDist-1];
               widx = widx % (1<<WT_RS);
            }
            else {
               widx = widx ^ folded_hist[NFOLDED_HIST-1];
               widx = widx % (1<<WT_RS);
            }

#endif
            if(nonDupFilteredGHR1[j] == 1)		// If history is Taken
               accum += weight_rs[widx];
            else 				// If history is Not-Taken
               accum -= weight_rs[widx];
#endif
            HistConsd[WL_1+distantHist] = pathPCDist;

            idxConsd[corrFoundCurCycle] = widx;
            dirConsd[corrFoundCurCycle] = nonDupFilteredGHR1[j];
            corrFoundCurCycle++;

            distantHist++;
         }
         else {
            break;
         }
      }

      non_dup_consd = 0;
      int prevPathPCDist = 0;

      /*----- On few traces, a small set of branches executes repeatedly in the past history, as a result Recency Stack can not provide that many branches to the prediction function since it contains only the latest occurrence of a branch. The following For loop with the help of {filteredHA, filteredHADist, filteredGHR} and histLen array attempts to capture muliple instances of a branch in different distances in the past history to fill in that room and assists the prediction function-----*/
      /*----- Accumulate weights from weight_rs table for multple instances of Non_biased branches present in different distances (at absolute distance of at least 64) present in the filtered history-----*/
      for(int j=0, filteredHistPtr = filteredGHRHeadPtr; ((distantHist < WL_2) && (j < GHL)); j++) {
         int pathPCDist = numFetchedCondBr - filteredHADist[filteredHistPtr];
         if ((pathPCDist > 64) && (filteredHA[filteredHistPtr] != 0) && /*(pathPCDist <= GHL)*/(pathPCDist < 1024)){

            int prevPathPCBin = 0;
            int curPathPCBin = 0;
            for (int it1 = 0; it1 < (NHIST-1); it1++) {
               if ((pathPCDist >= histLen[it1]) && (pathPCDist < histLen[it1+1])) {
                  curPathPCBin = it1+1;
               }
               if ((prevPathPCDist >= histLen[it1]) && (prevPathPCDist < histLen[it1+1])) {
                  prevPathPCBin = it1+1;
               }
            }
            if (prevPathPCBin != curPathPCBin) {
               non_dup_consd = 0;
            }
            prevPathPCDist = pathPCDist;

            UINT32 widx = 0;
            UINT32 distance;
            bool considered = false;
#ifdef REMOVE_DUP_METHOD_1
            if (pathPCDist > REMOVE_DUP_DIST) {
               for (int it2 = 0; it2 < non_dup_consd; it2++) {
                  if (HAConsd[it2] == filteredHA[filteredHistPtr]) {
                     considered = true;
                     break;
                  }
               }
               if (considered == false) {
                  HAConsd[non_dup_consd] = filteredHA[filteredHistPtr];
                  non_dup_consd++;
               }
            }

            for (int it2 = 0; it2 < WL_1+distantHist; it2++) {
               if (HistConsd[it2] == pathPCDist) {
                  considered = true;
                  break;
               }
            }
            if (considered == false) {
               HistConsd[WL_1+distantHist] = pathPCDist;
               distantHist++;
            }
#endif
            if (considered == false) {
#ifdef RECENCY_STACK
               widx = gen_widx(PC, filteredHA[filteredHistPtr], WT_RS); // compute index to access either weight table 
               distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
               widx = widx ^ (distance);
               widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
               if ((pathPCDist-1) < NFOLDED_HIST) {
                  widx = widx ^ folded_hist[pathPCDist-1];
                  widx = widx % (1<<WT_RS);
               }
               else {
                  widx = widx ^ folded_hist[NFOLDED_HIST-1];
                  widx = widx % (1<<WT_RS);
               }
#endif
               if(filteredGHR[filteredHistPtr] == 1)		// If history is Taken
                  accum += weight_rs[widx];
               else 				// If history is Not-Taken
                  accum -= weight_rs[widx];
#endif

               idxConsd[corrFoundCurCycle] = widx;
               dirConsd[corrFoundCurCycle] = filteredGHR[filteredHistPtr];
               corrFoundCurCycle++;

               //distantHist++;
            }
         }
         else if ((filteredHA[filteredHistPtr] == 0) || (pathPCDist >= 1024)) {
            break;
         }
         filteredHistPtr++;
         if (filteredHistPtr == GHL) filteredHistPtr = 0;

      }

      for(int j=corrConsd;((distantHist < WL_2) && (j<WL_2)); j++) {
         int pathPCDist = numFetchedCondBr - nonDupFilteredHADist1[j];
         if ((nonDupFilteredHA1[j] != 0) && (pathPCDist >= 1024) && (pathPCDist <= GHL)){
            UINT32 widx = 0;
            UINT32 distance;
#ifdef RECENCY_STACK
            widx = gen_widx(PC, nonDupFilteredHA1[j], WT_RS); // compute index to access either weight table 
            distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            if ((pathPCDist-1) < NFOLDED_HIST) {
               widx = widx ^ folded_hist[pathPCDist-1];
               widx = widx % (1<<WT_RS);
            }
            else {
               widx = widx ^ folded_hist[NFOLDED_HIST-1];
               widx = widx % (1<<WT_RS);
            }
#endif
            if(nonDupFilteredGHR1[j] == 1)		// If history is Taken
               accum += weight_rs[widx];
            else 				// If history is Not-Taken
               accum -= weight_rs[widx];
#endif
            HistConsd[WL_1+distantHist] = pathPCDist;

            idxConsd[corrFoundCurCycle] = widx;
            dirConsd[corrFoundCurCycle] = nonDupFilteredGHR1[j];
            corrFoundCurCycle++;

            distantHist++;
         }
         else if ((nonDupFilteredHA1[j] == 0) || (pathPCDist > GHL)){
            break;
         }
      }

      pred = (accum >= 0);

   }

   prcpt_pred = pred;
#ifdef LOOPPREDICTOR
   predloop = getloop (PC);	// loop prediction
   pred = ((WITHLOOP >= 0) && (LVALID)) ? predloop : pred;
#endif

   if(accum>-threshold && accum<threshold)
      HTrain=true; // 1 means trainning needed
   else
      HTrain=false;

   Fetch_phist = (Fetch_phist << 1) + (PC & 1);
   Fetch_phist = (Fetch_phist & ((1 << PHISTWIDTH) - 1));

   return ((pred==true)?1:0);
}
    
void  PREDICTOR::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){
            
   bool t = resolveDir;
   int corrConsd = 0;
   //int histPtr = 0;
   //int filteredHistPtr = 0;

#ifdef LOOPPREDICTOR

   if (LVALID)
      if (prcpt_pred != predloop)
         ctrupdate (WITHLOOP, (predloop == resolveDir), 7);
   loopupdate (PC, resolveDir, (prcpt_pred != resolveDir));	//update the loop predictor
#endif
   if (bst_table[PC & ((1 << LOG_BST) - 1)].state == 0) {
      if (resolveDir == true) bst_table[PC & ((1 << LOG_BST) - 1)].state = 1;
      else bst_table[PC & ((1 << LOG_BST) - 1)].state = 2;
   }
   else if (((bst_table[PC & ((1 << LOG_BST) - 1)].state == 1) || (bst_table[PC & ((1 << LOG_BST) - 1)].state == 2)) && (t != ((bst_table[PC & ((1 << LOG_BST) - 1)].state == 1)? true:false))) {
      /*------ detect the current branch as Non-biased ------*/
      bst_table[PC & ((1 << LOG_BST) - 1)].state = 3;
   
      /*------ In the update phase, the predictor table indexes has been computed in the same way as done in the prediction proceude before ------*/
      UINT32 bias_widx = gen_bias_widx(PC, WT_REG);
      if (t == 1) weight_b[bias_widx]++;
      else weight_b[bias_widx]--;

      int non_dup_consd = 0;
      int distantHist = 0;

      for(int j=0, histPtr=GHRHeadPtr; j <WL_1; j++) {
         UINT32 widx = gen_widx(PC, HA[histPtr], WT_REG);
#ifdef FOLDED_GHR_HASH
         widx = widx ^ folded_hist[j];
         widx = widx % (1<<WT_REG);
#endif
         if(t == GHR[histPtr])
         { if(weight_m[widx][j]<63) weight_m[widx][j]++;}
         else if(t != GHR[histPtr])
         { if(weight_m[widx][j]>-64) weight_m[widx][j]--;}

         HistConsd[j] = j+1;

         histPtr++;
         if (histPtr == GHL) histPtr = 0;
      }

      for(int j=WL_1, histPtr=(GHRHeadPtr+WL_1)%GHL;j<REMOVE_DUP_DIST; j++) {
         if ((HA[histPtr] != 0) && (isBrStable[histPtr] == false)){
            UINT32 widx = 0;
            UINT32 distance;
#ifdef RECENCY_STACK
            widx = gen_widx(PC, HA[histPtr], WT_RS); // compute index to access either weight table 
            distance = (j+1) ^ ((j+1) / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            widx = widx ^ folded_hist[j];
            widx = widx % (1<<WT_RS);
#endif
            if(t == GHR[histPtr])
            { if(weight_rs[widx]<63) weight_rs[widx]++;}
            else if(t != GHR[histPtr])
            { if(weight_rs[widx]>-64) weight_rs[widx]--;}
#endif
            HistConsd[WL_1+distantHist] = j+1;
            distantHist++;
         }
         histPtr++;
         if (histPtr == GHL) histPtr = 0;
      }

      for(int j=0;((distantHist < WL_2) && (j<WL_2)); j++, corrConsd++) {
         int pathPCDist = numFetchedCondBr - nonDupFilteredHADist1[j];
         if ((nonDupFilteredHA1[j] != 0) && /*(pathPCDist <= GHL)*/(pathPCDist < 1024)){
            UINT32 widx = 0;
            UINT32 distance;

#ifdef RECENCY_STACK
            widx = gen_widx(PC, nonDupFilteredHA1[j], WT_RS); // compute index to access either weight table 
            distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            if ((pathPCDist-1) < NFOLDED_HIST) {
               widx = widx ^ folded_hist[pathPCDist-1];
               widx = widx % (1<<WT_RS);
            }
            else {
               widx = widx ^ folded_hist[NFOLDED_HIST-1];
               widx = widx % (1<<WT_RS);
            }

#endif
            if(t == nonDupFilteredGHR1[j])
            { if(weight_rs[widx]<63) weight_rs[widx]++;}
            else if(t != nonDupFilteredGHR1[j])
            { if(weight_rs[widx]>-64) weight_rs[widx]--;}
#endif
            HistConsd[WL_1+distantHist] = pathPCDist;
            distantHist++;
         }
         else {
            break;
         }
      }

      non_dup_consd = 0;
      int prevPathPCDist = 0;

      for(int j=0, filteredHistPtr = filteredGHRHeadPtr; ((distantHist < WL_2) && (j < GHL)); j++) {
         int pathPCDist = numFetchedCondBr - filteredHADist[filteredHistPtr];
         if ((pathPCDist > 64) && (filteredHA[filteredHistPtr] != 0) && /*(pathPCDist <= GHL)*/(pathPCDist < 1024)){

            int prevPathPCBin = 0;
            int curPathPCBin = 0;
            for (int it1 = 0; it1 < (NHIST-1); it1++) {
               if ((pathPCDist >= histLen[it1]) && (pathPCDist < histLen[it1+1])) {
                  curPathPCBin = it1+1;
               }
               if ((prevPathPCDist >= histLen[it1]) && (prevPathPCDist < histLen[it1+1])) {
                  prevPathPCBin = it1+1;
               }
            }
            if (prevPathPCBin != curPathPCBin) {
               non_dup_consd = 0;
            }
            prevPathPCDist = pathPCDist;

            UINT32 widx = 0;
            UINT32 distance;
            bool considered = false;
#ifdef REMOVE_DUP_METHOD_1
            if (pathPCDist > REMOVE_DUP_DIST) {
               for (int it2 = 0; it2 < non_dup_consd; it2++) {
                  if (HAConsd[it2] == filteredHA[filteredHistPtr]) {
                     considered = true;
                     break;
                  }
               }
               if (considered == false) {
                  HAConsd[non_dup_consd] = filteredHA[filteredHistPtr];
                  non_dup_consd++;
               }
            }

            for (int it2 = 0; it2 < WL_1+distantHist; it2++) {
               if (HistConsd[it2] == pathPCDist) {
                  considered = true;
                  break;
               }
            }
            if (considered == false) {
               HistConsd[WL_1+distantHist] = pathPCDist;
               distantHist++;
            }
#endif
            if (considered == false) {
#ifdef RECENCY_STACK
               widx = gen_widx(PC, filteredHA[filteredHistPtr], WT_RS); // compute index to access either weight table 
               distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
               widx = widx ^ (distance);
               widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
               if ((pathPCDist-1) < NFOLDED_HIST) {
                  widx = widx ^ folded_hist[pathPCDist-1];
                  widx = widx % (1<<WT_RS);
               }
               else {
                  widx = widx ^ folded_hist[NFOLDED_HIST-1];
                  widx = widx % (1<<WT_RS);
               }

#endif
               if(t == filteredGHR[filteredHistPtr])
               { if(weight_rs[widx]<63) weight_rs[widx]++;}
               else if(t != filteredGHR[filteredHistPtr])
               { if(weight_rs[widx]>-64) weight_rs[widx]--;}
#endif
               //distantHist++;
            }
         }
         else if ((filteredHA[filteredHistPtr] == 0) || (pathPCDist >= 1024)) {
            break;
         }
         filteredHistPtr++;
         if (filteredHistPtr == GHL) filteredHistPtr = 0;

      }

      for(int j=corrConsd;((distantHist < WL_2) && (j<WL_2)); j++) {
         int pathPCDist = numFetchedCondBr - nonDupFilteredHADist1[j];
         if ((nonDupFilteredHA1[j] != 0) && (pathPCDist >= 1024) && (pathPCDist <= GHL)){
            UINT32 widx = 0;
            UINT32 distance;
#ifdef RECENCY_STACK
            widx = gen_widx(PC, nonDupFilteredHA1[j], WT_RS); // compute index to access either weight table 
            distance = pathPCDist ^ (pathPCDist / (1<<WT_RS));
            widx = widx ^ (distance);
            widx = widx % (1<<WT_RS);
#ifdef FOLDED_GHR_HASH
            if ((pathPCDist-1) < NFOLDED_HIST) {
               widx = widx ^ folded_hist[pathPCDist-1];
               widx = widx % (1<<WT_RS);
            }
            else {
               widx = widx ^ folded_hist[NFOLDED_HIST-1];
               widx = widx % (1<<WT_RS);
            }

#endif
            if(t == nonDupFilteredGHR1[j])
            { if(weight_rs[widx]<63) weight_rs[widx]++;}
            else if(t != nonDupFilteredGHR1[j])
            { if(weight_rs[widx]>-64) weight_rs[widx]--;}
#endif
            HistConsd[WL_1+distantHist] = pathPCDist;
            distantHist++;
         }
         else if ((nonDupFilteredHA1[j] == 0) || (pathPCDist > GHL)){
            break;
         }
      }
   }
   else if (((bst_table[PC & ((1 << LOG_BST) - 1)].state == 1) || (bst_table[PC & ((1 << LOG_BST) - 1)].state == 2)) && (t == ((bst_table[PC & ((1 << LOG_BST) - 1)].state == 1)? true:false))) {
   }
   else if( (t != predDir) || (HTrain == true) ) 	//Training needed if threshold not exceeded or predict wrong
   {
      UINT32 bias_widx = gen_bias_widx(PC, WT_REG); // compute index to access either weight table
      if (t == 1) weight_b[bias_widx]++;
      else weight_b[bias_widx]--;

      for(int j = 0; j < WL_1; j++)	{
         if(t == dirConsd[j])
         { if(weight_m[idxConsd[j]][j]<63) weight_m[idxConsd[j]][j]++;}
         else if(t != dirConsd[j])
         { if(weight_m[idxConsd[j]][j]>-64) weight_m[idxConsd[j]][j]--;}
      }
      for(int j = WL_1; j < corrFoundCurCycle; j++) {
         if(t == dirConsd[j])
         { if(weight_rs[idxConsd[j]]<63) weight_rs[idxConsd[j]]++;}
         else if(t != dirConsd[j])
         { if(weight_rs[idxConsd[j]]>-64) weight_rs[idxConsd[j]]--;}
      }
   }

   // threshold adjusting 
   if (bst_table[PC & ((1 << LOG_BST) - 1)].state == 3) {
      if(t != predDir) {
         TC++;
         if(TC==63) {
            TC = 0;
            threshold++;
         }
      }
      else if((t==predDir) && (HTrain == true)) {
         TC--;
         if(TC==-63) {
            TC = 0;
            threshold--;
         }		
      }
   }

   // Compute folded history in each bit position upto NFOLDED_HIST bits based upon the folded history already computed before in the last cycle
   for (int hiter = 0, histPtr=GHRHeadPtr; hiter < WL_1; hiter++) {
      folded_hist[hiter] = (folded_hist[hiter] << 1) | (resolveDir?1:0);
      folded_hist[hiter] ^= ((GHR[histPtr]?1:0) << OUTPOINT[hiter]);
      folded_hist[hiter] ^= (folded_hist[hiter] >> WT_REG);
      folded_hist[hiter] &= (1 << WT_REG) - 1;

      histPtr++;
      if (histPtr == GHL) histPtr = 0;

   }
   for (int hiter = WL_1, histPtr=(GHRHeadPtr+WL_1)%GHL; hiter < NFOLDED_HIST; hiter++) {
      folded_hist[hiter] = (folded_hist[hiter] << 1) | (resolveDir?1:0);
      folded_hist[hiter] ^= ((GHR[histPtr]?1:0) << OUTPOINT[hiter]);
      folded_hist[hiter] ^= (folded_hist[hiter] >> WT_RS);
      folded_hist[hiter] &= (1 << WT_RS) - 1;

      histPtr++;
      if (histPtr == GHL) histPtr = 0;

   }

   // If the current branch is "completely biased or stable" branch?
   bool isStableBranch = (bst_table[PC & ((1 << LOG_BST) - 1)].state != 3)?true:false;

   /*----- Update unfiltered GHR and path history ------*/
   GHRHeadPtr--;
   if (GHRHeadPtr == -1) GHRHeadPtr = GHL-1;
   GHR[GHRHeadPtr] = resolveDir;
   HA[GHRHeadPtr] = PC;
   isBrStable[GHRHeadPtr] = isStableBranch;

   //Update retire path history for loop predictor
   Retire_phist = (Retire_phist << 1) + (PC & 1);
   Retire_phist = (Retire_phist & ((1 << PHISTWIDTH) - 1));

   /*----- Update filtered history used to boost accuracy and provide multiple instances of a branch to the prediction function if required ------*/
   if (isStableBranch == false) {
      filteredGHRHeadPtr--;
      if (filteredGHRHeadPtr == -1) filteredGHRHeadPtr = GHL-1;
      filteredGHR[filteredGHRHeadPtr] = resolveDir?1:0;
      filteredHA[filteredGHRHeadPtr] = PC;
      filteredHADist[filteredGHRHeadPtr]=numFetchedCondBr;
   }

   //if the current branch is Non-baised, insert that into the Recency Stack (RS)
   if (numFetchedCondBr >= REMOVE_DUP_DIST) {
      if (isBrStable[(GHRHeadPtr+REMOVE_DUP_DIST)%GHL] == false) {
         int j = 0;
         for(j=0; j<WL_2; j++)
         {
            if (HA[(GHRHeadPtr+REMOVE_DUP_DIST)%GHL] == nonDupFilteredHA1[j]) {j++; break;}//Find if a prior occurrence is already present in RS
         }
         for (int k = j-1; k>0; k--) {
            nonDupFilteredGHR1[k] = nonDupFilteredGHR1[k-1];
            nonDupFilteredHA1[k]=nonDupFilteredHA1[k-1];
            nonDupFilteredHADist1[k]=nonDupFilteredHADist1[k-1];
         }
         nonDupFilteredGHR1[0] = GHR[(GHRHeadPtr+REMOVE_DUP_DIST)%GHL]?1:0;
         nonDupFilteredHA1[0]=HA[(GHRHeadPtr+REMOVE_DUP_DIST)%GHL];
         nonDupFilteredHADist1[0]=numFetchedCondBr-REMOVE_DUP_DIST;
      }
   }

   numFetchedCondBr++;

}

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
