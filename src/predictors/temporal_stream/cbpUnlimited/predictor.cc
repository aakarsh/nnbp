#include "predictor.h"

#define PHT_CTR_MAX  3
#define PHT_CTR_INIT 2

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

PREDICTOR::PREDICTOR(void){

  ghr = 0;

  for(UINT32 ii = 0; ii < PHT_SIZE; ii++){
    pht[ii]=PHT_CTR_INIT; 
  }

  tshead = ts.end();
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool   PREDICTOR::GetPrediction(UINT32 PC){

  UINT32 phtIndex = (PC^ghr) % PHT_SIZE;
  UINT32 phtCounter = pht[phtIndex];
  
  if(phtCounter > PHT_CTR_MAX/2){
    basepred = TAKEN; 
  }else{
    basepred = NOT_TAKEN; 
  }

  if (tshead != ts.end()) {
    ++tshead;
    if (!*tshead) {
      return !basepred;
    }
  }
  return basepred;
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bitset<TS_KEY_SIZE> PREDICTOR::ts_idx(UINT32 PC) {
    bitset<TS_KEY_SIZE> pc = bitset<TS_KEY_SIZE>(PC);
    return pc | (ts_gh << 32);
}

void  PREDICTOR::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget){

  // cerr << PC << ":" << basepred << "," << predDir << ";" << resolveDir << endl;

  UINT32 phtIndex   = (PC^ghr) % PHT_SIZE;
  UINT32 phtCounter = pht[phtIndex];

  // update the PHT
  if (resolveDir == TAKEN) {
    pht[phtIndex] = SatIncrement(phtCounter, PHT_CTR_MAX);
  } else {
    pht[phtIndex] = SatDecrement(phtCounter);
  }

  // update the GHR
  ghr = (ghr << 1);

  if(resolveDir == TAKEN){
    ghr++; 
  }

  /* ts stuff */
  // update ts_gh
  ts_gh <<= 1;
  if(resolveDir == TAKEN){
    ts_gh[0] = 1;
  }

  // update ts
  ts.push_back(basepred == resolveDir);

  if (tshead != ts.end() && resolveDir != predDir) { // replay starts to fail, stop it.
    tshead = ts.end();
  }

  if (basepred != resolveDir) { // base make a mistake

    bitset<TS_KEY_SIZE> idx = ts_idx(PC);

    if (tshead == ts.end()) { // not replaying, try start replay
      std::map<bitset<TS_KEY_SIZE>, std::list<char>::iterator, bitset<TS_KEY_SIZE> >::iterator pair = tstable.find(idx);
      if (pair != tstable.end()) tshead = pair->second;
    }

    // record location
    tstable[idx] = ts.end();
    --tstable[idx];
  }

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
