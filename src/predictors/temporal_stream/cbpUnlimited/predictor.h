#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
// #include "tracer.h"
#include <list>
#include <map>
#include <bitset>

#define PHT_SIZE 0x200000

#define TS_KEY_SIZE (140 + 32)

template<std::size_t N>
struct bitset_less {
  bool operator() (const std::bitset<N>& x, const std::bitset<N>& y) const
  {
    if (N <= 64) return x.to_ulong() < y.to_ulong();

    for (int i = N-1; i >= 0; i--) {
        if (x[i] && !y[i]) return false;
        if (!x[i] && y[i]) return true;
    }
    return false;
  }
};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

class PREDICTOR{

  // The state is defined for Gshare, change for your design

 private:
  UINT32  ghr;           // global history register
  UINT32  pht[PHT_SIZE]; // pattern history table

  bool basepred;
  std::list<char> ts;
  std::list<char>::iterator tshead;
  std::map<bitset<TS_KEY_SIZE>, std::list<char>::iterator, bitset_less<TS_KEY_SIZE> > tstable;
  std::bitset<TS_KEY_SIZE> ts_gh;

 public:

  // The interface to the four functions below CAN NOT be changed

  PREDICTOR(void);
  bool    GetPrediction(UINT32 PC);  
  void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
  void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget);

  // Contestants can define their own functions below
 private:
  bitset<TS_KEY_SIZE> ts_idx(UINT32 PC);

};



/***********************************************************/
#endif

