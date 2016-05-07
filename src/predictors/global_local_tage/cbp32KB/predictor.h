// -*- mode:c++; indent-tabs-mode:nil; -*-

// This source code is derived from ISL-TAGE (CBP-3).
// TAGE is based on the great work from Andre Seznec and Pierre Michaud.
// In this source code, my contribution for the performance is quite small.

// About ISL-TAGE, please refer to previous branch prediction championship.
// URL: http://www.jilp.org/jwac-2/

// About TAGE predictor, please refer to JILP online publication.
// URL: http://www.jilp.org/vol8/v8paper1.pdf

// In this predictor, we tried to combine local branch history to
// improve the performance of TAGE predictor because there are no
// effective way to exploit the local history for the partial tag
// matching. We combine global branch history and local branch
// history for the indexing part of TAGE branch predictor.
// It helps to reduce the branch miss prediction.

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include "utils.h"
// #include "tracer.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

// #define RESTRICT_VERSION

#ifndef RESTRICT_VERSION
// These feature do not affect the accuracy of the submitted branch predictor.
// However, to confirm the temporary variables do not affect the overall
// performance, these variables should be disabled.
  #define FOLDEDINDEXING
  #define NOTCLEARTEMPORARYVARIABLES
#endif

// Enabling / disabling features...
#define STATCOR
#define LOOPPREDICTOR

// Table configuration parameters
#define NSTEP 4
#define NDIFF 3
#define NALLOC 5//4,5,6,7
#define NHIST 20
#define NSTAT 5
#define MSTAT 5//5,6,7
#define TSTAT (NSTAT+MSTAT+1)

// Local global branch history ratio
#define LSTEP 2
#define LG_RATIO 60

// History length settings
#define MINHIST 5
#define MAXHIST 880
#define MINCGHIST 2
#define MAXCGHIST 13
#define MINCLHIST 2
#define MAXCLHIST 12

// Tage parameters
#define HYSTSHIFT 2
#define LOGB 14
#define LOGG 10

// Statistic corrector parameters
#define LOGC 11
#define CBANK 3
#define CSTAT 6

// Loop predictor parameters
#define LOGL 4
#define LOOPWAY 3
#define LOOPTAG 10
#define LOOPITER 10
#define LOOPCONF 3
#define LOOPAGE 3

// Maximum history width
#define PHISTWIDTH 16

// LHT parameters
#define LHTBITS 6
#define LHTSIZE (1<<LHTBITS)
#define LHTMASK (LHTSIZE-1)
#define LHISTWIDTH ((MAXCLHIST>(MAXHIST/LG_RATIO))?MAXCLHIST:(MAXHIST/LG_RATIO))

// Misc counter width
#define DC_WIDTH 10
#define WL_WIDTH 7
#define UC_WIDTH 5
#define UT_WIDTH 6
#define UA_WIDTH 4
#define TK_WIDTH 8

//////////////////////////////////////////////////////
// Base counter class
//////////////////////////////////////////////////////

// Counter base class
template<typename T, int MAX, int MIN>
class Counter {
private:
  T ctr;
public:
  T read(T val=0) { return ctr+val; }
  bool pred() { return ctr >= 0; }
  
  bool satmax(){ return ctr == MAX; }
  bool satmin(){ return ctr == MIN; }
  
  void setmax(){ ctr = MAX; }
  void setmin(){ ctr = MIN; }
  
  void write(T v) {
    assert(v <= MAX);
    assert(v >= MIN);
    ctr = v;
  }
  
  void add(T d) {
    ctr = ctr + d;
    if (ctr > MAX){
      ctr = MAX;
    }else if (ctr < MIN){
      ctr = MIN;
    }
  }
  
  void update(bool incr) {
    if (incr) {
      if (ctr < MAX)
        ctr = ctr + 1;
    } else {
      if (ctr > MIN)
        ctr = ctr - 1;
    }
  }
  
  virtual int budget() = 0;
};

//signed integer counter
template<int WIDTH>
class SCounter : public Counter<int32_t,((1<<(WIDTH-1))-1),(-(1<<(WIDTH-1)))>{
public:
  virtual int budget() {return WIDTH;}
};

//unsigned integer counter
template<int WIDTH>
class UCounter : public Counter<int32_t,((1<<(WIDTH))-1),0>{
public:
  virtual int budget() {return WIDTH;}
};


//////////////////////////////////////////////////////
// history managemet data structure
//////////////////////////////////////////////////////

class GlobalHistoryBuffer {
#ifdef FOLDEDINDEXING
  // This implementation is used to save the simulation time.
  static const int HISTBUFFERLENGTH = 4096;
private:
  int ptr;
  bool bhr[HISTBUFFERLENGTH];
  
public:
  int init() {
    for(int i=0; i<HISTBUFFERLENGTH; i++) { bhr[i] = false; }
    return MAXHIST;
  }
  
  void push(bool taken) {
    ptr--;
    bhr[ptr & (HISTBUFFERLENGTH-1)] = taken;
  }
  
  // read n_th history
  bool read(int n) { return bhr[(n+ptr) & (HISTBUFFERLENGTH-1)]; }
#else
private:
  bool bhr[MAXHIST];

public:
  int init() {
    for(int i=0; i<MAXHIST; i++) { bhr[i] = false; }
    return MAXHIST;
  }

  void push(bool taken) {
    for(int i=MAXHIST-2; i>=0; i--) { bhr[i+1] = bhr[i]; }
    bhr[0] = taken;
  }
  
  bool read(int n) {
    return bhr[n];
  }
#endif
};


class GlobalHistory : public GlobalHistoryBuffer {
  
#ifdef FOLDEDINDEXING
private:
  // Folded index (this register save the hash value of the global history,
  // this values can be regenerated if the all branch histories are stored in the GHR)
  class FoldedHistory {
  public:
    unsigned comp;
    int CLENGTH;
    int OLENGTH;
    int OUTPOINT;
    
    void init (int original_length, int compressed_length) {
      comp = 0;
      OLENGTH = original_length;
      CLENGTH = compressed_length;
      OUTPOINT = OLENGTH % CLENGTH;
    }
    
    void update (GlobalHistoryBuffer *h) {
      comp = (comp << 1) | h->read(0);
      comp ^= (h->read(OLENGTH) ? 1 : 0) << OUTPOINT;
      comp ^= (comp >> CLENGTH);
      comp &= (1 << CLENGTH) - 1;
    }
  };
  FoldedHistory ch_i[NHIST];
  FoldedHistory ch_c[NSTAT];
  FoldedHistory ch_t[3][NHIST];
  
public:
  void updateFoldedHistory() {
    for (int i=0; i<NSTAT; i++) {
      ch_c[i].update(this);
    }
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].update(this);
      ch_t[0][i].update(this);
      ch_t[1][i].update(this);
      ch_t[2][i].update(this);
    }
  }
  
  void setup(int *m, int *l, int *t, int *c, int size) {
    for (int i = 0; i < NHIST; i++) {
      ch_i[i].init(m[i], l[i]);
      ch_t[0][i].init(m[i], t[i]);
      ch_t[1][i].init(m[i], t[i] - 1);
      ch_t[2][i].init(m[i], t[i] - 2);
    }
    for (int i=0; i<NSTAT; i++) {
      ch_c[i].init(c[i], size);
    }
  }
  
  uint32_t gidx(int n, int length, int clength) {
    return ch_i[n].comp;
  }
  
  uint32_t gtag(int n, int length, int clength) {
    return ch_t[0][n].comp^(ch_t[1][n].comp<<1)^(ch_t[2][n].comp<<2);
  }
  
  uint32_t cgidx(int n, int length, int clength) {
    return ch_c[n].comp;
  }
  
#else
  
public:
  void updateFoldedHistory() { }
  
  void setup(int *m, int *l, int *t, int *c, int size) { }
  
  uint32_t comp(int length, int clength) {
    uint32_t comp = 0;
    for(int i=0; i<length; i++) {
      comp ^= (read(i) << (i % clength));
    }
    return comp;
  }
  
  uint32_t gidx(int n, int length, int clength) {
    return comp(length, clength);
  }
  
  uint32_t gtag(int n, int length, int clength) {
    return comp(length,clength)^
      (comp(length,clength-1)<<1)^
      (comp(length,clength-2)<<2);
  }
  
  uint32_t cgidx(int n, int length, int clength) {
    return comp(length, clength);
  }
#endif
  
public:
  void update(bool taken) {
    push(taken);
    updateFoldedHistory();
  }
};


class LocalHistory {
  uint32_t lht[LHTSIZE];
  
  uint32_t getIndex(uint32_t pc) {
    pc = pc ^ (pc >> LHTBITS) ^ (pc >> (2*LHTBITS));
    return pc & (LHTSIZE-1);
  }
  
public:
  int init() {
    for(int i=0; i<LHTSIZE; i++) {
      lht[i] = 0;
    }
    return LHISTWIDTH * LHTSIZE;
  }
  
  void update(uint32_t pc, bool taken) {
    lht[getIndex(pc)] <<= 1;
    lht[getIndex(pc)] |= taken ? 1 : 0;
    lht[getIndex(pc)] &= (1<<LHISTWIDTH) - 1;
  }
  
  uint32_t read(uint32_t pc, int length, int clength) {
    uint32_t h = lht[getIndex(pc)];
    h &= (1 << length) - 1;
    
    uint32_t v = 0;
    while(length > 0) {
      v ^= h;
      h >>= clength;
      length -= clength;
    }
    return v & ((1 << clength) - 1);
  }
};

//////////////////////////////////////////////////////////
// loop predictor (this predictor is derived from CBP-3)
// In this championship, we used 3-way 48-entry
// skewd-associative configuration.
//////////////////////////////////////////////////////////

class LoopPredictorEntry {
public:
  bool dir; // 1-bit
  uint32_t TAG; // 10-bit
  UCounter<LOOPITER> NIter; // 10-bit
  UCounter<LOOPITER> CIter; // 10-bit
  UCounter<LOOPCONF> confid; // 3-bit
  UCounter<LOOPAGE> age; // 3-bit
  
  LoopPredictorEntry () {
    confid.write(0);
    CIter.write(0);
    NIter.write(0);
    age.write(0);
    TAG = 0;
    dir = false;
  }
  
  int init(uint32_t ltag=0, bool taken=0) {
    dir = !taken;
    TAG = ltag;
    NIter.write(0);;
    age.write(7);
    confid.write(0);
    CIter.write(0);

    int budget = 0;
    budget += 1; //dir
    budget += LOOPTAG;
    budget += NIter.budget();
    budget += CIter.budget();
    budget += confid.budget();
    budget += age.budget();

    return budget; // total budget size (37-bit)
  }
  
  // Generate prediction
  bool predict(bool &valid) {
    valid = confid.satmax();
    if((CIter.read() + 1) == NIter.read()) {
      return !dir;
    } else {
      return  dir;
    }
  }
  
  void update(bool taken, bool useful) {
    bool valid;
    bool predloop = predict(valid);
    
    // Update confidence level
    if (valid) {
      if (taken != predloop) {
        NIter.write(0);
        age.write(0);
        confid.write(0);
        CIter.write(0);
        return;
      } else if (useful) {
        age.add(+1);
      }
    }
    
    // Increase the loop count
    CIter.add(+1);
    if (CIter.satmax()) {
      confid.write(7);
    }
    
    // When the loop count perform overflow, the confidence level set to 0
    if (CIter.read() > NIter.read()) {
      confid.write(0);
      NIter.write(0);
    }
    
    // When the direction is different from "dir".
    // checked the loop exit part or not.
    if (taken != dir) {
      bool success = CIter.read() == NIter.read();
      if(success) confid.add(+1);
      else        confid.setmin();
      NIter.write(CIter.read());
      CIter.write(0);
      
      // For short loop, the loop predictor do not applied.
      if (NIter.read() < 3) {
        dir = taken;
        NIter.write(0);
        age.write(0);
        confid.write(0);
      }
    }
  }
};

class LoopPredictor {
  static const int LOOPSIZE = LOOPWAY * (1<<LOGL);
  LoopPredictorEntry table[LOOPSIZE];

  static const int SEEDWIDTH = 6;
  int seed;
  
protected:
  int randomValue() {
    seed++;
    seed &= (1 << SEEDWIDTH) - 1;
    return seed ^ (seed >> 3) ;
  }
  
  // Hash function for index and tag...
  uint32_t getIndex(uint32_t pc, int way) {
    uint32_t v0 = pc & ((1 << (LOGL)) - 1);
    uint32_t v1 = (pc >> LOGL) & ((1 << (LOGL)) - 1);
    return (v0 ^ (v1 >> way)) | (way << LOGL) ;
  }

  uint32_t getTag(uint32_t pc) {
    uint32_t t;
    t = (pc >> (LOGL)) & ((1 << 2 * LOOPTAG) - 1);
    t ^= (t >> LOOPTAG);
    t = (t & ((1 << LOOPTAG) - 1));
    return t;
  }
  
  // Searching hit entry
  int searchEntry(uint32_t pc, uint32_t ltag) {
    for (int i = 0; i < LOOPWAY; i++) {
      int index = getIndex(pc, i);
      if (table[index].TAG == ltag) {
        return index;
      }
    }
    return -1;
  }
  
public:
  
  // Budget counting
  int init() {
    int budget = 0;

    seed = 0;
    budget += SEEDWIDTH;

    for(int i=0; i<LOOPSIZE; i++) {
      budget += table[i].init();
    }
    return budget;
  }
  
  // Generate prediction
  bool predict (uint32_t pc, bool &valid) {
    int LHIT = searchEntry(pc, getTag(pc));
    if(LHIT >= 0) {
      return table[LHIT].predict(valid);
    }
    valid = false;
    return (false);
  }
  
  // Update predictor
  void update (uint32_t pc, bool taken, bool alloc, bool useful) {
    int LHIT = searchEntry(pc, getTag(pc));
    if (LHIT >= 0) {
      useful = useful || ((randomValue() & 7) == 0) ;
      table[LHIT].update(taken, useful);
    } else if (alloc) {
      if ((randomValue() & 3) == 0) {
        uint32_t X = randomValue();
        for (int i = 0; i < LOOPWAY; i++) {
          int index = getIndex(pc, (X + i) % LOOPWAY);
          if (table[index].age.read() == 0) {
            table[index].init(getTag(pc), taken);
            return;
          }
        }
        for (int i = 0; i < LOOPWAY; i++) {
          int index = getIndex(pc, i);
          table[index].age.add(-1);
        }
      }
    }
  }
};


//////////////////////////////////////////////////////////
// Base predictor for TAGE predictor
// This predictor is derived from CBP3 ISL-TAGE
//////////////////////////////////////////////////////////

template <int BITS, int HSFT>
class Bimodal {
private:
  bool pred[1 << BITS];
  bool hyst[1 << (BITS-HSFT)];
  uint32_t getIndex(uint32_t pc, int shift=0) {
    return (pc & ((1 << BITS)-1)) >> shift ;
  }
  
public:
  int init() {
    for(int i=0; i<(1<<BITS); i++) { pred[i] = 0; }
    for(int i=0; i<(1<<(BITS-HSFT)); i++) { hyst[i] = 1; }
    return (1<<BITS)+(1<<(BITS-HSFT));
  }
  
  bool predict(uint32_t pc) {
    return pred[getIndex(pc)];
  }
  
  void update(uint32_t pc, bool taken) {
    int inter = (pred[getIndex(pc)] << 1) + hyst[getIndex(pc, HSFT)];
    if(taken) {
      if (inter < 3) { inter++; }
    } else {
      if (inter > 0) { inter--; }
    }
    pred[getIndex(pc)] = (inter >= 2);
    hyst[getIndex(pc,HSFT)] = ((inter & 1)==1);
  }
};

//////////////////////////////////////////////////////////
// Global component for TAGE predictor
// This predictor is derived from CBP3 ISL-TAGE
//////////////////////////////////////////////////////////

class GEntry {
public:
  uint32_t tag;
  SCounter<3> c;
  UCounter<2> u;
  
  GEntry () {
    tag = 0;
    c.write(0);
    u.write(0);
  }
  
  void init(uint32_t t, bool taken, int uval=0) {
    tag = t;
    c.write(taken ? 0 : -1);
    u.write(uval);
  }
  
  bool newalloc() {
    return (abs(2*c.read() + 1) == 1);
  }
};

//////////////////////////////////////////////////////////
// Put it all together.
// The predictor main component class
//////////////////////////////////////////////////////////

// Configuration of table sharing strategy
static const int STEP[NSTEP+1] = {0, NDIFF, NHIST/2, NHIST-NDIFF, NHIST};
  
class my_predictor {

  /////////////////////////////////////////////////////////
  // Constant Values
  //
  // These variables are not changed during simulation...
  // So, we do not count these variables as the storage.
  /////////////////////////////////////////////////////////
  
  // Tag width and index width of TAGE predictor
  int TB[NHIST];
  int logg[NHIST];
  
  // History length for TAGE predictor
  int m[NHIST];
  int l[NHIST];
  int p[NHIST];
  
  // History length for statistical corrector predictors
  int cg[NSTAT];
  int cp[NSTAT];
  int cl[MSTAT];
  
  /////////////////////////////////////////////////////////
  // Temporary values
  //
  // These values can be computed from the prediction resources,
  // but we use these values to save the simulation time
  /////////////////////////////////////////////////////////
  
  // Index variables of TAGE and some other predictors
  uint32_t CI[TSTAT];
  uint32_t GI[NHIST];
  uint32_t GTAG[NHIST];
  
  // Intermediate prediction result for TAGE
  bool HitPred, AltPred, TagePred;
  int HitBank, AltBank, TageBank;
  
  // Intermediate prediction result for statistical corrector predictor
  bool SCPred;
  int SCSum;
  
  // Intermediate prediction result for loop predictor
  bool loopPred;
  bool loopValid;

  // This function is used for confirming the reset of temporal values
  void clearTemporaryVariables() {
    memset(CI, 0, sizeof(CI));
    memset(GI, 0, sizeof(GI));
    memset(GTAG, 0, sizeof(GTAG));
    SCPred = HitPred = AltPred = TagePred = false;
    SCSum  = HitBank = AltBank = TageBank = 0;
    loopPred = loopValid = 0;
  }
  
  /////////////////////////////////////////////////////////
  // Hardware resoruce
  //
  // These variables are counted as the actual hardware
  // These variables are not exceed the allocated budgeet (32KB + 1K bit)
  /////////////////////////////////////////////////////////
  
  // Prediction Tables
  Bimodal<LOGB,HYSTSHIFT> btable; // bimodal table
  GEntry *gtable[NHIST]; // global components
  SCounter<CSTAT> *ctable[2]; // statistical corrector predictor table
  LoopPredictor ltable; // loop predictor
  
  // Branch Histories
  GlobalHistory ghist; // global history register
  LocalHistory lhist; // local history table
  uint32_t phist; // path history register

  // Profiling Counters
  SCounter<DC_WIDTH> DC; // difficulty counter
  SCounter<WL_WIDTH> WITHLOOP; // loop predictor usefulness
  UCounter<UC_WIDTH> UC; // statistical corrector predictor tracking counter
  SCounter<UT_WIDTH> UT; // statistical corrector predictor threshold counter
  UCounter<TK_WIDTH> TICK; // tick counter for reseting u bit of global entryies
  SCounter<UA_WIDTH> UA[NSTEP+1][NSTEP+1]; // newly allocated entry counter
  
private:
  
  //////////////////////////////////////////////////
  // Setup history length
  //////////////////////////////////////////////////
  
  void setHistoryLength(int *len, int num, int min, int max) {
    for(int i=0; i<num; i++) {
      double a = (double)max / (double)min;
      double j = (double)i/(double)(num-1);
      len[i] = (int)((min * pow(a, j)) + 0.5);
    }
    assert(len[0] == min);
    assert(len[num-1] == max);
  }
  
  void setupHistoryConfiguration() {
    printf("Setup history length...\n");
    setHistoryLength(m, NHIST, MINHIST, MAXHIST);
    setHistoryLength(cg, NSTAT, MINCGHIST, MAXCGHIST);
    setHistoryLength(cl, MSTAT, MINCLHIST, MAXCLHIST);

    for (int i=0; i<NHIST; i++) {
      l[i] = (int)(m[i] / LG_RATIO);
      l[i] = (l[i] > LHISTWIDTH) ? LHISTWIDTH : l[i];
      if(i < STEP[LSTEP]) {
        l[i] = 0;
      }
    }

    cg[0] -= 2;
    cg[2] -= 1;
    cg[3] += 3;
    cg[4] -= 13;
    cl[1] -= 2;
    m[0] -= 5;
    m[1] -= 3;
    m[2] -= 1;
    l[10] -= 1;
    l[11] += 1;
    l[14] -= 1;

    for (int i=0; i<NHIST; i++) {
      p[i] = m[i];
      p[i] = (p[i] > PHISTWIDTH) ? PHISTWIDTH : p[i];
    }
    printf("\n");
    for (int i=0; i<NSTAT; i++) {
      cp[i] = (cg[i] > PHISTWIDTH) ? PHISTWIDTH : cg[i];
    }

    for (int i=0; i<NHIST; i++) {
      if(i > 0) {
        assert(m[i-1] <= m[i]);
        assert(p[i-1] <= p[i]);
        assert(l[i-1] <= l[i]);
      }
      assert(m[i] >= 0);
      assert(m[i] <= MAXHIST);
      assert(p[i] >= 0);
      assert(p[i] <= PHISTWIDTH);
      assert(l[i] >= 0);
      assert(l[i] <= LHISTWIDTH);
    }
    for (int i=0; i<NSTAT; i++) {
      assert(cg[i] >= 0);
      assert(cp[i] >= 0);
      assert(cg[i] <= MAXHIST);
      assert(cp[i] <= MAXHIST);
    }
    for (int i=0; i<MSTAT; i++) {
      assert(cl[i] >= 0);
      assert(cl[i] <= LHISTWIDTH);
    }
    for(int i=0; i<NHIST; i++) printf("m[%d] = %d, l[%d] = %d\n", i, m[i], i, l[i]);
    for(int i=0; i<NSTAT; i++) printf("cg[%d] = %d\n", i, cg[i]);
    for(int i=0; i<MSTAT; i++) printf("cl[%d] = %d\n", i, cl[i]);
  }
  
public:
  
  my_predictor (void) {
    int budget=0;
    
    // Setup history length
    setupHistoryConfiguration();
    
    // Setup misc registers
    WITHLOOP.write(-1);
    DC.write(0);
    UC.write(0);
    UT.write(0);
    TICK.write(0);
    budget += WITHLOOP.budget();
    budget += DC.budget();
    budget += UC.budget();
    budget += UT.budget();
    budget += TICK.budget();

    for(int i=0; i<NSTEP+1; i++) {
      for(int j=0; j<NSTEP+1; j++) {
        UA[i][j].write(0);
        budget += UA[i][j].budget();
      }
    }

    // Setup global components
    logg[STEP[0]] = LOGG + 1;
    logg[STEP[1]] = LOGG + 3;
    logg[STEP[2]] = LOGG + 2;
    logg[STEP[3]] = LOGG - 1;
    TB[STEP[0]] =  7;
    TB[STEP[1]] =  9;
    TB[STEP[2]] = 11;
    TB[STEP[3]] = 13;
    for(int i=0; i<NSTEP; i++) {
      gtable[STEP[i]] = new GEntry[1 << logg[STEP[i]]];
      budget += (2/*U*/+3/*C*/+TB[STEP[i]]) * (1<<logg[STEP[i]]);
    }
    for(int i=0; i<NSTEP; i++) {
      for (int j=STEP[i]+1; j<STEP[i+1]; j++) {
        logg[j]=logg[STEP[i]]-3;
        gtable[j] = gtable[STEP[i]];
        TB[j] = TB[STEP[i]];
      }
    }
    
    // Setup bimodal table
    budget += btable.init();
    
    // Setup statistic corrector predictor
    ctable[0] = new SCounter<CSTAT>[1 << LOGC];
    ctable[1] = new SCounter<CSTAT>[1 << LOGC];
    for(int i=0; i<(1<<LOGC); i++) {
      ctable[0][i].write(-(i&1));
      ctable[1][i].write(-(i&1));
    }
    budget += 2 * CSTAT * (1<<LOGC);
    
    // Setup loop predictor
    budget += ltable.init();
    
    // Setup history register & table
    phist = 0;
    ghist.init();
    lhist.init();
    ghist.setup(m, logg, TB, cg, LOGC-CBANK);
    budget += PHISTWIDTH;
    budget += m[NHIST-1];
    budget += LHISTWIDTH * LHTSIZE;
    
    // Output the total hardware budget
    printf("Total Budget: Limit:%d, %d %d\n", (32*1024*8+1024), budget, budget/8);
  }
  
  
  //////////////////////////////////////////////////////////////
  // Hash functions for TAGE and static corrector predictor
  //////////////////////////////////////////////////////////////
  
  int F (int A, int size, int bank, int width) {
    int A1, A2;
    int rot = (bank+1) % width;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << width) - 1));
    A2 = (A >> width);
    A2 = ((A2 << rot) & ((1 << width) - 1)) + (A2 >> (width - rot));
    A = A1 ^ A2;
    A = ((A << rot) & ((1 << width) - 1)) + (A >> (width - rot));
    return (A);
  }
  
  // gindex computes a full hash of pc, ghist and phist
  uint32_t gindex(uint32_t pc, int bank, int hist) {
    // we combine local branch history for the TAGE index computation
    uint32_t index =
      lhist.read(pc, l[bank], logg[bank]) ^
      ghist.gidx(bank, m[bank], logg[bank]) ^
      F(hist, p[bank], bank, logg[bank]) ^
      (pc >> (abs (logg[bank] - bank) + 1)) ^ pc ;
    return index & ((1 << logg[bank]) - 1);
  }
  
  //  tag computation for TAGE predictor
  uint32_t gtag(uint32_t pc, int bank) {
    uint32_t tag = ghist.gtag(bank, m[bank], TB[bank]) ^ pc ;
    return (tag & ((1 << TB[bank]) - 1));
  }

  // index computation for statistical corrector predictor
  uint32_t cgindex (uint32_t pc, int bank, int hist, int size) {
    uint32_t index =
      ghist.cgidx(bank, cg[bank], size) ^
      F(hist, cp[bank], bank, size) ^
      (pc >> (abs (size - (bank+1)) + 1)) ^ pc ;
    return index & ((1 << size) - 1);
  }
  
  // index computation for statistical corrector predictor
  uint32_t clindex (uint32_t pc, int bank, int size) {
    uint32_t index =
      lhist.read(pc, cl[bank], size) ^
      (pc >> (abs (size - (bank+1)) + 1)) ^ pc ;
    return index & ((1 << size) - 1);
  }
  
  // index computation for usefulness of AltPred counters
  uint32_t uaindex(int bank) {
    for(int i=0; i<NSTEP; i++) {
      if(bank < STEP[i]) return i;
    }
    return NSTEP;
  }
  
  //////////////////////////////////////////////////////////////
  // Actual branch prediction and training algorithm
  //////////////////////////////////////////////////////////////
  
  //compute the prediction
  bool predict(uint32_t pc, uint16_t brtype) {
    // Final prediction result
    bool pred_taken = true;
    
    if (brtype == OPTYPE_BRANCH_COND) {
#ifndef NOTCLEARTEMPORARYVARIABLES
      clearTemporaryVariables();
#endif

      // Compute index values
      for (int i = 0; i < NHIST; i++) {
        GI[i] = gindex(pc, i, phist);
        GTAG[i] = gtag(pc, i);
      }
      
      // Update the index values for interleaving
      for (int s=0; s<NSTEP; s++) {
        for (int i=STEP[s]+1; i<STEP[s+1]; i++) {
          GI[i]=((GI[STEP[s]]&7)^(i-STEP[s]))+(GI[i]<<3);
        }
      }
      
      // Compute the prediction result of TAGE predictor
      HitBank = AltBank = -1;
      HitPred = AltPred = btable.predict(pc);
      for (int i=0; i<NHIST; i++) {
        if (gtable[i][GI[i]].tag == GTAG[i]) {
          AltBank = HitBank;
          HitBank = i;
          AltPred = HitPred;
          HitPred = gtable[i][GI[i]].c.pred();
        }        
      }
      
      // Select the highest confident prediction result
      TageBank = HitBank;
      TagePred = HitPred;
      if (HitBank >= 0) {
        int u = UA[uaindex(HitBank)][uaindex(AltBank)].read();
        if((u>=0)&&gtable[HitBank][GI[HitBank]].newalloc()) {
          TagePred = AltPred;
          TageBank = AltBank;
        }
      }
      pred_taken = TagePred;
      
#ifdef STATCOR
      // Compute the index values of the static corrector predictor
      CI[0] = pc & ((1<<LOGC)-1);
      for (int i=0; i<NSTAT; i++) {
        CI[i+1] = cgindex(pc, i, phist, LOGC-CBANK);
      }
      for (int i=0; i<MSTAT; i++) {
        CI[i+NSTAT+1] = clindex(pc, i, LOGC-CBANK);
      }
      for (int i=0; i<TSTAT; i++) {
        if (i == (NSTAT)) { CI[i] ^= (TageBank+1) ; }
        CI[i] <<= CBANK;
        CI[i] ^= (pc ^ i) & ((1 << CBANK)-1);
        CI[i] &= (1<<LOGC) - 1;
      }
      
      // Overwrite TAGE prediction result if the confidence of
      // the static corrector predictor is higher than the threshold.
      if (HitBank >= 0) {
        SCSum = 8 * (2 * gtable[HitBank][GI[HitBank]].c.read() + 1);
        for (int i=0; i < TSTAT; i++) {
          SCSum += (2 * ctable[TagePred][CI[i]].read()) + 1;
        }
        SCPred = (SCSum >= 0);
        if (abs (SCSum) >= UC.read(5)) {
          pred_taken = SCPred;
        }
      }
#endif
      
#ifdef LOOPPREDICTOR
      loopPred = ltable.predict (pc, loopValid);
      if((WITHLOOP.pred()) && (loopValid)) {
        pred_taken = loopPred ;
      }
#endif
    }
    
    return pred_taken;
  }
  
  void update(uint32_t pc, uint16_t brtype, bool taken, uint32_t target) {
    if (brtype == OPTYPE_BRANCH_COND) {
#ifndef NOTCLEARTEMPORARYVARIABLES
      clearTemporaryVariables();
      predict(pc, brtype); // re-calculate temporal variables
#endif

#ifdef LOOPPREDICTOR
      // Update loop predictor and usefulness counter
      if (loopValid && (TagePred != loopPred)) {
        WITHLOOP.update(loopPred == taken);
      }
      ltable.update (pc, taken, TagePred != taken, loopPred != TagePred);
#endif
      
#ifdef STATCOR
      if (HitBank >= 0) {
        // Updates the threshold of the static corrector predictor
        if (TagePred != SCPred) {
          if ( (abs (SCSum) >= UC.read(1)) &&
               (abs (SCSum) <= UC.read(3)) ) {
            UT.update(SCPred == taken);
            if (UT.satmax() && !UC.satmin()) {
              UT.write(0);
              UC.add(-2);
            }
            if (UT.satmin() && !UC.satmax()) {
              UT.write(0);
              UC.add(+2);
            }
          }
        }
        
        // Updates the static corrector predictor tables
        if ((SCPred != taken) || (abs (SCSum) < (21 + 8 * UC.read(5)))) {
          for (int i=0; i<TSTAT; i++) {
            ctable[TagePred][CI[i]].update(taken);
          }
        }
      }
#endif
      
      // Determining the allocation of new entries
      bool ALLOC = (TagePred != taken) && (HitBank < (NHIST-1));
      if (HitBank >= 0) {
        if (gtable[HitBank][GI[HitBank]].newalloc()) {
          if (HitPred == taken) {
            ALLOC = false;
          }
          if (HitPred != AltPred) {
            UA[uaindex(HitBank)][uaindex(AltBank)].update(AltPred == taken);
          }
        }
      }
      
      
      if (ALLOC) {
        // Allocate new entries up to "NALLOC" entries are allocated
        int T = 0;
        for (int i=HitBank+1; i<NHIST; i+=1) {
          if (gtable[i][GI[i]].u.read() == 0) {
            gtable[i][GI[i]].init(GTAG[i], taken, 0);
            TICK.add(-1);
            if (T == NALLOC) break;
            T += 1;
            i += 1 + T/2; // After T th allocation, we skip 1+(T/2) tables.
          } else {
            TICK.add(+1);
          }
        }
        
        // Reset useful bit to release OLD useful entries
        // When the accuracy of the predictor is lower than pre-defined
        // threshold, we aggressively reset the useful counters.
        bool resetUbit = ((T == 0) && DC.pred()) || TICK.satmax();
        if (resetUbit) {
          TICK.write(0);
          for (int s=0; s<NSTEP; s++) {
            for (int j=0; j<(1<<logg[STEP[s]]); j++)
              gtable[STEP[s]][j].u.add(-1);
          }
        }
      }
      
      // Tracking prediction difficulty of the current workload
      // This counter represent the prediction accuracy of the
      // TAGE branch predictor.
      DC.add((TagePred == taken) ? -1 : 32);
      
      // Update prediction tables
      // This part is same with ISL-TAGE branch predictor.
      if (HitBank >= 0) {
        gtable[HitBank][GI[HitBank]].c.update(taken);
        if ((gtable[HitBank][GI[HitBank]].u.read() == 0)) {
          if (AltBank >= 0) {
            gtable[AltBank][GI[AltBank]].c.update(taken);
          } else {
            btable.update(pc, taken);
          }
        }
      } else {
        btable.update(pc, taken);
      }
      
      // Update useful bit counter
      // This useful counter updating strategy is derived from
      // Re-reference interval prediction.
      if (HitBank >= 0) {
        bool useful = (HitPred == taken) && (AltPred != taken) ;
        if(useful) {
          gtable[HitBank][GI[HitBank]].u.setmax();
        }
      }
    }

    //////////////////////////////////////////////////
    // Branch history management.
    //////////////////////////////////////////////////

    // How many history bits are inserted for the buffer
    int maxt = 1;

    // Special treamtment for function call
    if (brtype == OPTYPE_INDIRECT_BR_CALL) maxt = 3 ;
    if (brtype == OPTYPE_CALL_DIRECT     ) maxt = 5 ;

    // Branch history information
    // (This feature is derived from ISL-TAGE)
    int T = ((target ^ (target >> 3) ^ pc) << 1) + taken;
    
    // Update global history and path history
    for (int t = 0; t < maxt; t++) {
      ghist.update((T >> t) & 1);
      phist <<= 1;
      phist += (pc >> t) & 1;
      phist &= (1 << PHISTWIDTH) - 1;
    }
    
    // Update local history
    lhist.update(pc, taken);
  }
  
};

/////////////////////////////////////////////////////////////

class PREDICTOR{

 private:
  my_predictor *tage;

 public:

  // The interface to the four functions below CAN NOT be changed

  PREDICTOR(void);
  bool    GetPrediction(UINT32 PC);  
  void    UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget);
  void    TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget);

  // Contestants can define their own functions below

};

/////////////////////////////////////////////////////////////

#endif

