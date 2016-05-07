#include "predictor.h"
#include <math.h>


/////////////// STORAGE BUDGET JUSTIFICATION ////////////////

// unlimited size

/////////////////////////////////////////////////////////////


#define ASSERT(cond) if (!(cond)) {fprintf(stderr,"file %s assert line %d\n",__FILE__,__LINE__); abort();}


#define DECPTR(ptr,size)                        \
{                                               \
  ptr--;                                        \
  if (ptr==(-1)) {                              \
    ptr = size-1;                               \
  }                                             \
}

#define INCSAT(ctr,max) {if (ctr < (max)) ctr++;}

#define DECSAT(ctr,min) {if (ctr > (min)) ctr--;}



bool
updctr(int8_t & ctr, bool inc, int nbits)
{
  ASSERT(nbits<=8);
  int ctrmin = -(1 << (nbits-1));
  int ctrmax = -ctrmin-1;
  bool issat = (ctr==ctrmax) || (ctr==ctrmin);
  if (inc) {
    INCSAT(ctr,ctrmax);
  } else {
    DECSAT(ctr,ctrmin);
  }
  return issat && ((ctr==ctrmax) || (ctr==ctrmin));
}


void
path_history::init(int hlen)
{
  hlength = hlen;
  h = new unsigned [hlen];
  for (int i=0; i<hlength; i++) {
    h[i] = 0;
  }
  ptr = 0;
}


void 
path_history::insert(unsigned val)
{
  DECPTR(ptr,hlength);
  h[ptr] = val;
}


unsigned & 
path_history::operator [] (int n)
{
  ASSERT((n>=0) && (n<hlength));
  int k = ptr + n;
  if (k >= hlength) {
    k -= hlength;
  }
  ASSERT((k>=0) && (k<hlength));
  return h[k];
}


compressed_history::compressed_history()
{
  reset();
}


void
compressed_history::reset()
{
  comp = 0; // must be consistent with path_history::reset()
}


void 
compressed_history::init(int original_length, int compressed_length, int injected_bits)
{
  olength = original_length;
  clength = compressed_length;
  nbits = injected_bits;
  outpoint = olength % clength;
  ASSERT(clength < 32);
  ASSERT(nbits <= clength);
  mask1 = (1<<clength)-1;
  mask2 = (1<<nbits)-1;
  reset();
}


void 
compressed_history::rotateleft(unsigned & x, int m)
{
  ASSERT(m < clength);
  ASSERT((x>>clength) == 0);
  unsigned y = x >> (clength-m);
  x = (x << m) | y;
  x &= mask1;
}


void 
compressed_history::update(path_history & ph)
{
  rotateleft(comp,1);
  unsigned inbits = ph[0] & mask2;
  unsigned outbits = ph[olength] & mask2;
  rotateleft(outbits,outpoint);
  comp ^= inbits ^ outbits;
}


coltentry::coltentry()
{
  for (int i=0; i<(1<<NPRED); i++) {
    c[i] = ((i>>(NPRED-1)) & 1)? 1:-2;
  }
}


int8_t & 
coltentry::ctr(bool predtaken[NPRED])
{
  int v = 0;
  for (int i=0; i<NPRED; i++) {
    v = (v << 1) | ((predtaken[i])? 1:0);
  }
  return c[v];
}


int8_t & 
colt::ctr(UINT32 pc, bool predtaken[NPRED])
{
  int i = pc & ((1<<LOGCOLT)-1);
  return c[i].ctr(predtaken);
}


bool
colt::predict(UINT32 pc, bool predtaken[NPRED])
{
  return (ctr(pc,predtaken) >= 0);
}


void 
colt::update(UINT32 pc, bool predtaken[NPRED], bool taken)
{
  updctr(ctr(pc,predtaken),taken,COLTBITS);
}


bftable::bftable()
{
  for (int i=0; i<BFTSIZE; i++) {
    freq[i] = 0;
  }
}


int & 
bftable::getfreq(UINT32 pc)
{
  int i = pc % BFTSIZE;
  ASSERT((i>=0) && (i<BFTSIZE));
  return freq[i];
}



void 
subpath::init(int ng, int hist[], int logg, int tagbits, int pathbits, int hp)
{
  ASSERT(ng>0);
  numg = ng;
  ph.init(hist[numg-1]+1);
  chg = new compressed_history [numg];
  chgg = new compressed_history [numg];
  cht = new compressed_history [numg];
  chtt = new compressed_history [numg];
  int ghlen = 0;
  for (int i=numg-1; i>=0; i--) {
    ghlen = (ghlen < hist[numg-1-i]) ? hist[numg-1-i] : ghlen+1;
    chg[i].init(ghlen,logg,pathbits);
    chgg[i].init(ghlen,logg-hp,pathbits);
    cht[i].init(ghlen,tagbits,pathbits);
    chtt[i].init(ghlen,tagbits-1,pathbits);
  }
}


void 
subpath::init(int ng, int minhist, int maxhist, int logg, int tagbits, int pathbits, int hp)
{
  int * h = new int [ng];
  for (int i=0; i<ng; i++) {
    h[i] = minhist * pow((double)maxhist/minhist,(double)i/(ng-1));
  }
  init(ng,h,logg,tagbits,pathbits,hp);
}



void 
subpath::update(UINT32 targetpc, bool taken)
{
  ph.insert((targetpc<<1)|taken);
  for (int i=0; i<numg; i++) {
    chg[i].update(ph);
    chgg[i].update(ph);
    cht[i].update(ph);
    chtt[i].update(ph);
  }
}


unsigned 
subpath::cg(int bank)
{
  ASSERT((bank>=0) && (bank<numg));
  return chg[bank].comp;
}


unsigned 
subpath::cgg(int bank)
{
  ASSERT((bank>=0) && (bank<numg));
  return chgg[bank].comp << (chg[bank].clength-chgg[bank].clength);
}


unsigned 
subpath::ct(int bank)
{
  ASSERT((bank>=0) && (bank<numg));
  return cht[bank].comp;
}


unsigned 
subpath::ctt(int bank)
{
  ASSERT((bank>=0) && (bank<numg));
  return chtt[bank].comp << (cht[bank].clength-chtt[bank].clength);
}


spectrum::spectrum()
{
  size = 0;
  p = NULL;
}


void
spectrum::init(int sz, int ng, int minhist, int maxhist, int logg, int tagbits, int pathbits, int hp)
{
  size = sz;
  p = new subpath [size];
  for (int i=0; i<size; i++) {
    p[i].init(ng,minhist,maxhist,logg,tagbits,pathbits,hp);
  }
}


void
freqbins::init(int nb)
{
  nbins = nb;
  maxfreq = 0;
}


int
freqbins::find(int bfreq)
{
  // find in which frequency bin the input branch frequency falls
  ASSERT(bfreq>=0);
  int b = -1;
  int f = maxfreq;
  for (int i=0; i<nbins; i++) {
    f = f >> FRATIOBITS;
    if (bfreq >= f) {
      b = i;
      break; 
    }
  }
  if (b < 0) {
    b = nbins-1;
  }
  return b;
}


void 
freqbins::update(int bfreq)
{
  if (bfreq > maxfreq) {
    ASSERT(bfreq==(maxfreq+1));
    maxfreq = bfreq;
  }
}


gentry::gentry()
{
  ctr = 0;
  tag = 0;
  u = 0;
}


potage::potage()
{
  b = NULL;
  g = NULL;
  gi = NULL;
  postp = NULL;
  nmisp = 0;
}


potage::~potage()
{
#ifdef VERBOSE
  printf("%s nmisp = %d\n",name.c_str(),nmisp);
#endif
}


void 
potage::init(const char * nm, int ng, int logb, int logg, int tagb, int ctrb, int ppb, int ru, int caph)
{
  ASSERT(ng>1);
  ASSERT(logb<30);
  ASSERT(logg<30);
  name = nm;
  numg = ng;
  bsize = 1 << logb;
  gsize = 1 << logg;
  tagbits = tagb;
  ctrbits = ctrb;
  postpbits = ppb;
  postpsize = 1 << ((1+POSTPEXTRA)*ctrbits+1);
  b = new int8_t [bsize];
  for (int i=0; i<bsize; i++) {
    b[i] = 0;
  }
  g = new gentry * [numg];
  for (int i=0; i<numg; i++) {
    g[i] = new gentry [gsize];
  }
  gi = new int [numg];
  postp = new int8_t [postpsize];
  for (int i=0; i<postpsize; i++) {
    postp[i] = -(((i>>1) >> (ctrbits-1)) & 1);
  }
  allocfail = 0;
  rampup = ru;
  caphist = caph;
}


int 
potage::bindex(UINT32 pc)
{
  return pc & (bsize-1);
}


int 
potage::gindex(UINT32 pc, subpath & p, int bank)
{
  return (pc ^ p.cg(bank) ^ p.cgg(bank)) & (gsize-1);
}


int 
potage::gtag(UINT32 pc, subpath & p, int bank)
{
  return (pc ^ p.ct(bank) ^ p.ctt(bank)) & ((1<<tagbits)-1);
}


int
potage::postp_index()
{
  // post predictor index function
  int ctr[POSTPEXTRA+1];
  for (int i=0; i<=POSTPEXTRA; i++) {
    ctr[i] = (i < (int)hit.size())? getg(hit[i]).ctr : b[bi];;
  }
  int v = 0;
  for (int i=POSTPEXTRA; i>=0; i--) {
    v = (v << ctrbits) | (ctr[i] & (((1<<ctrbits)-1)));
  }
  int u0 = (hit.size()>0)? getg(hit[0]).u : 1;
  v = (v << 1) | u0;
  v &= postpsize-1;
  return v;
}


gentry &
potage::getg(int i)
{
  ASSERT((i>=0) && (i<numg));
  return g[i][gi[i]];
}


bool 
potage::condbr_predict(UINT32 pc, subpath & p)
{
  hit.clear();
  bi = bindex(pc);
  for (int i=0; i<numg; i++) {
    gi[i] = gindex(pc,p,i);
    if (g[i][gi[i]].tag == gtag(pc,p,i)) {
      hit.push_back(i);
    }
  }
  predtaken = (hit.size()>0)? (getg(hit[0]).ctr>=0) : (b[bi]>=0); 
  altpredtaken = (hit.size()>1)? (getg(hit[1]).ctr>=0) : (b[bi]>=0);
  ppi = postp_index();
  ASSERT(ppi<postpsize);
  postpredtaken = (postp[ppi] >= 0);
  return postpredtaken;
}


void
potage::uclear()
{
  for (int i=0; i<numg; i++) {
    for (int j=0; j<gsize; j++) {
      g[i][j].u = 0;
    }
  }
}


void
potage::galloc(int i, UINT32 pc, bool taken, subpath & p)
{
  getg(i).tag = gtag(pc,p,i);
  getg(i).ctr = (taken)? 0 : -1;
  getg(i).u = 0;
}


void
potage::aggressive_update(UINT32 pc, bool taken, subpath & p)
{
  // update policy used during ramp up
  bool allsat = true;
  for (int i=0; i<(int)hit.size(); i++) {
    allsat &= updctr(getg(hit[i]).ctr,taken,ctrbits);
  }
  if (hit.size()==0) {
    allsat = updctr(b[bi],taken,ctrbits);
  }

  int i = (hit.size()>0)? hit[0] : numg;
  while (--i >= 0) {
    if (getg(i).u != 0) continue;
    if (! allsat || (p.chg[i].olength <= caphist)) {
      galloc(i,pc,taken,p);
    }
  }
}


void
potage::careful_update(UINT32 pc, bool taken, subpath & p)
{
  // update policy devised by Andre Seznec for the ISL-TAGE predictor (MICRO 2011)
  if (hit.size()>0) {
    updctr(getg(hit[0]).ctr,taken,ctrbits);
    if (getg(hit[0]).u==0) {
      if (hit.size()>1) {
  	updctr(getg(hit[1]).ctr,taken,ctrbits);
      } else {
  	updctr(b[bi],taken,ctrbits);
      }
    }
  } else {
    updctr(b[bi],taken,ctrbits);
  }

  if (mispred) {
    int nalloc = 0;
    int i = (hit.size()>0)? hit[0] : numg;
    while (--i >= 0) {
      if (getg(i).u == 0) {
	galloc(i,pc,taken,p);
	DECSAT(allocfail,0);
	i--;
	nalloc++;
	if (nalloc==MAXALLOC) break;
      } else {
	INCSAT(allocfail,ALLOCFAILMAX);
	if (allocfail==ALLOCFAILMAX) {
	  uclear();
	}
      }
    }
  }

}


bool
potage::condbr_update(UINT32 pc, bool taken, subpath & p)
{
  mispred = (postpredtaken != taken);

  if (mispred) {
    nmisp++;
  }

  if (nmisp < rampup) {
    aggressive_update(pc,taken,p);
  } else {
    careful_update(pc,taken,p);
  }

  // update u bit (see TAGE, JILP 2006)
  if (predtaken != altpredtaken) {
    ASSERT(hit.size()>0);
    if (predtaken == taken) {
      getg(hit[0]).u = 1;
    } else {
      getg(hit[0]).u = 0;
    }
  }

  // update post pred
  updctr(postp[ppi],taken,postpbits);

  return mispred;
}


void 
potage::printconfig(subpath & p)
{
  printf("%s path lengths: ",name.c_str());
  for (int i=numg-1; i>=0; i--) {
    printf("%d ",p.chg[i].olength);
  }
  printf("\n");
}



/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////


PREDICTOR::PREDICTOR(void)
{
  sp[0].init(P0_SPSIZE,P0_NUMG,P0_MINHIST,P0_MAXHIST,P0_LOGG,TAGBITS,PATHBITS,P0_HASHPARAM);
  sp[1].init(P1_SPSIZE,P1_NUMG,P1_MINHIST,P1_MAXHIST,P1_LOGG,TAGBITS,PATHBITS,P1_HASHPARAM);
  sp[2].init(P2_SPSIZE,P2_NUMG,P2_MINHIST,P2_MAXHIST,P2_LOGG,TAGBITS,PATHBITS,P2_HASHPARAM);
  sp[3].init(P3_SPSIZE,P3_NUMG,P3_MINHIST,P3_MAXHIST,P3_LOGG,TAGBITS,PATHBITS,P3_HASHPARAM);
  sp[4].init(P4_SPSIZE,P4_NUMG,P4_MINHIST,P4_MAXHIST,P4_LOGG,TAGBITS,PATHBITS,P4_HASHPARAM);

  pred[0].init("G",P0_NUMG,P0_LOGB,P0_LOGG,TAGBITS,CTRBITS,POSTPBITS,P0_RAMPUP,CAPHIST);
  pred[1].init("A",P1_NUMG,P1_LOGB,P1_LOGG,TAGBITS,CTRBITS,POSTPBITS,P1_RAMPUP,CAPHIST);
  pred[2].init("S",P2_NUMG,P2_LOGB,P2_LOGG,TAGBITS,CTRBITS,POSTPBITS,P2_RAMPUP,CAPHIST);
  pred[3].init("s",P3_NUMG,P3_LOGB,P3_LOGG,TAGBITS,CTRBITS,POSTPBITS,P3_RAMPUP,CAPHIST);
  pred[4].init("F",P4_NUMG,P4_LOGB,P4_LOGG,TAGBITS,CTRBITS,POSTPBITS,P4_RAMPUP,CAPHIST);

  bfreq.init(P4_SPSIZE); // number of frequency bins = P4 spectrum size

#ifdef VERBOSE
  for (int i=0; i<NPRED; i++) {
    pred[i].printconfig(sp[i].p[0]);
  }
#endif
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool   
PREDICTOR::GetPrediction(UINT32 PC)
{
  subp[0] = & sp[0].p[0]; // global path
  subp[1] = & sp[1].p[PC % P1_SPSIZE]; // per-address subpath
  subp[2] = & sp[2].p[(PC>>P2_PARAM) % P2_SPSIZE]; // per-set subpath
  subp[3] = & sp[3].p[(PC>>P3_PARAM) % P3_SPSIZE]; // another per-set subpath
  int f = bfreq.find(bft.getfreq(PC));
  ASSERT((f>=0) && (f<P4_SPSIZE));
  subp[4] = & sp[4].p[f]; // frequency subpath 

  for (int i=0; i<NPRED; i++) {
    predtaken[i] = pred[i].condbr_predict(PC,*subp[i]);
  }
  return co.predict(PC,predtaken);
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  
PREDICTOR::UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget)
{
  for (int i=0; i<NPRED; i++) {
    pred[i].condbr_update(PC,resolveDir,*subp[i]);
    subp[i]->update(branchTarget,resolveDir);
  }

  co.update(PC,predtaken,resolveDir);

  bfreq.update(bft.getfreq(PC));
  bft.getfreq(PC)++;
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////


void    
PREDICTOR::TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget)
{
  switch (opType) {
  case OPTYPE_CALL_DIRECT:
  case OPTYPE_RET:
  case OPTYPE_BRANCH_UNCOND:
  case OPTYPE_INDIRECT_BR_CALL:
    // also update the global path with unconditional branches
    sp[0].p[0].update(branchTarget,true);
    break;
  default: break;
  }
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
