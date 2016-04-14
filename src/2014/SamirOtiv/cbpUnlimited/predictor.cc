#include <stdio.h>
#include <cassert>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>


#define _INCLUDETAGE_
#include "predictor.h"

my_predictor *mypred;

#define prch(c) printf ("%u ", c)



/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

PREDICTOR::PREDICTOR(void){
    TageInit();
    
    local.InitPatternPredictor();
    
    int i;
    for (i=0; i< PATTROWS; i++){
    select[i][0] = select[i][1] = LOCAL_SEL_RANGE / 2;
    }
    
    
    //STATS
    s_asked = s_correct = l_asked = l_correct = 0;
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

bool   PREDICTOR::GetPrediction(UINT32 PC){
    int PCI = PC % PATTROWS;
    
    //FOR TESTING
    //cout << " (TAGE Prediction = " << TagePredict(PC) << ") ";
    
    if (select[PCI][local.GetTableID(PC)] > LOCAL_SEL_RANGE / 2 && local.IsPatternPredReady(PC)) {
        last = 0;
        l_asked++;
        
        //FOR TESTING
        //cout << "\nPredicted by nBPAT\n";
        
        return local.GetPatternPrediction(PC);
    }
    else {
        last = 1;
        s_asked++;
        
        //FOR TESTING
        //cout << "\nPredicted by TAGE\n";
        
        return TagePredict(PC);
    }
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  PREDICTOR::UpdatePredictor(UINT32 PC, bool taken, bool predDir, UINT32 branchTarget){
    int PCI = PC % PATTROWS;
    
    if (local.IsPatternPredReady(PC)){
        if (TagePredict(PC) != taken && local.GetPatternPrediction(PC) == taken){
            if (select[PCI][local.GetTableID(PC)] < LOCAL_SEL_RANGE) select[PCI][local.GetTableID(PC)]++;
        }
        else if (TagePredict(PC) == taken && local.GetPatternPrediction(PC) != taken){
            if (select[PCI][local.GetTableID(PC)] > 0) select[PCI][local.GetTableID(PC)]--;
        }
    }
    
    //FOR STATS ONLY
    if (taken == predDir){
        if (last == 0) l_correct++;
        if (last == 1) s_correct++;
    }
    
    if (local.TagMatches(PC) || (TagePredict(PC) != taken)) local.UpdatePatternPredictor(PC, taken);    
    TageUpdate(PC, taken, predDir, branchTarget);

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





void PREDICTOR::PrintStats(){
    float spercent, lpercent;
    
    spercent = 100 * (float)s_correct/s_asked;
    lpercent = 100 * (float)l_correct/l_asked;
    
    
    cout << "\n\nFor Gshare:\n";
    cout << "Sat Attempts = " << s_asked;
    cout << "\nSat Successes = " << s_correct;
    cout << "\nSat Strike Rate = " << spercent;
    
    cout << "\n\nFor Local Pat:\n";
    cout << "Pat Attempts = " << l_asked;
    cout << "\nPat Successes = " << l_correct;
    cout << "\nPat Strike Rate = " << lpercent << "\n\n";
}








bool   PREDICTOR::TagePredict(UINT32 PC){
	return mypred->predict_brcond(PC & 0x3ffff, IS_BR_CONDITIONAL);
}


/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

void  PREDICTOR::TageUpdate(UINT32 PC, bool taken, bool predDir, UINT32 branchTarget){
	mypred->FetchHistoryUpdate(PC & 0x3ffff, IS_BR_CONDITIONAL, taken, branchTarget & 0x7f);
	mypred->update_brcond(PC & 0x3ffff, IS_BR_CONDITIONAL, taken, branchTarget & 0x7f);
	return;
}



void PREDICTOR::TageInit(){
    mypred = new my_predictor();
    return;
}




/*-------------------------PATTERN PREDICTOR----------------------*/



void PREDICTOR::patternpredictor::InitPatternPredictor(){
    int i;
    
    for (i=0; i<PATTROWS; i++){
        ready[i][0] = ready[i][1] = 0;
        useful[i][0] = useful[i][1] = 0;
        tag[i][0] = tag[i][1] = 0;
    }
    
    resetcounter = 0;
}



void PREDICTOR::patternpredictor::ResetUseful(){
    UINT32 i;
    
    for (i=0; i<PATTROWS; i++){
        useful[i][0] = useful[i][1] = 0;
    }
    
    resetcounter = 0;
    
    return;
}



bool PREDICTOR::patternpredictor::GetPatternPrediction (UINT32 PC){
    UINT32 PCINDEX = (PC % PATTROWS);
    UINT32* sample;
    
    //Return garbage if not ready
    if (tag[PCINDEX][0] == tagfn(PC) && ready[PCINDEX][0] == 1) sample = &pattern_sample[PCINDEX][0];
    else if (tag[PCINDEX][1] == tagfn(PC) && ready[PCINDEX][1] == 1) sample = &pattern_sample[PCINDEX][1];
    else return 1;
    
    
    int i;
    for (i=1; i<= LOCALHISTORYLENGTH - MAXPLEN; i++){
        if ((*sample ^ (*sample >> i)) % (1 << MAXPLEN) == 0) return ((*sample >> (i-1)) % 2);
    }
    
    //Return garbage if not ready
    return 1;
}


bool PREDICTOR::patternpredictor::IsPatternPredReady (UINT32 PC){
    UINT32 PCINDEX = (PC % PATTROWS);
    return ((ready[PCINDEX][0] && tag[PCINDEX][0] == tagfn(PC)) || (ready[PCINDEX][1] && tag[PCINDEX][1] == tagfn(PC)));
}


bool PREDICTOR::patternpredictor::TagMatches (UINT32 PC){
    UINT32 PCINDEX = (PC % PATTROWS);
    return ((tag[PCINDEX][0] == tagfn(PC)) || (tag[PCINDEX][1] == tagfn(PC)));
}


void PREDICTOR::patternpredictor::UpdatePatternPredictor (UINT32 PC, bool taken){
    if (resetcounter++ == (1 << RESET_RANGE)) ResetUseful();

    UINT32 PCINDEX = (PC % PATTROWS);
    int tableID;
        
    
    if (tag[PCINDEX][0] == tagfn(PC)) {
        tableID = 0;
        if (GetPatternPrediction(PC) != taken || ready[PCINDEX][0] == 0){
            if (useful[PCINDEX][0] > 0) useful[PCINDEX][0]--;
        }
        
        //FOR TESTING
        //cout << " (Tag Match: 0) ";
    }
    else if (tag[PCINDEX][1] == tagfn(PC)) {
        tableID = 1;
        if (GetPatternPrediction(PC) != taken || ready[PCINDEX][1] == 0){
            if (useful[PCINDEX][1] > 0) useful[PCINDEX][1]--;
        }
        
        //FOR TESTING
        //cout << " (Tag Match: 1) ";
    }
    else {
        if (useful[PCINDEX][0] == 0) tableID = 0;
        else if (useful[PCINDEX][1] == 0) tableID = 1;
        else return;
        
        tag[PCINDEX][tableID] = tagfn (PC);
        useful[PCINDEX][tableID] = USEFUL_RANGE - 1;
        ready[PCINDEX][tableID] = 0;
    }
    
    
    
    UINT32* sample = &pattern_sample[PCINDEX][tableID];
    
    *sample = ((*sample << 1) + taken) % (1 << 2 * MAXPLEN);
    
    int i;
    ready [PCINDEX][tableID] = 0;
    for (i=1; i<= LOCALHISTORYLENGTH - MAXPLEN; i++){
        if ((*sample ^ (*sample >> i)) % (1 << MAXPLEN) == 0) ready [PCINDEX][tableID] = 1;
    }
}





void PREDICTOR::patternpredictor::PrintAll (){
    int i;
    UINT32 temp;
    
    UINT32 PCI = (1048575 % PATTROWS);
    
    cout << "##########################################";
    cout << "\nReset Counter: " << resetcounter;
    
    cout << "\n\nPattern Sample 0:\n";
    temp = pattern_sample[PCI][0];
    for (i=0; i<2 * MAXPLEN; i++){
        prch (temp % 2);
        temp = temp >> 1;
    }
    
    cout << "\n\nReady 0 = ";
    prch (ready[PCI][0]);
    
    cout << "\n\nUseful 0 = ";
    prch (useful[PCI][0]);
    
    cout << "\n\nTag 0 = ";
    prch (tag[PCI][0]);
    
    
    cout << "\n\nPattern Sample 1:\n";
    temp = pattern_sample[PCI][1];
    for (i=0; i<2 * MAXPLEN; i++){
        prch (temp % 2);
        temp = temp >> 1;
    }
    
    cout << "\n\nReady 1 = ";
    prch (ready[PCI][1]);
    
    cout << "\n\nUseful 1 = ";
    prch (useful[PCI][1]);
    
    cout << "\n\nTag 1 = ";
    prch (tag[PCI][1]);
    
    
    cout << "\n\n";
        cout << "##########################################\n";
}




UINT32 PREDICTOR::patternpredictor::tagfn (UINT32 PC){
    return ((PC >> LOCALHISTORYLENGTH) % TAGWIDTH);
}




int PREDICTOR::patternpredictor::GetTableID (UINT32 PC){
    UINT32 PCINDEX = (PC % PATTROWS);
    if (tag[PCINDEX][0] == tagfn(PC)) return 0;
    if (tag[PCINDEX][1] == tagfn(PC)) return 1;
    
    return 0;
}
