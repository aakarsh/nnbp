///////////////////////////////////////////////////////////////////////
//  Copyright 2015 Samsung Austin Semiconductor, LLC.                //
///////////////////////////////////////////////////////////////////////

//Description : Main file for CBP2016 

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <map>
using namespace std;

#include "utils.h"
#include "bt9.h"
#include "bt9_reader.h"
#include "predictor.h"


#define COUNTER     unsigned long long



void CheckHeartBeat(UINT64 numIter)
{
  UINT64 dotInterval=1000000;
  UINT64 lineInterval=30*dotInterval;

  if(numIter % dotInterval == 0){
    printf("."); 
    fflush(stdout);
  }

  if(numIter % lineInterval == 0){
    printf("\n");
    fflush(stdout);
  }

}//void CheckHeartBeat

// usage: predictor <trace>

int main(int argc, char* argv[]){
  
  if (argc != 2) {
    printf("usage: %s <trace>\n", argv[0]);
    exit(-1);
  }
  
  ///////////////////////////////////////////////
  // Init variables
  ///////////////////////////////////////////////
    
    PREDICTOR  *brpred = new PREDICTOR();  // this instantiates the predictor code
  ///////////////////////////////////////////////
  // read each trace recrod, simulate until done
  ///////////////////////////////////////////////

    std::string trace_path;
    trace_path = argv[1];
    bt9::BT9Reader bt9_reader(trace_path);

    std::string key = "total_instruction_count:";
    std::string value;
    bt9_reader.header.getFieldValueStr(key, value);
    UINT64     total_instruction_counter = std::stoull(value, nullptr, 0);
    key = "branch_instruction_count:";
    bt9_reader.header.getFieldValueStr(key, value);
    UINT64     branch_instruction_counter = std::stoull(value, nullptr, 0);
    UINT64     numMispred =0;  
    UINT64     numMispred_btbMISS =0;  
    UINT64     numMispred_btbANSF =0;  
    UINT64     numMispred_btbATSF =0;  
    UINT64     numMispred_btbDYN =0;  

    UINT64 cond_branch_instruction_counter=0;
    UINT64 btb_ansf_cond_branch_instruction_counter=0;
    UINT64 btb_atsf_cond_branch_instruction_counter=0;
    UINT64 btb_dyn_cond_branch_instruction_counter=0;
    UINT64 btb_miss_cond_branch_instruction_counter=0;
    UINT64 uncond_branch_instruction_counter=0;


    fprintf(stderr,"Tracefile: %s\n Total Instructions: %d \nBranch Instructions : %d\n",trace_path.c_str(),total_instruction_counter,branch_instruction_counter);
  ///////////////////////////////////////////////
  // model simple branch marking structure
  ///////////////////////////////////////////////
    std::map<UINT64, UINT32> myBtb; 
    map<UINT64, UINT32>::iterator myBtbIterator;

    myBtb.clear();
   
  ///////////////////////////////////////////////
  // read each trace record, simulate until done
  ///////////////////////////////////////////////

      OpType opType;
      UINT64 PC;
      bool branchTaken;
      UINT64 branchTarget;
      UINT64 numIter = 0;
      
      // Top Failing Branches
      
      for (auto it = bt9_reader.begin(); it != bt9_reader.end(); ++it) {
        CheckHeartBeat(++numIter);

        try {
          bt9::BrClass br_class = it->getSrcNode()->brClass();

          bool dirDynamic = (it->getSrcNode()->brObservedTakenCnt() > 0) && (it->getSrcNode()->brObservedNotTakenCnt() > 0); //JD2_2_2016
//          bool dirNeverTkn = (it->getSrcNode()->brObservedTakenCnt() == 0) && (it->getSrcNode()->brObservedNotTakenCnt() > 0); //JD2_2_2016

//JD2_2_2016 break down branch instructions into all possible types
          opType = OPTYPE_ERROR; 

          if ((br_class.type == bt9::BrClass::Type::UNKNOWN) && (it->getSrcNode()->brNodeIndex())) { //only fault if it isn't the first node in the graph (fake branch)
            opType = OPTYPE_ERROR; //sanity check
          }
//NOTE unconditional could be part of an IT block that is resolved not-taken
//          else if (dirNeverTkn && (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)) {
//            opType = OPTYPE_ERROR; //sanity check
//          }
//JD_2_22 There is a bug in the instruction decoder used to generate the traces
//          else if (dirDynamic && (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)) {
//            opType = OPTYPE_ERROR; //sanity check
//          }
          else if (br_class.type == bt9::BrClass::Type::RET) {
            if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
              opType = OPTYPE_RET_COND;
            else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
              opType = OPTYPE_RET_UNCOND;
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else if (br_class.directness == bt9::BrClass::Directness::INDIRECT) {
            if (br_class.type == bt9::BrClass::Type::CALL) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
                opType = OPTYPE_CALL_INDIRECT_COND;
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
                opType = OPTYPE_CALL_INDIRECT_UNCOND;
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else if (br_class.type == bt9::BrClass::Type::JMP) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL)
                opType = OPTYPE_JMP_INDIRECT_COND;
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL)
                opType = OPTYPE_JMP_INDIRECT_UNCOND;
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else if (br_class.directness == bt9::BrClass::Directness::DIRECT) {
            if (br_class.type == bt9::BrClass::Type::CALL) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
                opType = OPTYPE_CALL_DIRECT_COND;
              }
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
                opType = OPTYPE_CALL_DIRECT_UNCOND;
              }
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else if (br_class.type == bt9::BrClass::Type::JMP) {
              if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) {
                opType = OPTYPE_JMP_DIRECT_COND;
              }
              else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) {
                opType = OPTYPE_JMP_DIRECT_UNCOND;
              }
              else {
                opType = OPTYPE_ERROR;
              }
            }
            else {
              opType = OPTYPE_ERROR;
            }
          }
          else {
            opType = OPTYPE_ERROR;
          }

  
          PC = it->getSrcNode()->brVirtualAddr();

          branchTaken = it->getEdge()->isTakenPath();
          branchTarget = it->getEdge()->brVirtualTarget();

          //printf("PC: %llx type: %x T %d N %d outcome: %d", PC, (UINT32)opType, it->getSrcNode()->brObservedTakenCnt(), it->getSrcNode()->brObservedNotTakenCnt(), branchTaken);

/************************************************************************************************************/

          if (opType == OPTYPE_ERROR) { 
            if (it->getSrcNode()->brNodeIndex()) { //only fault if it isn't the first node in the graph (fake branch)
              fprintf(stderr, "OPTYPE_ERROR\n");
              printf("OPTYPE_ERROR\n");
              exit(-1); //this should never happen, if it does please email CBP org chair.
            }
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::CONDITIONAL) { //JD2_17_2016 call UpdatePredictor() for all branches that decode as conditional
            //printf("COND ");

            myBtbIterator = myBtb.find(PC); //check BTB for a hit
            bool btbATSF = false;
            bool btbANSF = false;
            bool btbDYN = false;

            bool predDir = false;

            if (myBtbIterator == myBtb.end()) { //miss -> we have no history for the branch in the marking structure
              //printf("BTB miss ");
              // insert with curent value of PC along with the fact that branch was taken
              myBtb.insert(pair<UINT64, UINT32>(PC, (UINT32)branchTaken)); //on a miss insert with outcome (N->btbANSF, T->btbATSF)
              predDir = brpred->GetPrediction(PC, btbANSF, btbATSF, btbDYN); // all values are false
              fprintf(stderr,"GetPrediction pc: 0x%2x: Pred:%d Hit:0 btbANSF:%d btbATSF:%d\n",PC,branchTaken == predDir ,btbANSF,btbATSF);
              brpred->UpdatePredictor(PC, opType, branchTaken, predDir, branchTarget, btbANSF, btbATSF, btbDYN); // all booleans are false
            }
            else { // hit in the btb

              btbANSF = (myBtbIterator->second == 0);
              btbATSF = (myBtbIterator->second == 1);
              btbDYN = (myBtbIterator->second == 2);
              //printf("BTB hit ANSF: %d ATSF: %d DYN: %d ", btbANSF, btbATSF, btbDYN);

              predDir = brpred->GetPrediction(PC, btbANSF, btbATSF, btbDYN);              
              fprintf(stderr,"GetPrediction pc: 0x%2x: Pred:%d Hit:1 btbANSF:%d btbATSF:%d\n",PC,branchTaken == predDir ,btbANSF,btbATSF);
              
              brpred->UpdatePredictor(PC, opType, branchTaken, predDir, branchTarget, btbANSF, btbATSF, btbDYN); 

              if (  (btbANSF && branchTaken)   // only exhibited N until now and we just got a T -> upgrade to dynamic conditional
                 || (btbATSF && !branchTaken)  // only exhibited T until now and we just got a N -> upgrade to dynamic conditional
                 ) {
                myBtbIterator->second = 2; //2-> dynamic conditional (has exhibited both taken and not-taken in the past)
              }
            }
            //puts("");

            if(predDir != branchTaken){
              numMispred++; // update mispred stats
              if(btbATSF)
                numMispred_btbATSF++; // update mispred stats
              else if(btbANSF)
                numMispred_btbANSF++; // update mispred stats
              else if(btbDYN)
                numMispred_btbDYN++; // update mispred stats
              else
                numMispred_btbMISS++; // update mispred stats
            }
            cond_branch_instruction_counter++;

            if (btbDYN)
              btb_dyn_cond_branch_instruction_counter++; //number of branches that have been N at least once after being T at least once
            else if (btbATSF)
              btb_atsf_cond_branch_instruction_counter++; //number of branches that have been T at least once, but have not yet seen a N after the first T
            else if (btbANSF)
              btb_ansf_cond_branch_instruction_counter++; //number of cond branches that have not yet been observed T
            else
              btb_miss_cond_branch_instruction_counter++; //number of cond branches that have not yet been observed T
          }
          else if (br_class.conditionality == bt9::BrClass::Conditionality::UNCONDITIONAL) { // for predictors that want to track unconditional branches
            uncond_branch_instruction_counter++;
            brpred->TrackOtherInst(PC, opType, branchTaken, branchTarget);
          }
          else {
            fprintf(stderr, "CONDITIONALITY ERROR\n");
            printf("CONDITIONALITY ERROR\n");
            exit(-1); //this should never happen, if it does please email CBP org chair.
          }

/************************************************************************************************************/
        }
        catch (const std::out_of_range & ex) {
          std::cout << ex.what() << '\n';
          break;
        }
      
      } //for (auto it = bt9_reader.begin(); it != bt9_reader.end(); ++it)


    ///////////////////////////////////////////
    //print_stats
    ///////////////////////////////////////////

    //NOTE: competitors are judged solely on MISPRED_PER_1K_INST. The additional stats are just for tuning your predictors.

      printf("  TRACE \t : %s" , trace_path.c_str()); 
      printf("  NUM_INSTRUCTIONS            \t : %10llu",   total_instruction_counter);
      printf("  NUM_BR                      \t : %10llu",   branch_instruction_counter-1); //JD2_2_2016 NOTE there is a dummy branch at the beginning of the trace...
      printf("  NUM_UNCOND_BR               \t : %10llu",   uncond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR          \t : %10llu",   cond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR_BTB_MISS \t : %10llu",   btb_miss_cond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR_BTB_ANSF \t : %10llu",   btb_ansf_cond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR_BTB_ATSF \t : %10llu",   btb_atsf_cond_branch_instruction_counter);
      printf("  NUM_CONDITIONAL_BR_BTB_DYN  \t : %10llu",   btb_dyn_cond_branch_instruction_counter);
      printf("  NUM_MISPREDICTIONS          \t : %10llu",   numMispred);
      printf("  NUM_MISPREDICTIONS_BTB_MISS \t : %10llu",   numMispred_btbMISS);
      printf("  NUM_MISPREDICTIONS_BTB_ANSF \t : %10llu",   numMispred_btbANSF);
      printf("  NUM_MISPREDICTIONS_BTB_ATSF \t : %10llu",   numMispred_btbATSF);
      printf("  NUM_MISPREDICTIONS_BTB_DYN  \t : %10llu",   numMispred_btbDYN);
      printf("  MISPRED_PER_1K_INST         \t : %10.4f",   1000.0*(double)(numMispred)/(double)(total_instruction_counter));
      printf("  MISPRED_PER_1K_INST_BTB_MISS\t : %10.4f",   1000.0*(double)(numMispred_btbMISS)/(double)(total_instruction_counter));
      printf("  MISPRED_PER_1K_INST_BTB_ANSF\t : %10.4f",   1000.0*(double)(numMispred_btbANSF)/(double)(total_instruction_counter));
      printf("  MISPRED_PER_1K_INST_BTB_ATSF\t : %10.4f",   1000.0*(double)(numMispred_btbATSF)/(double)(total_instruction_counter));
      printf("  MISPRED_PER_1K_INST_BTB_DYN \t : %10.4f",   1000.0*(double)(numMispred_btbDYN)/(double)(total_instruction_counter));
      printf("\n");
}



