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
//#define PRINTSIZE

#include "utils.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
////////////////////////////////////////////////////////////////////////////////////////////////////
// BEGIN PERCEPTRON
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace PERCEPTRON {
  

struct binfo {
        unsigned int address, opcode, br_flags;
};

#define BR_CONDITIONAL 1

// this class contains information that is kept between prediction and update
// this is mostly cruft left over from my own simulation infrastructure

class branch_info {
	bool _prediction;
public:

	branch_info (void) { }

	void prediction (bool p) {
		_prediction = p;
	}

	bool prediction (void) {
		return _prediction;
	}
};

class branch_predictor {
public:
	branch_predictor (void) { }
	virtual branch_info *lookup (unsigned int pc, bool evertaken, bool evernottaken) = 0;
	virtual void update (branch_info *p, bool taken) = 0;
	virtual void info (unsigned int pc, unsigned int optype) { }
};

// history sampling hashed perceptron predictor

typedef char byte;

// this class holds information learned or computed in the prediction phase
// that we would like to remember for the update phase. it does not count
// against the hardware budget because it can be recomputed; it is a programming 
// convenience rather than a necessity.

#define MAXIMUM_HISTORY_LENGTH  896

class sampler_info : public branch_info {
public:

	// the perceptron output for this prediction
  /**
   * As we go through the system we are going to accumulate the
   * perceptron result into the weighted and unweighted sum
   */
  float	sum, weighted_sum; 

  // the pc of the branch being predicted
  unsigned int pc;
  
  /**
   * A list of table indices.      
   */
  // table indices
  unsigned int *indices, local_index;

  sampler_info (void) { }
};

class sampler : public branch_predictor {
private:

	// information about this prediction to keep until the update

	sampler_info p;

	// variables for managmenent of custom memory allocator

	unsigned char 
		*my_heap, 
		*bump_pointer, 
		*heap_end;

	// RUNTIME CONSTANTS that are parameters to the constructor

	const int 
		local_shift,			// how far left to shift local history before XORing
		bytes,				// total number of bytes allowed to predictor 
		max_exp, 			// maximum value in coeffs array 
		bits, 				// bit width of weights 
		num_tables,			// number of tables 
		folding,			// number of bits in output of history folding function 
		filter_size,			// size of never/always taken branch filter 
		path_bit,			// which bit of PC should be recorded in path history 
		speed, 				// speed for adaptive threshold fitting 
		local_table_size,		// size of the table for the local history predictor 
		local_history_length,		// history length for the local predictor 
		local_num_histories;		// number of local histories to keep 

	// RUNTIME CONSTANTS that are computed in the constructor based on parameters

	int
		table_size,			// size of each perceptron table (1097 for 32KB version, 283 for 4KB version)
		max_weight, 			// maximum weight value 
		min_weight, 			// minimum weight value 
		history_length, 		// maximum history length in the samples 
		history_size;			// number of 64-bit history elements 

	const unsigned int 
		mask;				// tells us which non-conditional-branch PCs should be incorporated into the global and/or path history

	const int 
		*samples;			// the samples (see paper)

	const double
		theta_fuzz,			// describes a neighborhood around theta where a correct prediction may probabilistically lead to an update
		coeff_local, 			// multiply the local weight times this before adding it to the sum 
		coeff_train, 			// multiply the perceptron output by this before deciding whether to train 
		coeff_base;			// we add to the weighted sum coeff_base to a coefficient power times the weight 

	const bool
		static_prediction,		// what to predict when we don't know what to predict
		primey;				// if true, decrease table sizes to the nearest prime number for better hashing :-)


	// PREDICTOR STATE that counts against the hardware budget

	unsigned int
		my_seed;			// seed for a random number generator

	int
		signature,
		psel,				// choose between sum and weighted sum for prediction
		tc, 				// threshold counter for adaptive threshold fitting 
		theta; 				// threshold for perceptron learning

	unsigned long long int
		*global_history,		// array of history_size 64-bit unsigned integers keeping the global pattern history
		*path_history;			// array of history_size 64-bit unsigned integers keeping the global path history

	unsigned long long int
		callstack_history;		// a single 64-bit unsigned integer keeping a 1-bit wide 64-deep callstack (actually no more than 5 bits are ever used)

	bool *filter[2];			// array of two filter_size vectors of bits used to remember whether a branch has non-trivial behavior

	byte **table;				// perceptron weights tables: a num_tables x table_size array of small signed integers

	byte *local_pht;			// local pattern history table: an array of local_table_size small signed integers

	unsigned int
		*local_histories;		// local histories: an array of local_num_histories bit vectors (i.e. unsigned integers) of local_history_length bit each

	int *coeffs;				// the logs base coeff_base of numbers multiplied by weights to get the weighted sum
	unsigned long long int *histories[3];


	// METHODS

	// index an array of unsigned long long ints holding path/pattern/callstack history

        
        /**
         * We are given a - bit inside index_history Take an array of
         * history integers and then index into a particular bit of
         * the history
         * 1. First figure out which word it lies in 
         * 2. Second figure out which bit it is supposed to index into
         */
	bool index_history (unsigned long long int *v, unsigned int index) {
          
          // First determine the index into the actual word
          unsigned int word_index = index / (sizeof (unsigned long long int) * 8);
          
          // If word index greater than number of elements in history return 0
          if (word_index >= (unsigned int) history_size) return false; // too far
          
          // Select the bit inside the word offset.
          unsigned int word_offset = index % (sizeof (unsigned long long int) * 8);
          return (v[word_index] >> word_offset) & 1;
          
	}

	// fold a history of 'on' bits into 'cn' bits, skipping 'st'
	// bits. inspired by Seznec's history folding function

        /**
         * Fold from on -> cn bits while skipping st bits
         * ok this is totally mysterious
         *  a <- 
         *  b <-
         * on <- end
         * cn <- folding
         * st <- stride
         */
	unsigned int fold_history(unsigned long long int *v, int start, int on, int cn, int st) {
          
          unsigned int r = 0;

          // first part: go up to the highest multiple of 'cn' less than 'on'

          /**
           *
           * on/cn * cn -> block number with cn
           *
           * 1. 
           *
           */
          int lim = (on / cn) * cn;
                
          int i;
          unsigned int q;
          
          /**
           * Start from 0 and keep iterating through groups of cn. 
           * 
           * 'on' will have at most lim # of cn groups
           *
           */
          // keep iterating in groups of cn
          for (i = 0; i < lim; i += cn) {
            
            q = 0;
            
            /**
             * Start at the beginning of the group.  Traverse the
             *  group selected in strides of length st till end of the
             *  group
             */
            for (int j = i; j < i + cn; j += st) {
              
              /**
               * Move shift q by 1
               */
              q <<= 1;
              /**
               * pick the j'th element from history register and append it to q.
               */
              q |= index_history (v, start+j);                                
            }
            
            // q encapsulates our stride wise selection of elements from this group
            // r xor q 
            // 000 ^ 101 = > 101 , 111 ^ 101 = 010
            
                        
            r ^= q;
            
          }          

          // second part: peel off last iteration and check we don't go over 'on'

          q = 0;
          for (int j=i; j<on; j++) {
            
            q <<= 1;
            q |= index_history (v, start+j);
            
          }
          
          // why the xor oh my why the xor ????
          r ^= q;

          // r with all stride length bits set to 1???
          
          return r & ((1<<cn)-1);
          
	}
        

	// hash functions by Thomas Wang, best live link is http://burtleburtle.net/bob/hash/integer.html

	unsigned int hash1 (unsigned int a) {
		a = (a ^ 0xdeadbeef) + (a<<4);
		a = a ^ (a>>10);
		a = a + (a<<7);
		a = a ^ (a>>13);
		return a;
	}

        unsigned int hash2 (unsigned int key) {
                int c2=0x27d4eb2d; // a prime or an odd constant
                key = (key ^ 61) ^ (key >> 16);
                key = key + (key << 3);
                key = key ^ (key >> 4);
                key = key * c2;
                key = key ^ (key >> 15);
                return key;
        }

	// hash a key with the i'th hash function using the common Bloom filter trick
	// of linearly combining two hash functions with i as the slope

	unsigned int hash (unsigned int key, unsigned int i) {
		return hash2 (key) * i + hash1 (key);
	}

	// custom memory allocator

	unsigned char *my_malloc (int size) {
		unsigned char *r = bump_pointer;
		while (size & 3) size++; // 8-byte align
		bump_pointer += size;
		if (bump_pointer >= heap_end) assert (0);
		return r;
	}

	// returns true if x is prime

	bool prime (unsigned int x) {
		for (int i=2; i<=(int)x/2; i++) if ((x % i) == 0) return false;
		return true;
	}

public:

	// constructor

	sampler (
		// parameters 
		int _bytes = 32768+(1024/8), 		// hardware budget in bytes
                
                /**
                 * We limit the number of bits for each perceptron input weight
                 */
		int _bits = 6, 				// bit width of a weight
                
                /**
                 * Q:Why do we keep 32 perceptron weight tables?
                 * 
                 */
		int _num_tables = 32, 			// number of perceptron weights tables
                
                /**
                 * How do we know that his is the width of hash function domain
                 */
		int _folding = 24, 			// width of hash function domain
                
                /**
                 * Q:Why do we use a filter size , is it simply to
                 * inibit trivial branches from poluting the
                 * perceptrons ?
                 */
		int _filter_size = 16384, 		// number of entries in filter
                
		int _theta = 24, 			// initial value of threshold
                
		int _path_bit = 6, 			// which bit of the PC to use for global path history

                
                /**
                 * This is based on work by Seznec on dynamic threshold fitting.
                 *
                 * 1. Optimial threshold varies for applications.
                 * 2. We always check that speed is 7
                 *
                 */
		int _speed = 7, 			// speed for dynamic threshold fitting
                
                
                /**
                 * 
                 */
		double _coeff_local = 2.25,	 	// coefficient for local predictor weight

                /**
                 * Keeps a per hash(pc) history of previous taken/not taken branches.
                 */
		int _local_table_size = 2048, 		// size of local predictor pattern history table
                
                /**
                 * Limit for the length of per pc local history which
                 * we decide to keep.
                 */
		int _local_history_length = 7, 		// history length for local predictor
                
                /**
                 * Number of distinct local history(per pc) entreis we
                 * are allowed to keep We will be subject to aliazing
                 * problems with low numbers in this field.
                 *
                 */
		int _local_num_histories = 256,		// number of local histories to keep
                
		double _theta_fuzz = 8.72, 		// neighborhood around theta to probabilistically train on correct prediction
                
		double _coeff_train = 1.004, 		// fudge factor for sum to decide whether training is needed
                
		double _coeff_base = 1.0000258,		// coefficients for weights learned at run-time are this raised to coeffs[i] power
                
		int _max_exp = 131072,	 		// maximum value for coeffs[i]
                
		bool _primey = true,			// if true, decrease table sizes to nearest prime number
                
		unsigned int _mask = 0x40000018,	// used to determine which bits of non-conditional branch PCs to shift into which histories
                
		bool _static_prediction = false,
                
		int _local_shift = 0 ) :

		// initialize 

		branch_predictor () ,
		local_shift(_local_shift),
		bytes(_bytes),
		max_exp(_max_exp),
		bits(_bits),
		num_tables(_num_tables),
		folding(_folding),
		filter_size(_filter_size),
		path_bit(_path_bit),
		speed(_speed),
		local_table_size(_local_table_size),
		local_history_length(_local_history_length),
		local_num_histories(_local_num_histories),
		mask(_mask),
		theta_fuzz(_theta_fuzz),
		coeff_local(_coeff_local),
		coeff_train(_coeff_train),
		coeff_base(_coeff_base),
		static_prediction(_static_prediction),
		primey(_primey) 
		{

		// set up custom memory allocator

#define PERCEPTRON_N 1000000
                  // #define VERBOSE 1
		my_heap = new unsigned char[PERCEPTRON_N];
		memset (my_heap, 0, PERCEPTRON_N);
		heap_end = &my_heap[PERCEPTRON_N];
		bump_pointer = my_heap;
		signature = 0;

		// we have this many total bits to work with

		int total_bits = bytes * 8;
                //#ifdef VERBOSE
		printf ("start with %d bits\n", total_bits);
                //#endif

		// global and local indices kept in the sampler_info struct

		total_bits -= 11 * num_tables + 11;

		// sum and weighted sum are two 32-bit floats

		total_bits -= 32 * 2;

		// initialize random number generator and account for its 32 bits of state

		my_seed = 0xd00ba1;
		total_bits -= sizeof (unsigned int) * 8;

		// initialize call stack history and account for its 64 bits of state

		callstack_history = 0;
		// total_bits -= sizeof (unsigned long long int) * 8;

		// we only use 64 bits of callstack history

		total_bits -= 64;

		// initialize weighted versus non-weighted selector and account for its 10 bit width
                
                /**
                 * A selector between weighted and non-weigthed selector
                 */
		psel = 0;
		total_bits -= 10;

		// coefficients are 18 bits each

		total_bits -= 18 * num_tables;
                
		coeffs = (int *) my_malloc (sizeof (int) * num_tables);
                
		for (int i = 0; i < num_tables; i++)
                  coeffs[i] = 0;

		// set up and account for local predictor

                // This will always be taken since in 32 kb we always keep local histories.
		if (local_table_size) {
			local_pht = (byte *) my_malloc (local_table_size);
			local_histories = (unsigned int *) my_malloc (local_num_histories * sizeof (unsigned int));
			total_bits -= local_table_size * bits;
			total_bits -= local_num_histories * local_history_length;
#ifdef VERBOSE
			printf ("%d bits after accounting for local predictor\n", total_bits);
#endif
		} else {
			local_pht = NULL;
			local_histories = NULL;
		}

		// figure out the history length as the maximum indexed history position in any sample

		history_length = MAXIMUM_HISTORY_LENGTH;

#ifdef VERBOSE
		// this will print out 752 for the 32KB version, 120 for the 4KB version
		printf ("history length is %d\n", history_length);
#endif

		// account for the history lengths

		total_bits -= history_length * 2;

		// increase the history length by a safe amount for sizing history arrays
		// so we will allocate the right number of unsigned long long ints

		while (history_length & 7) history_length++;
		history_length += folding;
		history_size = (history_length / (8 * sizeof (unsigned long long int))) + 1;
#ifdef VERBOSE
		printf ("history size is %d\n", history_size);
#endif

		// get an array to remember the indices from this iteration to the next
		// does not count against h/w budget because we can recompute it (anyway, 
		// if you want to count it it's at most 10 x 32 = 320 bits
                /**
                 * Indices is an array of integers one per table of perceptrons
                 */
		p.indices = (unsigned int *) my_malloc (num_tables * sizeof (unsigned int));

		// compute maximum and minimum weights
                
                /**
                 * The granularity of the weights determine how much
                 * weight can be assigned to each perceptron edge.
                 */
		max_weight = (1<<bits)/2-1;
		min_weight = -(1<<bits)/2;

		// initialize and accont for adaptive threshold fitting counter (it is never more than 8 bits)
                
                /**
                 * Some sort of threshold counter.
                 */
		tc = 0;
                
		total_bits -= 8;

		// initialize and account for theta (it is never more than 9 bits)

		total_bits -= 9;
		theta = _theta;

		// global history
                
                /**
                 * Global history is maintained in a set of `unsigned
                 * long long int` of history size.
                 */
		global_history = (unsigned long long int *) my_malloc (sizeof (unsigned long long int) * (history_size+1));

		// path history
                /**
                 * Path history is mainted as a set of `unsigned long
                 * long int` . The total history is a concatenation of these values.
                 */
		path_history = (unsigned long long int *) my_malloc (sizeof (unsigned long long int) * (history_size+1));

		// filter

		if (filter_size) {
                  for (int i=0; i<2; i++) {
                    total_bits -= filter_size;
                    filter[i] = (bool *) my_malloc (filter_size);
                  }
#ifdef VERBOSE
                  printf ("%d bits after accounting for filter\n", total_bits);
#endif
		}

		table_size = (total_bits / bits) / num_tables;

		if (primey) while (!prime(table_size)) table_size--;
#ifdef VERBOSE
		// this will print 1097 for the 32KB version, 283 for the 4KB version
		printf ("table size is %d\n", table_size);
#endif

		// perceptron weights tables

		table = (byte **) my_malloc (num_tables * sizeof (byte *));
		for (int i=0; i<num_tables; i++) {
			table[i] = (byte *) my_malloc (table_size);
			memset (table[i], 0, table_size);
			total_bits -= bits * table_size;
		}
#ifdef VERBOSE
		printf ("ended up with %d bits left over.\n", total_bits);
#endif
		assert (total_bits >= 0);
	}

	// destructor

	~sampler (void) {
		delete[] my_heap;
	}

	void reset_tables (void) {
		for (int i=0; i<num_tables; i++) {
			memset (table[i], 0, table_size);
		}
	}
        

	// shift a history bit into the history array
        
        /**
         * Shift history allows us to treat the history array as a
         * single list of histories .
         *
         * 1. Make room for the element which is going to
         *    be dropped off from the shift
         * 2. Add the new history value.
         * 3. The drop off bit becomes the 
         *    new bit to be appended to the array.
         */
	void shift_history(unsigned long long int *v, bool t) {
          for (int i=0; i<= history_size; i++) {
            
            bool nextt = !!(v[i] & (1ull<<63));
            
            // Shift history by 1
            v[i] <<= 1;
            
            // Append the value of t
            v[i] |= t;
            
            // The drop off bit will become the value bit for the next
            // one
            t = nextt;
          }
	}


        /**
         * Saturating increment/decrement 
         */
	byte satincdec (byte weight, bool taken, int max, int min) {
          if (taken) {
            return (weight < max) ? weight + 1 : weight;
          } else {
            return (weight > min) ? weight - 1 : weight;
          }
	}

	// dynamic threshold updating per Seznec
        
        /**
         * 
         */
	void threshold_setting (sampler_info *u, bool correct, int a) {
          
          if (!correct) { 
            tc++; // increase tc on every incorrect guess
            if (tc >= speed) {
              theta++;
              tc = 0;
            }
          }
          
          if (correct && a < theta) {
            
            /**
             * Reduce speed on every correct guess where maginitude is
             * less than theta.
             */
            tc--;
            if (tc <= -speed) {
              theta--;
              // 
              tc = 0;
            }
          }
	}

	// return a local history for this pc
        
        /**
         * Takes the pc. Runs it throught on of the hashes. and cuts
         * off some bits to map into the local_history table.  After
         * it fetches the entry in local history table, it applies
         * some local hist right now we dont know what local shift
         * stands for ??
         * The returned history seems to be and with  with the pc ??
         */
	unsigned int get_local_history (unsigned int pc) {
		unsigned int l = local_histories[hash(pc,0) % local_num_histories];
		l &= (1<<local_history_length)-1;
		l <<= local_shift;
		return (l ^ pc) % local_table_size;
	}

        /**
         * 
         */
	void compute_partial_index (unsigned int *index, unsigned int pc, int start, int end, int kind, int which, int stride) {
          // Pick a certain hash
          unsigned int partial_index = hash (pc, which);
          
          // Stride over the global history
          // Folding - width of hash function domain
          
          /**
           * 
           */
          partial_index ^= fold_history(histories[kind], start, end-start, folding, stride);

          *index = *index * 2 + partial_index;

          fprintf(stderr,"\ncompute_partial_index pc:%10x index:%10u  history:%10d hash-which:%d  [%4d-%4d,%4d]\n",pc,*index,kind,which,start,end,stride);
          
	}


        /**
         * Reads a perceptron table. value in perceptron table to weight
         * sets perceptron u->indices[t] = to index in the table
         */
	void finish_index (unsigned int *index, int this_table, sampler_info *u) {
          fprintf(stderr,"\nfinish_index[Entry]: Table:%-5d Index %-5u  Sum:%5d\n",this_table,*index,u->sum);
          /* modulo the computed index by the table size */
          *index %= table_size;
          
          /* Read one of the perceptron tables */
          int x = table[this_table][*index];

          /*    */
          u->sum += x;
          if(this_table == 0)
            fprintf(stderr,"\n");
          
          fprintf(stderr,"finish_index: Table:%-5d Index %-5u Weight:%5d Sum:%5d\n",this_table,*index,x,u->sum);
          
          u->weighted_sum += pow (coeff_base, coeffs[this_table]) * x;
          u->indices[this_table] = *index;
          
          *index = 0;
	}


//void compute_partial_index (unsigned int *index, unsigned int pc, int start, int end, int kind(history type), int which(hash type), int stride(stride length) )
/**
   start -> a
   end -> b
   kind -> c
   which -> t
   stride -> d
   
 */
#define C(a, b, c, d) compute_partial_index (&j, pc, a, b, c, t, d);

        
/**
 * Compute the partial index with with start - > 0  end -> 0 , kind -> t, stride -> 0 
 int start -> 0,
 int end   -> 0, 
 int kind  -> 0,
 int which -> t,
 int stride -> 0
 */
#define B() compute_partial_index (&j, pc, 0, 0, 0, t, 0);

/**
 * Move on to next table 
 */
/**
   
 */
#define F() finish_index (&j, t++, u);


	double compute_indices_0 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
                /**
                 * Understanding single generated stride
                 */
                // start with j=0 pc=pc , start=0,end=0,kind=0 /* global history */ ,stride =0
		B(); 
                // t = t // which is 0 for global history
                C(0,3,0,1); 
                F(); // t in incremented 

		C(1,4,1,5);
                C(14,27,0,1);
                F(); // incremnt t=1
                
		C(0,7,1,4);
                F(); //increment t=2
                
		B();C(2,9,0,2);C(5,33,1,3);
                F(); // t = 3
                
		B();C(0,14,1,5);
                F(); // t = 4
                
		C(0,18,0,3);
                F(); // t = 5
                
		C(0,120,0,3);C(7,13,0,6);
                F(); // t = 6
                
		C(0,1,1,3);C(0,2,0,4);C(0,17,1,5);C(11,30,1,1);
                F(); // t = 7

		C(7,30,1,4);C(0,11,1,2);C(0,4,0,8);C(8,12,1,4);
                F(); // t = 8
                
		C(0,42,0,7);
                F(); // t = 9 
                
		C(1,6,0,4);C(0,16,0,8);C(2,55,0,1);C(8,39,0,4);
                F(); // t = 10
                
		C(49,57,1,5);
                F(); // t = 11
                
		C(7,12,1,5);
                F(); // t=
                
		C(11,30,1,1);C(7,11,0,1);C(0,50,0,4);C(13,58,0,4);C(13,20,1,1);C(0,3,1,6);
                F(); // t=
                
		C(0,4,0,2);
                F(); // t=
                
		B();B();C(0,116,0,3);C(0,63,0,4);
                F(); // t=
                
		C(2,7,0,5);C(66,139,0,2);C(9,16,0,3);
                F(); // t=
                
		C(99,142,0,2);C(0,9,0,3);
                F(); // t=
                
		C(27,71,1,8);C(1,24,1,8);C(46,147,1,1);C(38,57,0,7);
                F(); // t=
                
		C(0,65,0,4);C(0,40,0,4);C(14,21,1,6);C(0,52,0,4);C(25,46,1,2);C(3,214,1,2);C(56,76,1,5);
                F(); // t=
                
		C(9,221,1,6);
                F(); // t=
                
		C(0,31,1,2);C(0,253,1,8);C(0,54,0,5);
                F(); // t=
                
		C(0,255,1,8);B();C(1,12,1,4);
                F(); // t=
                
		B();C(0,45,0,8);C(17,26,1,3);C(0,2,0,1);C(0,4,1,1);B();B();C(0,27,0,7);C(0,7,0,2);B();C(0,4,1,7);C(14,258,0,1);C(10,28,1,7);
                F(); // t=
                
		C(6,14,0,4);C(0,260,0,1);
                F(); // t=
                
		C(31,37,0,2);C(31,43,1,3);C(10,264,1,2);
                F(); // t=
                
		C(0,7,0,1);C(45,351,1,2);C(6,27,1,4);C(0,42,0,8);C(6,14,0,3);
                F(); // t=
                
		C(0,2,0,1);C(65,403,1,2);C(0,16,0,1);C(0,16,1,8);C(0,65,1,2);
                F(); // t=
                
		C(2,20,0,1);C(5,9,1,3);C(0,8,0,4);C(76,417,1,5);C(14,258,0,1);C(0,26,1,6);C(38,57,1,3);
                F(); // t=
                
		C(82,434,1,5);C(19,54,1,7);C(4,253,1,8);
                F(); // t=
                
		C(19,54,1,7);
                F(); // t=
                
		C(0,153,1,6);C(8,718,1,4);C(1,11,1,7);C(6,712,1,5);
                F(); // t=
                
                
		return 2.399032;
	}

	double compute_indices_1 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,8,2,3);F();
		C(11,19,1,5);F();
		C(0,2,0,7);F();
		C(0,1,1,4);C(0,4,1,4);C(0,2,0,4);B();C(0,2,1,1);F();
		C(0,5,1,2);C(0,5,1,2);C(0,4,0,5);C(0,2,1,5);F();
		C(0,7,2,3);F();
		C(0,3,1,8);C(0,8,1,6);C(1,4,0,8);C(0,7,0,1);C(4,6,1,6);F();
		C(0,4,2,4);F();
		B();C(2,9,0,2);B();F();
		C(5,9,1,4);C(11,16,0,1);C(2,22,1,5);F();
		C(2,8,0,4);C(0,9,1,1);C(1,11,0,5);C(42,47,2,7);F();
		B();B();C(1,12,1,3);F();
		C(9,16,0,3);F();
		C(15,16,1,4);C(19,26,0,7);F();
		C(42,57,1,2);F();
		B();C(2,9,0,4);C(22,27,1,2);B();C(0,7,1,4);C(0,18,1,8);B();F();
		C(2,7,1,1);F();
		C(0,7,0,3);C(0,5,1,2);C(0,5,0,3);F();
		C(8,23,0,2);F();
		C(53,61,0,2);F();
		C(7,19,0,4);C(28,30,1,8);C(7,9,0,3);C(0,1,1,2);F();
		C(0,17,0,1);F();
		C(7,49,1,7);C(2,5,1,3);F();
		C(31,38,2,6);F();
		C(0,8,1,5);F();
		C(14,21,1,7);F();
		C(25,46,1,2);F();
		C(0,7,1,3);C(0,3,0,3);C(2,12,0,7);F();
		C(0,16,0,6);F();
		C(2,5,0,8);F();
		C(0,12,1,6);C(0,9,1,7);F();
		C(4,38,1,1);C(0,10,1,4);B();C(2,5,1,5);B();C(6,7,1,4);B();C(2,17,1,5);F();
		return 1.781267;
	}

	double compute_indices_2 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,3,1,1);C(0,3,0,5);F();
		B();C(16,24,1,6);F();
		C(3,6,0,3);C(7,21,0,1);F();
		B();C(0,10,1,1);C(0,4,1,4);C(0,9,1,4);C(1,6,0,6);C(0,4,1,8);C(0,4,1,7);C(10,22,0,8);F();
		C(0,5,1,2);C(0,6,1,7);C(0,6,0,1);F();
		C(0,8,0,2);F();
		C(31,38,1,8);C(0,1,0,5);F();
		C(0,22,0,5);C(5,23,0,4);B();F();
		C(0,1,0,1);C(0,9,1,1);F();
		C(22,33,0,4);F();
		C(0,12,1,1);C(0,4,1,7);C(1,11,0,7);F();
		C(36,45,0,3);C(0,3,0,1);F();
		C(40,49,1,6);C(1,2,1,7);C(0,2,0,1);F();
		C(5,13,1,3);C(8,12,0,1);C(0,11,0,1);B();F();
		C(30,42,0,2);F();
		C(3,10,1,5);C(29,53,1,3);C(0,1,0,7);F();
		C(26,32,0,3);F();
		C(0,7,1,5);C(17,26,1,3);F();
		B();C(31,43,1,3);C(5,13,1,4);F();
		C(2,5,0,7);B();C(23,31,1,1);F();
		C(3,4,1,3);C(6,14,1,3);C(24,31,1,6);F();
		C(16,25,0,2);B();C(56,63,2,8);C(0,26,0,7);F();
		C(34,42,1,2);B();F();
		C(19,30,1,2);F();
		C(8,39,1,4);C(8,12,1,4);F();
		C(0,3,0,5);C(34,42,0,2);C(0,1,1,6);F();
		C(40,49,1,5);F();
		C(14,21,0,2);F();
		C(6,52,1,2);F();
		C(0,26,1,1);F();
		C(11,30,1,1);F();
		C(0,19,1,4);C(9,57,1,3);C(8,11,1,2);C(0,10,1,4);F();
		return 2.587821;
	}

	double compute_indices_3 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(2,20,0,1);F();
		C(0,2,0,4);C(0,91,0,8);F();
		C(0,2,0,1);C(0,2,0,1);C(2,4,1,2);F();
		B();C(0,1,1,1);C(23,49,1,2);C(7,40,0,4);C(8,9,0,4);C(1,8,0,7);F();
		C(15,103,0,1);C(0,4,0,5);C(3,15,1,1);C(12,18,0,3);F();
		C(46,147,1,1);F();
		C(83,225,1,2);F();
		C(5,7,2,1);C(3,8,0,2);C(20,113,0,3);C(4,6,0,4);C(0,7,1,7);C(5,25,0,2);C(8,163,0,8);F();
		C(2,9,0,2);F();
		C(93,141,0,3);C(5,9,1,4);C(1,11,0,7);F();
		C(6,27,0,5);F();
		C(0,10,1,4);C(8,163,1,3);F();
		C(0,259,1,8);C(4,6,0,4);F();
		C(0,2,2,4);F();
		C(0,4,0,5);C(2,10,0,5);C(8,20,0,5);C(0,4,0,5);C(9,221,1,6);C(0,3,0,1);C(0,11,1,3);F();
		C(1,61,0,2);C(3,8,0,5);F();
		C(0,20,1,5);C(48,76,1,5);C(27,51,1,4);F();
		C(2,4,0,2);C(2,13,1,3);F();
		C(5,13,1,4);C(7,19,0,2);C(5,36,1,2);F();
		C(2,5,0,4);C(50,54,2,5);C(0,65,1,6);F();
		B();C(24,31,1,6);F();
		C(0,18,1,7);F();
		B();C(9,47,0,2);F();
		C(2,55,0,1);F();
		C(61,78,0,3);F();
		C(10,12,1,4);C(0,5,0,2);C(56,86,1,5);C(22,63,0,6);F();
		B();B();C(39,85,1,2);C(1,5,0,4);C(24,31,1,6);F();
		C(26,35,1,7);C(9,18,0,5);F();
		B();B();C(0,5,0,8);C(0,7,1,1);F();
		C(0,4,1,7);C(1,6,1,2);C(0,208,1,5);C(0,14,1,4);C(8,20,1,2);C(11,67,1,7);C(29,33,1,4);F();
		C(46,147,1,1);C(62,64,0,5);C(84,263,1,2);C(6,11,1,4);C(7,20,0,4);C(0,36,0,7);C(0,75,1,4);C(62,64,0,5);F();
		C(0,1,1,6);C(11,57,0,3);C(0,27,0,8);F();
		return 3.694038;
	}

	double compute_indices_4 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(49,63,2,3);F();
		C(0,2,0,2);F();
		B();F();
		C(0,6,0,7);F();
		C(0,6,1,6);F();
		C(0,10,0,2);F();
		C(1,13,1,4);C(0,5,0,8);F();
		C(11,20,1,5);F();
		C(8,12,0,6);C(12,24,1,2);C(0,2,1,2);C(15,16,1,8);C(12,24,1,2);F();
		C(7,41,1,7);F();
		C(0,5,0,7);C(40,42,1,5);C(10,20,0,5);C(0,5,0,7);C(4,11,1,5);F();
		C(0,1,1,1);B();C(34,42,1,2);C(0,10,0,7);F();
		C(9,266,1,8);F();
		C(22,35,0,5);C(15,53,0,4);F();
		C(5,65,0,1);C(5,65,1,1);F();
		C(56,86,1,5);C(11,12,1,3);C(0,7,1,7);C(0,7,1,7);B();F();
		B();C(18,24,0,4);C(18,24,0,4);C(0,116,0,1);C(18,24,0,4);C(0,40,0,7);F();
		C(0,5,0,3);C(43,154,1,1);C(6,12,1,6);C(15,103,0,1);C(28,29,0,2);F();
		C(14,21,0,1);F();
		C(0,26,0,6);C(0,5,1,2);C(0,13,1,6);C(0,5,1,2);C(70,338,0,3);C(56,346,1,3);C(0,3,0,8);C(5,13,1,4);C(56,346,1,3);C(21,22,1,3);C(9,18,1,4);C(5,13,1,4);B();F();
		C(2,16,1,8);C(80,498,0,2);C(0,12,0,5);F();
		C(0,5,0,4);C(31,38,0,3);C(31,38,0,3);C(121,785,1,3);F();
		C(0,97,1,7);C(121,785,0,7);C(6,14,0,8);F();
		C(0,6,1,2);C(118,785,0,2);C(0,15,0,6);C(0,6,1,1);F();
		C(3,4,0,5);C(14,20,1,1);C(57,68,1,7);C(114,806,0,6);C(16,93,0,3);C(11,12,0,6);C(19,33,0,3);C(3,4,0,8);F();
		C(114,806,0,8);C(9,14,0,6);F();
		C(105,808,1,6);C(1,6,0,4);C(6,11,0,5);C(17,26,1,3);C(1,221,0,5);C(0,204,1,5);C(0,204,1,5);F();
		B();C(3,8,1,4);C(97,844,0,5);C(4,5,1,8);C(12,18,0,2);B();C(0,5,0,5);C(13,23,0,4);C(55,725,1,2);C(27,78,1,4);C(84,409,1,5);C(0,14,0,1);F();
		C(4,9,1,1);C(97,844,1,1);C(3,23,1,6);F();
		B();C(97,844,0,2);C(24,31,1,6);C(57,63,2,3);F();
		C(0,10,0,8);C(0,10,1,6);C(3,8,1,6);C(0,17,1,6);C(97,844,1,4);C(31,37,0,2);C(31,37,0,2);C(3,8,1,3);F();
		C(97,844,1,5);C(97,844,0,5);C(0,4,0,1);C(0,4,1,7);F();
		return 6.427857;
	}

	double compute_indices_5 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,70,0,3);C(0,1,1,8);C(0,16,1,8);C(8,79,0,2);C(0,43,1,7);C(2,20,0,1);C(24,30,1,2);C(7,14,1,1);C(0,1,2,8);F();
		B();C(0,2,1,5);C(0,208,1,6);F();
		C(51,134,0,3);F();
		B();C(0,10,1,3);F();
		C(0,5,1,2);C(0,4,0,5);F();
		C(0,6,1,7);B();B();C(20,108,1,3);B();C(0,255,1,4);F();
		C(0,3,1,8);F();
		B();C(3,8,1,5);C(9,16,0,3);B();B();C(99,142,0,2);F();
		C(0,8,0,6);F();
		C(44,55,2,1);C(0,2,1,4);C(0,13,0,7);F();
		C(6,9,1,4);C(0,1,1,5);B();C(5,43,1,3);F();
		C(14,30,0,1);F();
		C(2,5,2,3);F();
		C(8,163,1,8);F();
		C(0,2,1,5);F();
		C(0,7,1,2);C(0,8,1,8);C(21,27,1,8);C(0,18,1,3);B();C(40,42,0,6);F();
		C(0,275,1,5);C(0,255,1,6);C(0,15,1,3);F();
		C(82,132,0,3);F();
		C(4,8,0,3);F();
		C(0,79,0,4);F();
		C(2,16,0,8);C(2,9,1,3);F();
		C(0,12,0,7);C(0,40,0,4);C(48,63,1,5);C(66,139,0,3);C(10,54,1,2);F();
		C(0,2,2,7);F();
		C(0,40,0,4);C(25,46,1,4);C(25,46,1,8);C(2,55,1,6);F();
		C(2,55,0,1);C(0,44,1,7);C(41,63,2,7);B();F();
		C(36,60,1,8);C(62,64,0,5);C(2,4,0,2);F();
		C(39,85,0,2);F();
		C(18,106,0,7);F();
		C(27,34,1,1);C(2,9,0,1);C(8,113,0,3);C(0,17,1,6);B();B();F();
		C(0,208,0,5);C(0,14,1,4);C(1,6,1,2);C(0,14,1,4);F();
		B();C(46,147,0,1);B();C(62,64,0,1);C(11,23,1,1);F();
		C(9,221,1,6);F();
		return 3.936192;
	}

	double compute_indices_6 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(2,20,1,5);C(0,62,0,3);B();C(2,54,1,2);C(0,16,1,8);C(75,80,0,6);C(8,12,0,2);C(1,22,1,2);B();B();F();
		C(2,16,0,8);C(0,4,0,5);C(0,2,0,5);C(32,70,0,1);C(0,6,1,1);C(0,4,0,3);F();
		B();C(0,2,0,1);C(0,2,0,2);F();
		C(1,8,0,4);F();
		C(9,47,0,2);F();
		C(4,19,0,7);F();
		C(0,15,0,3);F();
		C(48,63,2,5);C(11,57,0,3);C(0,13,0,4);F();
		C(2,15,1,8);C(0,255,0,2);C(0,253,0,8);C(0,4,0,7);C(0,6,0,6);F();
		C(5,46,0,7);C(0,58,1,3);F();
		C(0,16,0,5);F();
		C(0,260,0,1);F();
		C(0,5,1,2);C(0,5,1,2);C(0,4,0,6);C(0,2,1,5);F();
		B();C(0,6,1,6);C(12,24,0,3);C(0,2,0,4);C(9,16,0,3);C(7,13,0,6);F();
		C(14,31,0,7);F();
		C(14,21,0,2);C(0,1,1,6);C(0,8,0,4);C(0,24,1,3);C(0,1,0,8);F();
		B();C(123,635,0,1);C(0,4,1,4);B();C(2,6,1,8);F();
		C(38,65,0,2);C(0,42,0,1);F();
		C(11,30,0,4);C(31,37,0,2);C(121,782,0,2);F();
		C(23,31,0,1);B();C(46,147,0,1);C(5,7,0,2);B();C(7,11,0,3);C(0,698,0,1);F();
		C(0,698,0,1);C(0,17,0,2);C(38,65,1,2);C(8,33,0,1);F();
		C(31,38,1,3);C(31,37,0,2);C(21,27,1,4);F();
		C(0,11,0,2);F();
		C(0,105,1,6);F();
		C(0,116,0,1);C(13,15,0,3);F();
		C(0,6,0,3);F();
		C(33,90,0,2);F();
		C(39,85,0,7);C(7,13,0,6);F();
		C(79,415,0,1);F();
		C(17,85,0,5);F();
		C(0,14,1,6);C(14,26,1,5);B();C(46,61,0,4);B();C(0,11,1,4);C(103,266,0,3);C(25,31,1,2);C(8,20,0,7);F();
		C(2,6,1,4);C(106,650,1,6);C(0,153,1,6);C(73,445,1,5);F();
		return 1.272366;
	}

	double compute_indices_7 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,22,0,5);C(0,12,0,8);C(0,16,1,8);C(0,65,1,2);C(65,403,1,2);F();
		C(5,13,2,5);C(15,16,1,4);C(1,8,0,5);C(1,36,1,8);C(8,39,0,1);B();C(0,8,0,8);C(0,4,1,4);F();
		B();F();
		C(0,2,1,5);C(0,1,0,1);C(0,1,1,4);C(0,4,1,3);F();
		C(0,13,0,5);C(0,17,0,5);C(42,62,0,2);F();
		C(0,246,0,5);F();
		C(7,41,1,3);F();
		B();C(2,252,1,2);C(0,4,0,2);C(2,252,1,1);C(0,43,0,8);C(0,232,1,1);F();
		C(0,13,1,6);C(10,26,1,1);C(0,2,1,4);C(0,230,0,4);F();
		B();B();F();
		C(0,23,1,4);B();C(0,8,1,1);F();
		B();C(0,116,0,1);C(18,24,0,4);C(18,24,0,4);F();
		C(0,6,0,8);C(111,651,1,2);C(35,55,2,7);F();
		C(8,12,0,5);C(9,16,0,3);C(39,85,0,2);F();
		B();C(0,44,1,7);F();
		C(0,15,0,7);F();
		C(49,51,0,1);F();
		B();C(93,143,0,2);C(0,4,1,6);C(0,3,0,1);C(17,26,1,1);C(1,6,1,4);F();
		C(42,150,1,2);C(0,4,0,3);C(5,13,0,4);C(12,18,0,2);C(17,26,1,7);F();
		C(3,11,0,4);B();C(8,37,0,1);F();
		C(83,225,0,5);F();
		C(0,13,1,2);B();C(31,37,0,1);C(0,22,0,5);C(34,42,1,2);F();
		C(34,42,1,2);C(1,11,0,4);F();
		C(0,40,1,4);C(30,40,0,2);C(0,40,0,4);C(2,7,1,1);C(0,40,1,4);B();C(2,55,1,6);C(0,16,0,8);C(37,60,0,2);C(2,162,1,1);C(40,54,0,4);F();
		C(57,82,0,1);C(2,9,1,4);C(0,257,0,4);C(0,4,0,7);C(2,15,1,8);F();
		C(56,86,1,5);C(8,22,1,2);C(22,63,0,6);C(14,258,0,1);C(0,38,0,8);C(22,35,0,5);C(62,66,0,7);C(0,2,1,6);C(0,11,0,2);C(0,2,0,8);C(6,98,1,3);C(0,41,1,3);B();C(0,17,0,4);C(7,14,1,1);C(25,46,0,2);C(25,37,1,4);F();
		C(0,153,1,3);C(114,650,1,2);C(6,712,1,5);C(8,718,1,4);F();
		C(3,21,0,5);F();
		C(15,63,1,6);C(43,154,1,1);C(6,12,1,6);F();
		C(0,55,0,3);C(0,208,0,5);C(1,6,1,2);C(2,6,0,5);C(0,4,1,7);F();
		B();C(62,64,0,5);C(82,264,1,3);C(15,25,0,7);F();
		C(82,434,1,5);C(114,650,1,2);C(0,153,1,6);C(8,718,1,4);C(6,712,1,5);C(0,255,1,2);C(6,712,1,5);F();
		return 6.132593;
	}

	double compute_indices_8 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(26,260,0,1);F();
		B();B();C(0,5,2,4);C(57,68,1,7);F();
		C(0,7,1,8);C(7,14,0,1);C(52,70,1,5);C(8,17,0,4);C(5,43,0,3);C(0,15,1,6);C(40,42,0,5);C(0,38,1,6);C(2,55,1,6);F();
		C(0,255,0,1);F();
		C(0,10,1,2);C(8,26,1,1);C(0,4,0,3);C(0,4,1,7);C(5,9,1,3);F();
		B();C(0,6,0,8);B();C(2,6,1,2);F();
		C(0,14,1,2);C(46,147,1,1);F();
		C(12,18,1,4);C(44,79,0,5);C(8,15,1,4);C(2,6,1,6);F();
		C(0,281,1,5);C(22,63,1,1);F();
		C(0,6,0,4);C(1,264,0,8);C(0,16,1,7);C(9,51,0,3);C(9,18,0,5);F();
		C(0,2,1,2);C(0,15,1,6);C(2,9,1,4);C(0,24,1,5);F();
		C(0,13,0,5);F();
		C(5,65,0,1);F();
		C(0,260,0,1);F();
		C(19,30,1,2);C(0,43,0,8);F();
		C(0,6,2,7);F();
		C(9,17,0,7);F();
		C(0,40,0,6);F();
		C(0,19,0,2);C(31,70,0,3);C(6,15,1,3);C(22,28,1,7);F();
		C(2,5,0,4);C(56,86,1,5);B();C(23,31,0,1);C(5,43,0,2);C(68,425,1,1);C(35,55,2,7);F();
		C(3,11,0,7);F();
		B();B();C(31,38,1,3);C(0,11,1,8);C(0,7,0,6);C(0,21,0,1);C(97,844,0,2);C(41,52,1,4);C(0,30,1,2);C(6,14,1,1);C(2,21,0,7);C(4,35,1,3);F();
		C(34,42,1,1);B();C(18,24,0,1);C(0,49,0,3);F();
		C(9,23,1,5);F();
		C(7,20,1,2);C(31,38,1,3);F();
		C(0,8,1,5);B();C(0,36,0,7);F();
		C(0,8,0,6);F();
		C(0,30,1,1);F();
		B();C(0,8,0,3);C(8,101,0,3);C(0,17,1,6);C(10,264,0,1);F();
		C(0,253,1,4);C(2,11,1,5);F();
		C(0,17,0,7);B();C(62,64,0,5);C(82,264,0,3);C(8,12,1,1);F();
		C(0,9,1,3);F();
		return 5.306239;
	}

	double compute_indices_9 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(2,20,0,1);C(0,60,0,3);C(0,16,1,8);C(0,253,1,8);F();
		C(0,1,0,8);F();
		C(0,2,0,1);C(41,57,1,8);C(37,57,1,2);C(0,26,0,6);F();
		C(0,4,0,4);C(0,2,0,4);C(0,1,1,1);B();C(51,174,1,7);C(0,1,1,3);C(0,2,0,4);C(0,17,1,5);C(11,30,1,1);C(11,30,1,1);F();
		C(0,18,0,6);F();
		C(0,2,1,8);C(0,2,2,5);C(8,20,0,2);C(0,1,1,4);F();
		C(0,3,1,8);C(0,8,1,2);C(0,15,1,3);F();
		C(0,68,1,8);F();
		B();C(0,26,1,7);B();C(2,9,0,2);F();
		C(0,260,0,1);F();
		C(0,15,1,6);B();F();
		C(0,14,0,5);F();
		C(8,12,1,4);F();
		C(14,16,1,5);C(9,16,1,3);C(3,5,1,5);C(1,5,0,7);F();
		C(0,2,2,3);C(38,65,0,2);F();
		B();B();C(4,38,1,3);C(25,46,0,2);C(0,1,0,4);C(1,11,0,5);C(0,14,1,5);C(1,4,0,4);F();
		C(1,8,0,3);F();
		B();C(57,138,1,2);C(22,255,1,7);C(59,711,1,2);F();
		C(5,13,0,6);C(0,116,1,3);C(13,23,0,4);F();
		C(11,16,1,6);F();
		C(0,78,0,8);C(0,17,1,1);C(0,54,1,3);C(5,8,0,2);C(0,2,2,4);F();
		C(31,64,0,5);C(0,1,1,5);C(27,51,1,4);C(6,40,0,5);C(17,26,1,3);F();
		C(34,42,1,2);B();C(6,712,1,5);C(43,62,0,3);F();
		C(0,1,1,7);C(11,30,0,4);C(82,434,1,2);F();
		C(34,42,1,2);F();
		C(19,65,1,3);F();
		C(3,8,1,5);C(39,85,0,2);C(9,24,0,1);C(0,116,0,3);F();
		C(26,171,0,6);C(1,3,1,6);C(41,53,1,1);C(0,1,1,6);C(42,62,0,2);C(0,11,1,6);C(61,78,0,8);C(8,20,1,2);C(37,57,1,2);C(3,69,1,3);C(0,9,1,5);F();
		B();C(0,19,1,7);C(0,17,1,6);C(0,2,0,6);F();
		C(0,14,1,8);C(5,209,0,4);C(1,6,1,2);C(0,4,1,7);F();
		B();C(62,64,0,5);C(46,147,1,1);F();
		C(0,18,1,3);C(0,34,0,4);C(114,650,1,2);C(85,420,1,5);C(0,153,1,6);C(82,434,1,5);C(0,153,1,3);F();
		return 5.371660;
	}

	double compute_indices_10 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,3,0,4);F();
		C(0,3,0,5);F();
		C(0,6,0,4);F();
		C(1,6,1,3);F();
		C(0,8,0,2);F();
		C(2,9,1,1);F();
		B();C(8,9,0,2);C(1,11,1,8);C(1,12,1,8);F();
		C(0,9,1,1);F();
		C(5,10,1,2);F();
		C(0,5,1,2);C(0,12,0,5);F();
		C(0,15,1,6);C(0,3,1,3);F();
		B();C(15,16,0,2);F();
		C(7,13,0,5);C(0,16,1,8);B();F();
		C(9,18,1,8);F();
		C(0,3,0,4);C(0,18,1,3);C(16,17,1,3);B();C(0,7,1,8);C(0,6,0,3);C(0,2,0,5);C(0,4,1,3);F();
		C(0,2,2,3);C(12,19,1,1);B();F();
		C(7,8,0,3);B();C(9,16,1,1);C(0,19,0,1);C(0,5,1,8);F();
		C(2,13,0,1);F();
		C(14,21,0,3);F();
		C(0,12,0,5);C(5,9,1,4);C(1,22,0,7);F();
		C(0,4,1,4);C(17,26,1,3);C(3,4,1,2);C(0,4,1,1);F();
		C(21,27,1,4);C(1,12,0,5);C(0,4,1,1);C(21,27,1,4);F();
		C(0,19,0,2);F();
		C(22,28,0,7);F();
		C(24,31,1,7);F();
		B();C(18,19,1,7);C(0,36,1,7);F();
		C(31,38,1,3);B();C(31,37,0,8);F();
		C(37,49,1,3);C(0,2,0,1);F();
		C(34,42,1,2);F();
		C(50,55,1,1);B();F();
		C(9,18,0,8);C(48,63,2,5);F();
		C(64,71,0,7);C(64,71,0,7);F();
		return 3.215231;
	}

	double compute_indices_11 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,18,0,5);C(0,21,0,2);F();
		C(5,11,1,1);C(0,35,1,7);F();
		C(11,28,1,6);C(0,2,1,5);B();F();
		C(31,38,1,4);C(29,33,1,4);F();
		C(26,37,0,2);F();
		C(0,6,0,8);B();C(0,3,1,6);F();
		C(7,16,0,8);F();
		C(8,20,0,2);C(48,61,2,7);F();
		C(1,11,0,7);F();
		C(0,15,0,1);F();
		B();F();
		B();C(0,3,1,4);C(1,12,1,4);C(6,8,1,1);F();
		C(5,13,0,5);C(46,54,2,5);F();
		C(15,16,1,4);C(9,16,0,3);C(8,12,1,8);F();
		C(6,25,1,2);B();F();
		C(6,20,1,3);B();C(0,7,1,4);B();C(14,21,0,8);C(2,18,1,2);F();
		B();C(18,24,0,5);F();
		C(0,5,1,7);C(1,6,1,4);B();B();C(0,1,0,1);C(4,6,0,5);C(17,26,0,3);B();F();
		C(1,5,1,6);F();
		B();C(23,31,0,1);C(2,5,0,4);C(0,6,0,3);F();
		C(1,11,1,8);C(24,31,0,2);F();
		C(31,37,0,4);C(31,38,1,5);C(0,2,1,7);F();
		C(34,42,1,2);B();F();
		C(0,40,1,2);C(25,36,1,2);C(0,40,0,4);C(22,47,1,3);F();
		C(0,9,0,3);F();
		C(37,49,1,5);F();
		C(0,1,0,3);F();
		C(0,4,1,4);C(0,2,0,3);F();
		C(0,14,1,5);F();
		C(0,2,2,8);C(0,2,1,4);F();
		C(0,2,0,6);F();
		C(0,14,0,6);F();
		return 2.291307;
	}

	double compute_indices_12 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,16,1,8);C(2,7,0,8);C(0,68,1,3);F();
		C(0,2,1,5);B();C(0,281,1,6);C(0,244,0,5);F();
		C(0,2,0,1);F();
		C(0,2,0,4);C(0,4,1,4);F();
		C(0,4,0,5);F();
		B();C(6,11,1,4);F();
		C(0,12,1,6);C(0,18,1,3);C(0,18,1,3);C(0,281,0,6);F();
		C(0,10,2,4);C(3,8,0,5);B();C(88,136,1,2);C(7,22,0,3);F();
		B();C(2,9,0,2);C(2,9,1,4);F();
		C(0,48,0,4);C(0,153,1,8);F();
		C(12,257,1,4);F();
		C(0,18,0,2);F();
		C(0,14,0,5);C(0,14,0,1);C(59,146,0,3);F();
		C(9,16,0,3);C(15,16,1,4);C(15,16,1,4);F();
		C(1,10,1,6);C(17,26,1,7);C(0,15,1,4);C(0,15,1,4);F();
		B();C(0,5,0,4);B();C(0,7,1,4);B();B();F();
		C(18,24,0,4);C(0,3,1,1);F();
		C(1,6,1,4);B();B();B();C(17,26,1,3);C(8,163,1,8);F();
		C(4,6,0,7);C(5,13,0,4);C(6,27,0,4);C(7,19,0,2);F();
		C(0,1,0,1);F();
		B();C(27,30,1,6);F();
		C(31,37,1,2);C(31,37,1,2);C(31,38,1,3);B();C(6,11,1,3);C(14,31,0,7);F();
		C(34,42,1,2);B();F();
		C(25,46,1,2);C(0,40,0,4);C(0,40,0,4);C(21,32,0,2);C(56,63,2,8);F();
		C(8,39,1,4);C(2,55,0,1);F();
		C(56,86,1,5);B();C(22,63,0,1);F();
		B();C(39,85,0,2);F();
		B();F();
		C(0,17,1,6);B();C(0,116,0,3);B();F();
		C(12,68,0,1);F();
		B();C(5,8,0,2);F();
		C(114,650,1,2);C(6,712,1,5);C(82,434,1,5);C(8,718,1,4);F();
		return 4.576463;
	}

	double compute_indices_13 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		B();C(3,37,1,4);C(82,434,1,2);F();
		C(59,711,1,2);C(6,24,1,1);F();
		C(2,151,1,3);C(0,5,0,2);C(111,144,1,6);C(0,208,1,6);C(0,15,1,4);C(76,417,1,5);C(0,1,0,4);C(0,11,1,1);F();
		C(82,434,1,5);F();
		C(0,710,1,1);C(0,153,1,6);C(114,650,1,2);F();
		C(0,1,1,6);C(0,3,0,2);F();
		C(22,63,0,1);C(36,60,1,8);C(60,63,2,6);C(0,233,1,2);C(11,20,1,8);C(4,20,1,4);C(9,11,1,1);F();
		C(0,5,1,7);C(80,498,1,2);C(46,147,1,1);F();
		C(5,46,0,7);C(1,252,0,1);C(12,268,1,2);C(0,3,1,1);C(45,351,1,2);C(0,2,0,1);F();
		C(14,259,1,6);C(97,844,1,3);C(1,12,1,8);C(111,646,1,7);C(0,6,1,3);F();
		C(61,69,0,5);C(14,726,1,5);F();
		C(1,6,1,4);C(0,4,1,7);B();B();C(76,417,1,5);C(9,91,1,5);C(0,59,1,1);C(36,48,0,4);F();
		C(17,26,0,6);C(6,142,1,4);C(0,253,1,7);C(76,417,1,1);F();
		C(0,153,1,3);C(114,650,1,2);C(8,718,1,4);C(6,712,1,5);F();
		C(12,253,1,4);C(7,14,0,1);C(12,18,0,2);C(0,9,1,6);C(42,62,0,2);C(0,18,1,5);C(61,78,0,8);C(41,53,1,1);C(0,69,1,3);F();
		C(82,434,1,5);C(18,176,1,5);F();
		C(12,31,0,4);C(0,3,1,1);C(120,838,1,6);C(22,31,0,4);C(0,4,2,6);C(0,208,1,6);F();
		C(0,6,0,6);B();F();
		C(9,10,1,7);C(22,253,1,4);C(2,5,2,3);C(0,230,0,4);C(0,18,1,1);C(0,43,1,5);B();C(51,60,0,8);C(76,417,1,1);F();
		C(79,415,0,1);B();F();
		C(31,38,1,7);C(22,24,1,8);C(0,715,1,3);F();
		C(0,3,0,7);C(0,54,1,2);C(111,144,1,6);C(82,434,1,2);F();
		C(34,42,0,4);C(0,1,0,4);C(15,21,1,2);C(0,10,1,2);C(83,225,1,2);C(2,6,1,7);C(0,719,1,4);C(8,27,1,5);F();
		C(0,255,1,5);C(0,255,1,5);C(2,5,2,3);C(0,255,1,5);C(0,253,1,8);C(3,246,0,4);C(1,8,1,1);C(56,346,1,3);C(6,25,0,4);F();
		C(12,27,1,4);C(18,21,0,8);C(10,711,0,7);C(8,163,0,3);F();
		C(114,650,1,2);C(64,101,1,7);F();
		C(0,31,1,4);C(54,704,1,2);C(3,214,1,2);C(56,346,1,4);F();
		B();C(0,259,1,3);C(97,844,1,5);F();
		C(0,59,0,1);B();C(31,37,0,2);C(31,38,0,3);C(0,59,0,1);C(0,9,0,1);F();
		C(97,844,1,3);C(9,266,1,8);C(5,8,0,1);C(11,22,2,4);C(123,646,1,5);C(45,351,1,2);C(81,429,1,4);C(5,8,0,1);F();
		C(0,3,1,8);C(46,147,0,1);B();C(2,55,1,6);C(0,5,1,1);C(0,1,0,1);C(8,39,1,1);F();
		C(80,498,0,2);F();
		return 1.488160;
	}

	double compute_indices_14 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,68,1,8);F();
		C(24,103,1,1);F();
		C(0,2,0,1);B();F();
		B();C(0,1,1,1);B();C(0,4,1,3);F();
		C(0,5,1,2);C(0,4,0,6);F();
		B();C(1,5,1,1);C(0,6,0,8);F();
		B();F();
		C(0,8,1,5);C(5,10,0,8);F();
		C(2,9,1,4);C(2,9,0,7);C(0,12,1,8);F();
		C(1,11,0,7);F();
		C(0,2,0,2);F();
		C(0,8,0,1);F();
		C(0,14,1,6);C(6,27,1,4);F();
		C(6,21,0,1);C(15,16,1,4);C(0,20,0,4);C(0,12,1,7);C(1,11,0,3);F();
		B();C(0,16,1,8);F();
		C(6,14,0,4);C(0,3,1,7);F();
		C(6,13,1,1);C(0,10,1,1);F();
		C(18,24,0,4);C(0,2,0,2);C(0,3,1,2);F();
		C(0,9,1,6);F();
		C(0,15,1,5);C(2,5,0,5);B();C(20,26,0,6);F();
		B();C(24,31,1,6);F();
		C(31,37,0,4);C(31,38,1,3);C(12,23,1,3);C(0,4,1,1);F();
		C(20,22,1,2);F();
		C(0,16,1,8);F();
		C(0,14,0,8);C(5,23,0,4);F();
		C(0,36,0,8);F();
		C(1,6,1,2);F();
		C(2,8,0,4);B();C(24,31,0,6);F();
		C(7,19,1,3);C(0,17,1,1);C(0,18,0,2);C(0,4,1,6);F();
		C(61,78,1,3);F();
		C(14,21,1,6);C(9,13,0,3);F();
		B();C(28,66,1,6);F();
		return 3.139493;
	}

	double compute_indices_15 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,19,1,2);C(0,62,1,3);F();
		C(14,31,0,7);F();
		C(0,2,0,1);F();
		C(0,10,1,7);C(0,2,1,4);C(0,10,0,3);F();
		C(0,2,2,5);C(0,9,1,2);F();
		C(0,10,1,5);C(7,16,0,6);C(8,11,0,5);C(1,5,1,4);C(6,14,0,8);F();
		C(0,3,0,8);C(0,10,0,4);C(0,3,1,8);F();
		C(0,18,0,3);B();C(0,208,1,7);C(99,142,0,2);C(1,11,0,5);F();
		C(0,5,2,4);F();
		C(1,6,1,2);F();
		C(0,4,0,1);C(0,4,0,5);C(0,5,1,2);F();
		C(8,20,0,7);F();
		C(12,18,1,7);F();
		C(2,10,0,5);F();
		C(2,6,1,7);F();
		C(0,1,0,6);F();
		B();C(18,24,1,4);F();
		B();C(1,6,1,4);B();C(17,26,1,5);B();F();
		C(64,264,1,5);F();
		B();F();
		B();F();
		C(20,108,1,3);C(49,57,1,5);C(8,20,1,2);F();
		C(0,4,0,2);C(23,147,1,2);F();
		C(0,40,1,4);C(0,33,0,4);C(25,46,1,2);C(6,27,0,4);F();
		C(0,51,1,2);C(2,55,0,1);C(46,147,1,4);C(0,17,0,4);F();
		C(6,11,1,2);C(7,41,1,7);C(0,16,0,1);F();
		C(39,85,0,7);C(9,13,0,7);F();
		C(7,8,1,3);F();
		C(0,3,0,6);F();
		C(0,4,1,7);C(0,14,1,4);C(1,6,1,2);C(0,4,1,7);C(0,14,1,4);C(3,7,1,8);C(0,8,0,8);F();
		C(0,4,0,1);F();
		C(5,36,1,8);C(8,54,0,5);F();
		return 1.265077;
	}

	double compute_indices_16 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(2,76,1,2);C(2,20,0,1);C(9,31,1,2);F();
		B();C(0,2,2,5);C(22,28,1,7);F();
		C(0,2,0,1);F();
		C(0,4,1,4);C(0,2,0,4);F();
		C(0,4,0,4);C(0,16,1,5);F();
		C(0,7,1,7);C(0,1,2,3);C(7,8,1,8);C(1,7,0,6);F();
		C(6,14,1,1);C(6,14,1,8);F();
		C(6,15,1,7);C(0,10,0,3);F();
		C(12,24,1,2);F();
		C(1,11,1,3);C(5,9,1,4);F();
		C(0,5,1,3);B();F();
		B();F();
		B();C(0,1,0,4);F();
		C(12,233,1,2);C(2,162,1,2);C(0,11,0,6);F();
		C(0,2,0,2);F();
		C(0,4,2,3);F();
		C(18,24,0,4);F();
		C(6,18,1,3);C(6,11,1,1);C(3,7,1,2);C(1,6,1,3);F();
		C(0,1,0,4);C(0,261,0,5);C(4,24,1,8);F();
		B();C(8,39,1,4);F();
		B();C(1,5,0,4);C(24,31,1,6);F();
		C(0,4,0,1);F();
		C(0,15,0,3);C(34,42,1,8);B();C(17,21,0,7);C(11,54,1,6);C(31,38,1,3);C(14,36,1,6);C(16,18,1,1);C(18,19,1,8);C(29,33,1,2);F();
		C(1,118,1,3);F();
		C(49,74,1,7);C(2,55,1,1);C(22,31,1,4);F();
		C(49,83,0,8);C(2,4,1,6);C(22,63,0,1);F();
		B();C(39,85,1,2);F();
		B();B();F();
		C(1,3,1,1);F();
		C(46,147,1,1);F();
		C(23,31,0,4);C(2,9,1,1);B();F();
		C(0,42,0,5);B();C(0,43,1,7);B();C(7,19,0,2);C(20,26,0,6);C(0,11,1,4);C(13,48,0,6);C(0,3,0,6);C(8,39,0,1);F();
		return 1.078297;
	}

	double compute_indices_17 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(9,19,1,1);C(0,16,1,8);C(0,60,0,7);C(0,38,1,8);F();
		C(5,9,1,3);C(0,2,2,5);F();
		C(0,2,0,1);C(17,26,1,4);F();
		C(0,4,0,1);B();B();B();F();
		C(0,4,0,2);C(0,65,1,2);C(0,5,1,7);C(2,13,1,1);F();
		C(0,8,1,2);F();
		C(1,4,0,8);F();
		B();C(0,3,1,6);C(3,5,1,6);C(10,22,0,3);F();
		B();F();
		C(8,10,1,8);C(0,9,1,4);F();
		C(0,15,1,8);F();
		C(0,1,0,3);C(0,1,0,3);F();
		C(21,47,1,6);C(0,40,0,4);C(25,46,1,2);C(38,65,0,2);F();
		B();C(7,41,0,1);C(9,16,0,2);B();C(0,20,0,6);C(13,14,0,4);C(0,15,1,6);B();F();
		C(24,31,1,6);C(8,20,1,2);B();C(3,4,0,6);C(0,7,1,4);F();
		C(0,8,0,6);C(0,3,1,8);C(0,11,1,4);C(14,21,0,6);B();C(0,17,0,8);C(31,38,1,2);F();
		C(18,27,1,4);F();
		C(5,13,1,4);F();
		C(8,14,0,7);C(7,19,0,2);F();
		C(19,42,1,8);F();
		B();F();
		C(0,6,0,8);C(0,8,0,3);C(0,3,0,2);F();
		C(0,1,1,7);F();
		C(2,5,1,5);F();
		C(0,41,1,3);F();
		C(56,86,0,5);C(5,65,0,1);F();
		C(3,16,0,1);C(39,85,0,4);F();
		C(0,14,0,5);B();F();
		B();C(0,4,1,6);C(8,12,0,3);C(0,17,0,6);B();C(0,116,1,3);C(0,10,0,3);F();
		B();C(5,25,0,2);C(8,39,1,4);C(56,86,1,5);C(0,7,1,4);C(17,26,1,3);F();
		C(0,9,0,3);F();
		C(0,3,0,6);F();
		return 1.485590;
	}

	double compute_indices_18 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(2,3,1,8);F();
		C(0,2,2,6);C(5,8,1,3);F();
		C(0,2,0,8);F();
		B();C(0,1,1,1);C(0,4,1,4);F();
		C(0,5,0,4);C(0,3,0,3);F();
		C(0,2,1,8);F();
		C(5,36,1,2);F();
		C(0,18,0,3);F();
		C(2,9,1,4);C(0,5,0,2);B();B();F();
		C(3,8,1,7);F();
		B();F();
		C(17,26,0,6);F();
		B();F();
		C(0,4,0,7);C(15,16,1,4);C(0,42,0,8);C(11,46,0,8);F();
		C(0,36,1,5);C(17,24,1,2);C(0,16,1,3);C(9,24,0,1);F();
		C(0,29,1,4);F();
		C(40,54,0,3);F();
		C(0,12,0,8);F();
		B();F();
		C(2,3,1,1);C(23,31,0,1);F();
		C(20,40,1,6);C(3,8,0,3);C(52,63,1,8);F();
		C(2,20,0,1);C(0,14,0,2);C(1,12,1,4);C(2,7,0,6);F();
		C(0,59,0,1);F();
		C(3,9,2,2);F();
		C(0,4,0,3);F();
		B();F();
		C(0,8,0,2);F();
		C(0,2,0,2);F();
		C(6,12,1,6);B();C(0,116,0,3);C(6,109,0,3);F();
		C(1,6,0,2);C(5,19,1,6);C(0,202,1,8);C(0,14,1,4);B();C(0,4,1,7);C(0,6,1,5);C(2,31,1,4);C(0,63,0,4);F();
		C(46,147,1,1);C(43,135,1,5);B();C(0,4,0,4);F();
		C(3,15,1,1);C(0,199,1,8);C(0,2,0,5);F();
		return 0.676698;
	}

	double compute_indices_19 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(2,20,0,1);C(0,65,1,2);C(0,60,0,1);C(0,16,1,1);F();
		C(19,65,1,3);F();
		C(0,2,0,1);C(0,5,1,2);F();
		C(0,2,0,4);F();
		B();F();
		C(0,3,1,8);B();F();
		C(3,8,1,5);F();
		C(82,132,0,3);C(2,22,0,1);C(0,4,0,5);C(24,31,1,5);F();
		C(0,17,1,8);C(0,44,1,3);F();
		C(9,221,1,5);C(13,29,1,5);C(0,2,1,4);B();B();F();
		C(1,8,0,6);B();C(0,18,1,6);F();
		C(0,5,0,3);F();
		C(0,4,0,1);F();
		C(0,9,1,2);C(0,7,0,8);F();
		B();F();
		C(0,18,1,3);B();C(0,7,1,8);C(0,13,0,2);C(0,23,1,3);F();
		C(0,1,1,1);F();
		C(0,255,1,8);F();
		C(5,13,1,4);C(6,27,0,4);C(7,19,0,2);C(8,163,1,2);C(0,8,1,1);F();
		C(2,5,2,3);F();
		B();C(0,21,1,1);C(24,31,0,7);F();
		C(6,27,0,5);F();
		C(0,9,1,8);C(0,9,1,1);F();
		C(0,40,0,4);C(3,11,0,4);C(0,40,0,4);C(25,44,0,3);C(0,14,0,2);C(0,8,0,5);C(6,21,1,5);C(0,4,1,6);C(17,26,1,7);C(0,15,1,2);C(0,15,1,4);F();
		C(0,1,1,1);C(8,39,1,4);C(7,32,0,2);C(12,25,1,3);F();
		C(22,58,0,3);B();C(0,6,1,1);C(56,86,1,7);C(2,4,1,2);F();
		C(2,13,1,1);F();
		B();F();
		C(0,10,0,4);C(20,24,0,2);F();
		C(0,208,1,5);C(0,4,1,1);C(2,6,1,2);C(0,14,1,4);C(1,6,1,2);C(0,4,1,5);F();
		B();C(50,54,2,5);C(0,8,2,2);F();
		C(2,6,1,5);C(0,8,1,2);F();
		return 0.810853;
	}

	double compute_indices_20 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(0,60,0,3);C(5,10,1,4);C(7,8,1,1);C(0,16,1,8);B();F();
		C(0,2,2,5);F();
		C(0,2,0,1);C(0,22,1,5);C(0,4,0,2);F();
		C(0,2,0,1);C(0,4,1,5);F();
		C(0,4,0,5);F();
		C(0,5,1,4);C(4,7,1,3);C(0,2,1,7);C(10,16,1,7);C(3,8,0,7);F();
		C(0,247,1,5);B();C(0,246,0,5);C(0,262,0,5);C(0,255,0,5);C(0,247,1,5);F();
		C(0,3,0,4);C(3,8,0,8);C(2,9,1,2);C(0,7,0,1);F();
		C(1,3,1,4);F();
		C(0,10,1,7);C(0,9,1,6);C(10,16,1,1);C(0,257,1,4);C(0,255,1,1);C(59,146,0,3);C(14,259,0,6);F();
		C(9,24,0,1);C(8,163,1,2);C(0,5,0,8);C(0,5,0,5);F();
		C(1,7,1,3);F();
		C(2,12,1,4);F();
		C(2,9,0,2);C(15,25,1,3);F();
		C(8,20,1,2);C(0,6,1,4);C(0,27,0,8);C(14,21,0,2);C(0,14,1,4);F();
		C(80,142,1,7);F();
		C(0,6,1,7);B();B();C(0,5,0,8);C(0,7,1,1);C(1,4,0,4);F();
		B();B();B();C(0,4,1,6);C(39,85,0,2);F();
		C(0,13,1,1);F();
		C(0,6,1,3);C(0,17,0,3);F();
		C(24,31,1,1);C(0,5,1,1);C(45,351,1,2);C(2,9,0,1);F();
		C(0,54,0,2);C(76,417,1,1);C(0,6,1,7);F();
		C(0,15,0,3);C(111,144,1,6);C(0,208,1,6);C(8,163,1,3);C(76,417,1,5);C(0,1,0,4);C(0,5,0,2);C(0,11,1,1);F();
		C(3,4,1,4);C(0,40,0,4);C(25,46,1,2);C(0,9,1,8);C(82,434,1,5);C(0,43,0,1);F();
		B();F();
		C(46,147,0,8);C(82,434,1,5);C(0,14,0,6);C(0,6,0,7);B();B();F();
		C(0,259,0,8);C(29,52,0,1);C(0,5,1,5);C(12,68,0,1);F();
		B();C(10,52,1,3);F();
		C(10,245,1,7);C(0,2,2,3);C(0,14,0,8);C(5,13,0,3);F();
		C(0,208,0,5);C(0,14,1,4);C(1,6,1,2);C(0,4,1,7);C(9,27,0,4);C(8,34,1,4);C(8,12,1,4);F();
		C(85,268,0,3);C(47,72,0,5);C(8,26,1,1);C(0,60,0,4);F();
		C(25,46,1,2);F();
		return 1.909132;
	}

	double compute_indices_21 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(20,22,2,8);F();
		C(2,11,0,5);C(0,2,2,5);C(0,3,1,2);C(0,9,1,7);F();
		C(0,2,1,7);C(0,8,1,1);C(0,6,0,8);F();
		C(0,1,1,3);C(0,4,1,3);C(2,14,1,5);C(0,10,0,4);F();
		C(0,4,0,2);C(0,5,1,2);C(0,5,1,2);F();
		C(0,8,0,5);C(0,12,1,4);F();
		C(12,19,1,4);F();
		C(3,8,1,5);C(0,60,1,6);C(49,69,0,7);F();
		B();C(2,9,1,3);F();
		C(1,11,0,7);F();
		C(0,15,1,6);C(0,7,1,1);C(0,8,0,3);F();
		C(5,6,0,2);B();C(5,9,1,3);C(1,12,1,4);F();
		C(0,8,0,2);C(0,2,1,8);C(0,7,1,2);B();C(0,2,0,6);F();
		C(9,16,1,3);C(8,12,1,6);C(15,16,1,4);F();
		B();C(0,6,1,5);C(8,20,1,2);F();
		B();C(15,24,1,8);C(0,28,1,7);B();C(0,18,1,3);C(14,21,0,1);F();
		C(13,32,1,6);C(0,4,0,8);F();
		C(2,11,1,1);C(1,6,1,4);C(0,6,0,3);B();C(2,13,1,8);B();C(2,11,1,1);C(0,3,1,4);C(0,6,0,3);C(2,13,1,8);F();
		C(9,16,0,3);C(8,12,0,4);C(11,16,1,4);F();
		C(51,63,2,5);F();
		C(24,31,1,6);F();
		B();C(31,37,0,2);C(31,38,1,7);F();
		C(0,5,0,2);C(34,42,1,6);F();
		C(16,26,1,1);F();
		C(23,34,1,1);F();
		C(14,21,1,5);F();
		C(0,6,1,1);C(0,1,0,7);C(37,57,1,2);F();
		C(17,26,1,3);F();
		B();C(11,24,1,2);C(10,15,0,1);F();
		C(0,1,2,4);F();
		C(17,30,1,8);F();
		B();C(31,37,0,2);F();
		return 4.758228;
	}

	double compute_indices_22 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,2,2,8);C(8,163,1,3);F();
		C(8,718,1,1);C(0,2,2,8);C(121,785,0,7);F();
		C(9,18,1,4);C(0,35,1,7);C(21,79,1,4);C(0,9,1,5);C(3,10,0,3);F();
		C(0,281,0,6);F();
		C(16,17,0,3);C(0,4,1,3);C(56,86,0,5);C(28,66,1,6);C(28,66,1,6);F();
		C(97,844,1,1);F();
		C(83,439,0,5);C(0,3,0,8);F();
		C(99,142,0,2);C(4,6,0,4);C(3,5,0,8);C(0,7,0,5);C(2,5,0,4);C(0,21,1,1);C(25,75,0,1);C(28,29,0,2);F();
		C(58,702,1,2);F();
		C(1,11,0,7);C(5,9,1,3);C(6,39,0,1);C(0,43,1,7);F();
		C(11,22,0,3);C(1,14,1,7);F();
		C(0,253,0,7);C(0,8,1,3);C(11,30,0,3);C(7,15,1,6);B();C(2,13,1,3);F();
		C(8,33,0,1);F();
		C(0,281,1,6);C(0,244,0,5);F();
		C(2,55,1,3);C(4,38,0,8);F();
		C(0,262,0,5);F();
		C(0,253,1,2);F();
		C(6,24,1,2);B();C(0,153,0,6);C(22,47,1,3);C(28,66,0,6);F();
		C(2,17,1,1);C(18,19,1,1);F();
		C(0,249,1,2);F();
		B();C(0,8,1,2);C(0,6,0,6);F();
		C(5,31,0,3);C(31,37,0,2);C(18,19,0,8);C(10,245,1,7);F();
		C(99,841,0,4);C(4,20,1,6);F();
		C(2,59,0,3);C(11,36,1,7);C(0,40,0,4);C(25,46,0,2);C(13,20,1,6);F();
		C(2,55,0,1);C(8,96,1,4);F();
		C(0,2,0,1);F();
		C(0,1,0,1);C(39,85,0,2);C(0,17,0,7);F();
		C(0,7,0,4);F();
		B();C(9,125,1,3);B();C(0,15,0,3);F();
		C(0,1,1,6);C(0,5,1,7);C(0,3,1,4);C(0,14,0,5);F();
		C(2,5,1,4);C(57,71,0,5);C(82,264,1,3);C(2,7,1,6);C(0,29,0,1);C(0,1,1,2);F();
		C(7,12,1,1);C(2,701,0,5);C(82,434,1,5);F();
		return 2.694825;
	}

	double compute_indices_23 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,4,0,7);C(0,4,1,7);B();B();F();
		C(1,29,1,3);C(20,35,2,8);B();F();
		C(29,33,1,2);C(6,27,1,4);F();
		C(6,47,1,1);C(5,9,1,7);C(0,3,1,8);F();
		C(2,55,1,6);C(8,39,1,1);F();
		C(0,68,1,8);C(0,8,1,3);F();
		C(0,4,0,4);C(5,103,0,1);C(5,46,1,5);C(0,14,0,2);C(0,2,2,8);F();
		C(4,25,1,2);C(66,139,1,3);B();C(8,18,0,8);F();
		C(0,93,1,8);C(0,1,1,4);C(0,85,1,7);C(66,139,1,3);C(20,24,0,2);C(0,7,0,6);C(23,31,1,2);F();
		C(0,8,0,3);C(0,7,0,1);B();C(0,54,1,3);C(46,147,1,1);F();
		C(0,4,0,4);C(0,6,0,3);C(0,8,0,5);C(0,9,1,5);C(0,14,0,5);C(0,43,1,7);C(72,144,0,1);C(0,6,0,3);C(15,53,0,4);C(2,23,1,2);C(3,14,1,8);C(0,10,1,4);C(2,4,0,2);C(46,77,0,5);C(5,46,0,4);C(0,255,0,1);C(0,22,1,3);C(45,497,0,4);F();
		C(54,74,0,8);C(14,258,0,1);C(0,246,0,5);C(18,21,1,8);F();
		C(3,31,1,4);C(59,711,0,1);F();
		C(22,260,0,1);C(11,46,1,8);C(19,30,1,4);C(11,46,0,3);C(0,32,1,7);F();
		C(123,635,0,1);F();
		C(14,33,1,5);C(0,1,1,8);C(0,1,2,5);C(0,28,0,3);C(26,260,0,1);C(13,27,0,8);B();F();
		C(3,35,1,8);C(0,239,0,7);C(33,161,0,1);C(0,275,1,5);C(2,4,0,1);C(0,262,0,2);C(0,5,1,1);C(0,239,0,7);F();
		C(0,12,0,3);C(9,18,1,1);C(56,346,0,3);C(0,6,1,4);C(56,346,1,3);C(11,30,0,1);C(6,39,0,1);C(13,23,0,4);B();C(0,10,0,8);C(0,4,2,3);B();B();C(56,63,2,8);C(0,233,1,2);F();
		C(82,434,0,5);C(111,147,0,6);C(0,255,0,1);C(8,12,1,4);C(42,62,0,2);C(0,20,1,2);F();
		B();C(0,261,1,4);C(0,255,0,5);C(0,9,0,4);C(76,502,0,2);C(6,21,0,1);C(15,16,1,4);C(0,20,0,4);C(0,12,1,7);C(1,11,0,3);C(7,19,1,2);C(0,7,1,4);C(0,17,1,8);C(14,21,0,2);C(0,255,1,5);C(6,19,1,8);C(1,11,0,3);C(9,27,0,4);C(11,24,1,1);B();C(8,10,1,6);B();C(71,123,0,7);F();
		C(0,5,0,8);C(118,645,0,6);C(0,10,0,4);C(18,24,0,4);C(2,287,0,6);C(53,61,0,4);C(82,434,1,5);C(24,37,1,1);C(40,42,0,1);F();
		C(0,116,0,3);C(59,711,0,2);C(0,60,0,3);C(14,31,0,7);C(43,80,0,4);F();
		C(0,255,0,1);C(0,255,0,1);C(7,18,0,2);C(7,18,0,2);C(59,711,0,2);F();
		C(14,62,1,4);C(14,258,0,1);C(39,42,0,1);C(0,2,1,1);C(6,712,0,1);C(0,255,1,7);F();
		C(15,48,0,1);C(8,718,0,4);C(5,43,0,1);C(53,61,1,5);C(0,255,0,1);C(0,259,1,8);F();
		C(0,38,0,1);B();C(3,246,0,4);C(0,15,1,5);C(10,31,0,7);C(17,26,1,3);C(8,723,0,6);F();
		B();C(9,18,1,5);C(4,728,0,2);F();
		C(37,57,0,5);C(0,4,2,1);C(9,42,0,6);C(120,838,1,6);F();
		C(97,844,0,2);C(10,19,1,8);F();
		C(82,264,1,3);C(5,13,1,2);C(1,8,1,8);C(1,12,1,4);C(0,7,0,7);C(111,646,1,7);C(0,6,1,3);C(97,844,0,3);C(14,259,1,6);C(8,70,1,3);F();
		C(82,264,0,4);C(18,24,0,1);C(2,55,0,7);C(0,2,2,8);C(6,19,1,7);C(0,42,0,5);C(8,163,1,2);C(97,844,0,5);C(5,6,0,4);C(0,208,0,6);F();
		C(0,2,0,3);C(106,872,0,1);C(92,840,1,5);F();
		return 1.755473;
	}

	double compute_indices_24 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,60,0,3);C(0,16,1,8);C(2,20,0,2);F();
		C(6,98,0,3);F();
		C(5,22,0,4);C(0,10,1,3);C(5,22,0,4);F();
		C(19,30,1,6);C(0,2,1,6);C(0,20,0,2);C(0,68,0,4);C(2,6,0,6);C(8,12,1,4);F();
		C(0,5,1,2);C(82,434,1,1);C(0,4,1,1);B();F();
		C(0,113,0,4);F();
		C(10,28,1,7);F();
		C(12,15,1,5);C(99,142,0,2);C(3,8,0,5);C(4,6,0,4);F();
		C(5,47,1,3);C(0,2,1,2);C(59,146,0,3);C(62,64,0,3);F();
		C(0,10,0,8);B();C(0,4,1,5);C(42,86,0,2);C(64,70,0,3);F();
		C(0,15,1,5);B();C(0,11,1,5);F();
		B();C(0,4,0,8);C(1,12,1,4);C(0,11,1,8);B();F();
		C(0,14,0,7);C(0,3,1,4);C(0,1,0,1);F();
		C(97,844,1,3);C(111,646,1,7);C(0,6,1,4);C(14,259,1,6);C(114,657,1,7);F();
		C(102,215,0,5);C(0,47,1,2);C(0,253,1,4);F();
		C(0,17,0,3);C(47,59,2,6);C(2,16,0,4);C(0,36,0,8);B();C(7,22,0,3);C(19,47,0,4);C(0,18,1,3);C(5,9,1,2);C(0,12,0,4);C(18,19,1,8);C(15,25,0,7);C(5,9,1,3);C(23,31,0,4);C(0,8,1,1);F();
		C(0,3,1,3);C(0,4,0,1);C(0,4,1,6);C(0,3,1,5);B();C(0,2,2,8);F();
		C(22,47,1,3);C(0,94,0,7);B();C(21,102,0,6);F();
		C(9,18,0,3);C(0,60,1,2);C(8,163,0,8);C(7,19,0,4);C(11,19,1,5);C(20,23,1,4);C(0,2,1,4);C(8,12,0,1);C(0,28,0,1);C(103,135,1,6);C(0,11,1,8);C(10,14,0,5);F();
		C(56,86,1,1);C(83,272,1,3);F();
		C(1,8,0,8);C(0,3,1,8);F();
		C(11,18,0,2);C(31,37,0,2);C(4,26,0,5);C(31,38,1,3);C(14,17,1,8);F();
		C(7,14,0,2);C(5,46,0,4);C(108,149,0,2);F();
		C(30,164,0,6);C(10,11,1,5);C(48,69,0,5);F();
		C(2,54,1,1);C(116,636,1,3);C(56,346,1,1);C(114,650,1,2);B();F();
		B();C(22,63,0,1);F();
		B();C(39,85,0,2);F();
		C(16,93,0,3);C(9,11,1,1);C(7,21,0,1);F();
		C(8,39,0,7);C(0,5,1,4);C(0,11,1,6);B();C(14,110,0,3);C(0,4,1,2);C(1,12,1,4);C(0,10,1,8);F();
		C(0,8,1,1);F();
		C(15,41,1,2);C(46,147,1,6);C(62,64,0,4);B();C(82,241,0,4);C(62,64,0,5);F();
		C(6,712,1,5);C(4,18,1,1);C(114,650,1,2);C(82,434,1,2);B();C(2,5,2,5);C(0,4,2,3);F();
		return 1.884520;
	}

	double compute_indices_25 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,7,0,4);C(10,11,1,5);C(0,5,1,5);F();
		C(2,7,0,1);C(5,10,1,2);F();
		C(0,28,0,1);C(3,43,1,1);C(3,43,1,1);F();
		C(0,3,0,4);C(0,1,1,1);F();
		C(0,4,0,5);B();C(0,1,0,5);F();
		C(12,24,1,2);F();
		C(0,3,1,8);C(0,8,0,2);C(0,2,2,5);C(10,31,1,1);C(6,27,0,5);C(0,1,1,1);F();
		C(0,43,1,5);F();
		C(0,3,1,6);C(2,5,0,8);F();
		C(12,17,1,5);C(0,18,1,7);C(0,3,1,1);C(2,5,0,4);C(16,18,1,6);F();
		C(8,31,0,5);C(9,42,1,4);F();
		C(0,17,1,5);F();
		C(0,4,0,4);F();
		C(5,16,1,7);C(14,25,0,4);C(0,7,1,1);B();C(5,11,2,6);C(0,5,0,8);F();
		C(0,4,1,5);C(0,5,0,2);C(8,20,1,2);C(8,18,0,3);C(8,20,1,8);F();
		C(2,5,0,8);F();
		C(3,20,1,4);C(6,10,0,5);C(0,26,0,5);C(17,21,0,1);C(0,37,1,6);C(0,6,0,3);F();
		B();C(1,6,1,4);C(24,34,1,5);C(0,3,1,4);F();
		C(8,39,0,1);B();C(7,19,0,2);C(20,26,0,6);C(0,11,1,4);C(13,48,0,6);C(0,3,0,6);F();
		C(4,38,1,3);B();C(2,5,1,6);C(23,31,0,1);F();
		C(39,51,1,7);F();
		C(19,33,0,4);C(14,20,1,1);F();
		C(0,8,1,1);C(34,42,1,2);F();
		C(25,46,1,8);F();
		B();C(7,14,0,1);F();
		C(0,1,1,4);F();
		C(0,8,0,6);B();C(0,15,0,8);C(0,42,0,8);C(5,33,0,3);F();
		C(0,17,0,6);B();C(24,31,0,3);B();C(4,38,0,3);C(2,11,1,5);F();
		C(4,38,1,8);C(0,17,1,6);C(3,9,0,1);C(0,6,0,7);C(0,9,0,8);C(24,37,1,1);C(0,4,0,4);C(0,2,1,1);C(0,9,0,8);C(24,37,1,1);C(0,4,0,4);F();
		C(0,6,1,1);C(0,4,0,3);F();
		C(31,43,1,3);F();
		C(2,20,0,1);C(6,27,0,3);C(0,8,0,6);F();
		return 6.181545;
	}

	double compute_indices_26 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,78,1,2);C(2,20,0,1);C(0,60,0,3);C(65,403,1,2);B();B();B();B();B();F();
		C(0,2,2,5);C(0,246,0,5);C(21,159,0,7);C(3,19,0,6);F();
		C(11,24,1,2);F();
		C(11,20,1,4);F();
		B();C(2,252,0,3);C(7,30,1,4);C(0,11,1,2);C(8,12,1,4);F();
		C(0,14,0,8);C(6,27,1,1);C(6,14,0,4);F();
		C(76,417,1,5);C(0,3,0,5);C(0,22,1,1);C(0,3,0,5);F();
		B();C(4,6,0,4);C(8,39,1,4);C(48,53,1,6);F();
		C(2,9,0,3);C(57,63,2,3);F();
		C(1,11,0,7);C(5,9,1,4);F();
		C(0,15,1,6);C(22,47,1,3);B();F();
		B();C(1,12,1,4);F();
		C(15,268,1,8);F();
		C(8,12,1,4);C(15,16,1,2);C(0,232,1,6);F();
		C(13,23,0,2);F();
		C(8,16,0,5);F();
		B();F();
		B();B();B();C(24,32,1,3);C(0,4,0,2);C(0,4,1,1);C(0,9,1,4);F();
		C(5,13,1,8);C(0,47,0,1);C(0,2,2,8);F();
		C(78,159,0,2);C(2,5,0,4);C(0,42,0,8);B();B();C(14,47,1,3);C(4,10,1,8);F();
		C(0,11,1,1);C(0,7,1,2);C(41,55,1,6);C(6,712,1,5);F();
		B();C(14,21,0,2);F();
		C(9,41,0,1);C(0,2,0,8);C(0,18,0,3);F();
		C(10,264,0,1);C(25,46,1,2);C(0,40,0,4);C(0,40,0,4);C(10,11,1,2);C(39,85,1,2);C(10,28,1,7);C(10,28,1,7);F();
		C(8,39,0,3);F();
		C(0,5,0,5);B();C(2,13,0,6);C(8,163,0,2);F();
		C(11,30,0,4);F();
		C(1,3,1,2);C(0,5,1,8);C(0,1,0,4);B();F();
		C(0,17,1,6);B();B();C(0,116,0,3);C(0,7,0,4);F();
		C(0,59,1,1);C(0,208,1,5);C(0,4,1,7);F();
		C(46,147,1,1);C(62,64,0,5);C(82,264,1,3);C(18,712,1,4);F();
		C(82,434,1,5);C(114,650,1,2);C(6,712,1,5);C(0,153,1,6);F();
		return 2.077974;
	}

	double compute_indices_27 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(10,16,1,6);F();
		B();C(0,3,2,3);F();
		C(17,26,1,3);B();F();
		C(0,1,1,1);B();C(0,1,2,8);C(0,2,0,4);C(2,3,1,5);F();
		C(0,4,0,7);C(0,5,1,2);F();
		C(1,5,2,3);F();
		C(0,3,1,1);C(0,8,1,4);F();
		C(0,8,0,8);C(2,9,1,8);C(3,8,0,5);F();
		C(2,9,1,3);B();F();
		C(14,23,0,6);F();
		B();C(7,11,0,3);C(44,55,2,1);F();
		C(18,24,0,4);F();
		C(0,4,0,1);F();
		C(14,15,1,8);C(9,16,0,3);F();
		C(0,5,0,3);C(0,17,0,2);C(1,11,1,3);C(1,12,0,4);F();
		C(0,6,0,1);C(0,18,1,3);C(0,11,0,6);C(0,16,1,1);C(19,28,0,7);C(8,12,0,4);B();B();F();
		C(19,30,1,2);F();
		C(13,20,1,6);F();
		C(8,12,1,7);F();
		C(0,31,1,6);C(8,12,0,4);C(0,3,0,4);F();
		C(0,1,1,4);C(24,31,0,4);F();
		C(0,8,1,7);C(0,7,1,1);C(4,7,0,1);F();
		B();C(34,42,1,7);B();F();
		B();F();
		C(1,6,0,1);F();
		C(8,15,0,4);F();
		C(0,4,0,1);C(38,46,0,4);C(4,5,1,1);F();
		C(49,63,2,3);C(11,15,0,3);F();
		C(7,10,0,2);C(0,8,0,4);C(0,3,0,2);F();
		C(7,40,0,4);F();
		C(0,11,0,3);C(0,15,1,1);C(0,6,0,6);C(0,7,1,6);C(0,2,1,5);C(0,6,1,5);C(4,9,0,4);F();
		C(0,40,0,3);C(0,11,0,1);C(4,6,0,5);C(0,9,1,2);F();
		return 1.178962;
	}

	double compute_indices_28 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(14,105,0,6);F();
		C(0,153,1,6);C(0,153,1,6);F();
		C(0,65,0,2);F();
		C(10,22,0,7);C(0,232,1,6);C(6,29,0,6);F();
		C(32,107,1,1);F();
		C(111,144,1,6);F();
		B();C(0,8,1,3);C(31,75,1,8);F();
		C(0,8,1,4);C(99,142,0,2);C(4,6,0,4);B();F();
		C(2,9,0,1);C(45,355,0,2);C(18,24,1,1);C(0,17,1,6);F();
		C(48,63,2,5);C(4,253,1,8);F();
		C(9,12,0,5);B();C(41,53,1,3);C(0,208,1,6);C(7,19,0,8);F();
		C(22,63,1,3);C(56,86,1,5);C(2,4,1,2);F();
		C(6,10,1,3);C(8,163,1,2);F();
		C(0,232,1,6);F();
		C(21,23,0,6);C(0,4,0,2);C(51,174,1,7);C(36,42,1,7);F();
		C(0,6,1,4);C(20,108,0,3);F();
		C(0,2,2,5);F();
		C(53,61,0,8);C(2,13,1,8);C(0,5,1,7);F();
		C(4,38,1,1);C(56,86,1,5);C(5,13,1,4);C(0,3,0,5);C(0,4,1,7);C(1,5,0,4);C(8,18,0,6);C(1,11,1,2);F();
		C(10,28,1,7);C(0,12,0,6);F();
		C(0,5,0,5);F();
		C(10,57,0,2);C(0,28,0,1);C(0,255,0,3);C(0,275,1,2);C(45,351,1,2);C(0,21,1,1);F();
		C(0,17,0,1);C(3,24,1,5);C(0,14,1,5);C(45,52,1,1);C(6,14,0,4);C(8,31,1,3);C(9,42,1,7);C(13,17,1,5);C(14,27,1,7);C(23,28,1,2);C(0,14,0,5);F();
		C(3,19,0,2);C(0,42,0,5);C(0,7,1,4);F();
		C(2,12,1,4);F();
		B();C(2,4,1,2);C(22,63,0,1);C(0,6,0,7);F();
		B();B();C(39,85,0,2);C(0,3,0,8);C(24,31,1,6);C(0,1,0,7);C(1,5,0,4);F();
		C(73,82,1,7);C(15,53,0,4);F();
		C(101,136,1,8);B();C(4,6,0,4);B();F();
		C(64,101,1,7);F();
		C(6,712,1,5);F();
		C(0,147,1,6);C(29,715,1,8);C(0,255,0,2);F();
		return 3.599668;
	}

	double compute_indices_29 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(15,38,0,5);C(0,16,0,8);C(3,8,1,2);C(2,20,0,1);C(8,15,0,4);B();F();
		C(0,4,0,8);C(23,164,1,6);C(16,27,1,7);C(9,221,1,6);F();
		C(0,43,1,7);C(15,53,0,4);C(6,26,0,6);B();F();
		C(0,4,2,3);C(2,5,2,5);F();
		C(21,102,0,1);C(29,30,0,8);C(7,14,0,1);C(53,61,0,4);C(0,17,0,1);C(14,21,0,6);F();
		C(0,28,0,8);C(25,46,2,4);C(0,2,2,3);C(0,40,0,4);C(0,16,1,6);C(25,46,2,5);C(22,31,0,8);C(8,163,1,3);C(31,55,2,2);F();
		C(2,4,0,2);C(0,18,0,2);C(0,14,1,6);C(0,11,0,2);C(0,34,0,6);C(2,6,0,1);F();
		C(90,153,0,2);C(0,8,1,1);C(0,44,0,3);C(2,9,1,4);C(1,6,1,4);F();
		C(0,2,1,6);F();
		B();C(6,9,0,8);C(2,6,0,1);F();
		C(5,9,1,6);B();C(0,17,1,1);C(13,58,0,4);C(2,13,1,8);C(8,12,0,4);C(4,38,1,8);F();
		C(0,47,0,2);C(34,42,1,5);C(0,60,0,3);F();
		C(5,46,0,7);C(8,96,0,4);F();
		C(0,44,0,1);F();
		C(0,89,0,3);C(5,46,0,7);C(0,68,0,4);C(0,1,2,5);C(31,70,0,6);F();
		B();C(2,15,1,4);C(0,17,1,8);C(2,9,0,2);C(19,112,0,1);C(2,55,1,1);C(11,20,1,5);F();
		C(0,27,0,3);C(18,24,0,5);C(3,37,0,4);C(0,1,0,6);B();F();
		B();B();C(0,19,0,8);C(66,139,0,3);C(7,41,0,7);C(0,10,0,1);C(0,6,0,2);C(21,27,1,5);C(2,55,0,1);C(0,4,0,3);C(21,102,0,1);C(0,2,0,4);C(8,39,1,1);C(0,16,1,3);F();
		C(5,13,0,4);C(2,9,1,7);C(6,27,0,6);C(101,141,0,2);C(0,17,1,5);F();
		C(13,16,1,6);C(45,136,0,3);C(0,47,0,1);C(33,62,0,4);F();
		C(57,67,1,8);C(8,163,1,2);C(31,70,0,3);C(0,8,0,8);F();
		C(43,154,1,1);C(6,12,1,6);F();
		C(0,23,1,4);C(0,6,0,3);C(1,19,0,5);F();
		C(38,65,1,4);C(2,55,0,1);F();
		C(0,20,0,1);C(0,116,0,1);F();
		C(56,86,0,5);B();C(7,19,0,2);C(22,50,0,1);C(1,2,0,8);C(0,23,0,6);C(11,57,0,3);C(37,49,1,5);F();
		C(39,85,0,2);C(0,4,0,3);B();C(2,9,1,4);C(3,39,0,8);F();
		C(0,4,1,1);B();F();
		B();C(0,116,0,3);B();C(4,38,0,4);C(14,83,1,3);F();
		C(5,36,0,2);C(3,72,1,2);B();C(0,16,1,8);C(21,47,0,5);C(15,25,0,4);F();
		B();C(39,146,0,1);C(62,64,0,5);C(19,33,0,8);F();
		C(46,147,1,1);C(2,9,1,6);C(3,8,0,5);F();
		return 1.050831;
	}

	double compute_indices_30 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(44,103,1,8);F();
		C(0,7,0,4);C(1,5,0,3);F();
		C(0,6,2,7);C(0,5,0,1);F();
		C(0,2,0,4);C(0,2,1,1);B();C(2,5,2,3);F();
		C(0,4,0,8);C(0,4,1,1);C(49,57,1,5);C(1,6,1,2);C(0,5,1,2);F();
		C(2,9,0,2);C(1,12,1,4);C(0,1,1,1);C(0,8,0,8);F();
		C(0,17,1,1);F();
		C(23,31,1,1);F();
		C(8,112,0,6);F();
		C(9,23,1,5);C(0,91,0,8);F();
		C(95,137,1,6);F();
		C(112,130,0,2);C(0,3,1,1);F();
		C(0,13,1,1);C(2,4,2,1);C(0,26,1,1);F();
		C(8,96,1,4);C(0,3,0,5);C(7,41,1,3);F();
		C(0,4,1,4);C(0,36,0,2);F();
		C(0,17,0,7);F();
		C(0,13,0,3);C(0,6,1,1);F();
		C(36,48,0,7);C(3,13,0,4);C(76,417,1,5);C(0,4,0,1);C(0,59,0,1);C(0,9,0,2);F();
		C(0,12,1,2);C(0,37,1,3);F();
		C(23,31,0,1);B();C(29,53,1,3);C(0,46,0,8);C(0,10,1,1);F();
		C(4,38,1,1);C(6,52,1,8);C(42,62,0,2);C(0,18,1,5);C(61,78,0,8);C(41,53,1,5);B();B();C(0,2,0,2);F();
		C(0,255,0,8);C(0,7,0,1);C(37,51,1,3);C(2,9,0,1);C(31,37,0,7);C(0,18,1,1);C(0,18,1,1);F();
		C(0,13,0,4);C(0,42,0,7);F();
		C(25,46,1,7);F();
		C(0,19,0,4);C(0,19,0,4);C(56,346,1,3);C(14,258,0,1);F();
		C(1,41,1,6);F();
		C(0,68,1,4);F();
		C(12,18,0,2);C(7,8,0,6);C(0,60,0,3);B();C(13,23,1,2);C(4,5,1,8);F();
		B();B();B();B();C(1,24,1,5);C(8,12,1,2);C(0,17,1,6);C(23,31,0,3);F();
		C(0,208,0,5);C(0,10,1,5);C(0,4,1,8);F();
		C(15,25,1,3);F();
		C(11,20,1,5);F();
		return 3.809187;
	}

	double compute_indices_31 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(3,8,0,3);F();
		C(2,27,1,4);F();
		C(0,60,0,1);C(8,13,1,4);C(0,2,1,1);C(0,40,0,7);C(0,4,0,2);C(0,6,1,6);C(0,11,1,1);F();
		C(3,54,0,8);C(53,61,0,8);F();
		C(7,31,0,4);C(3,8,0,3);C(0,18,0,3);F();
		B();C(22,63,1,5);C(0,17,1,1);C(0,2,1,1);F();
		C(0,8,1,7);C(0,3,1,6);C(5,46,0,4);F();
		C(0,1,1,3);C(0,29,0,6);C(6,39,0,1);F();
		C(23,31,0,1);C(0,9,1,2);C(22,28,1,1);F();
		C(1,11,0,7);C(20,28,0,4);F();
		C(16,25,0,6);C(0,3,0,4);B();F();
		C(16,18,1,6);C(0,58,1,3);C(0,23,1,3);F();
		C(0,14,0,5);C(0,31,1,2);C(11,25,1,5);C(0,21,1,5);C(2,6,1,2);F();
		C(5,36,0,2);F();
		C(0,10,0,8);B();C(2,36,1,2);F();
		C(0,6,0,8);C(20,26,0,7);C(0,7,1,1);B();C(0,31,0,4);C(0,7,0,7);F();
		C(18,24,0,7);C(0,3,1,3);C(0,6,0,1);C(0,6,0,1);F();
		C(0,1,0,3);C(17,26,0,2);C(0,4,1,3);C(0,4,1,8);F();
		B();C(0,14,0,5);C(8,23,0,2);C(2,13,0,4);C(2,8,0,4);C(6,29,0,2);C(6,11,1,4);C(5,33,1,3);F();
		C(2,5,0,4);B();C(6,9,1,8);C(0,8,0,3);F();
		C(16,24,0,6);C(24,31,0,6);C(0,8,0,1);F();
		C(3,15,1,1);C(0,32,0,7);C(0,1,1,4);C(0,36,0,7);F();
		C(7,25,1,3);C(9,13,0,3);C(34,42,1,2);F();
		C(0,5,1,2);C(0,4,0,5);F();
		C(20,30,1,4);C(0,5,1,2);C(0,4,0,5);C(30,164,0,6);C(48,69,0,5);F();
		C(8,18,0,2);C(56,76,1,5);F();
		C(21,47,0,5);C(15,25,0,4);F();
		C(27,51,1,4);C(6,40,0,5);C(17,26,1,3);F();
		C(21,27,0,4);C(0,4,2,3);C(11,30,0,4);C(10,19,0,4);C(0,23,0,3);F();
		C(2,9,0,1);F();
		C(0,9,1,2);C(0,32,1,7);F();
		C(20,22,1,2);F();
		return 2.183570;
	}

	double compute_indices_32 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,36,1,7);C(15,16,1,4);C(0,11,0,3);C(42,62,1,2);C(88,411,1,5);C(2,6,0,2);C(7,222,1,6);C(51,174,1,7);F();
		C(55,725,1,2);C(27,78,1,4);C(90,395,1,5);C(2,5,1,4);C(14,25,0,4);C(56,86,1,7);F();
		C(11,17,1,1);C(0,16,0,5);C(0,2,1,8);F();
		C(6,10,0,5);C(0,1,1,4);F();
		C(8,718,1,4);B();C(1,6,1,2);C(7,19,0,2);C(57,138,1,2);C(121,785,1,7);F();
		C(12,18,0,1);C(3,214,1,2);C(115,640,1,8);C(0,2,1,1);C(68,425,1,1);C(2,55,1,3);C(7,18,1,8);C(98,136,1,2);C(54,704,1,2);B();F();
		C(9,221,1,6);C(40,42,0,2);C(22,35,0,2);C(105,851,1,3);C(57,503,1,2);C(6,19,1,8);C(11,240,1,6);C(0,4,1,7);C(0,12,1,1);F();
		C(5,6,1,1);C(18,25,0,4);C(27,78,1,2);C(82,415,1,8);C(97,844,1,1);C(2,5,1,4);F();
		C(47,715,1,6);C(141,639,1,5);C(62,64,0,3);C(91,431,1,1);F();
		C(6,11,0,4);F();
		C(7,17,0,7);C(11,30,0,4);C(13,36,1,7);F();
		C(0,153,1,6);C(0,8,0,8);C(0,4,1,5);C(82,434,1,5);C(8,26,1,7);C(114,650,1,2);C(6,712,1,5);C(8,718,1,4);F();
		C(76,435,1,8);C(80,284,1,7);C(22,174,1,8);C(8,10,1,8);C(84,830,1,4);C(16,19,0,2);F();
		C(6,21,0,5);C(0,26,1,7);B();C(1,12,1,4);C(114,657,1,7);C(9,17,0,1);C(0,3,0,3);C(8,15,0,7);C(4,86,1,4);F();
		C(0,6,1,3);C(0,4,1,5);C(7,10,0,4);C(1,264,1,7);C(4,86,1,3);C(68,808,1,6);C(27,38,1,1);C(76,417,1,1);C(11,729,1,8);C(97,844,1,8);C(14,259,1,6);C(102,653,1,7);C(0,1,1,1);C(57,503,1,4);F();
		C(11,246,1,7);B();C(19,30,0,2);C(8,163,1,3);C(0,39,1,6);C(95,837,1,6);F();
		B();C(0,8,0,3);C(18,24,0,4);C(4,6,1,7);C(34,42,1,2);B();F();
		C(0,4,0,5);C(0,1,0,3);F();
		C(10,28,0,5);C(0,5,1,4);C(82,264,1,3);C(1,12,1,3);F();
		B();C(23,31,0,3);C(17,751,1,7);C(76,417,1,1);C(57,503,1,2);F();
		B();C(31,37,0,2);F();
		C(0,42,0,8);C(5,43,0,3);F();
		C(5,8,0,1);C(0,8,1,4);C(123,646,1,5);C(9,266,1,8);C(45,351,1,2);C(82,434,1,4);C(0,11,1,4);F();
		C(0,32,0,4);C(18,19,1,8);C(16,18,1,1);F();
		C(115,660,1,6);C(7,11,0,4);C(20,108,1,3);C(76,417,1,5);C(0,7,0,4);B();C(2,9,0,1);B();C(0,255,1,2);C(9,221,1,6);C(56,346,1,1);C(114,650,1,2);B();F();
		B();C(51,60,0,8);C(76,417,1,1);F();
		C(5,13,0,3);C(103,852,1,2);C(76,417,1,5);C(45,351,1,2);F();
		C(106,872,1,1);C(0,24,0,5);C(37,57,1,1);F();
		C(0,257,1,4);C(13,703,1,3);C(20,72,1,8);C(105,851,1,3);C(0,44,1,7);C(8,163,1,8);C(0,21,1,8);C(2,6,0,6);F();
		C(49,51,0,1);C(14,21,0,2);C(24,31,1,6);B();F();
		C(6,11,0,7);B();C(82,264,1,3);C(62,64,0,5);C(4,33,1,2);F();
		C(13,14,1,5);C(1,225,1,3);C(82,434,1,5);C(0,18,1,3);C(18,22,0,5);C(8,718,1,4);C(10,11,1,2);C(46,147,1,1);C(0,32,1,7);C(8,12,0,5);C(18,25,0,4);F();
		return 1.416208;
	}

	double compute_indices_33 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(31,38,1,3);F();
		C(0,2,2,5);F();
		B();C(0,1,1,8);C(39,63,2,2);C(7,47,1,8);F();
		C(0,10,1,4);B();F();
		B();C(0,4,1,1);F();
		C(0,6,0,5);C(0,6,0,2);F();
		C(0,6,0,8);C(0,8,1,7);C(1,5,2,6);F();
		C(0,20,1,3);C(4,38,1,4);F();
		C(0,3,0,2);C(2,9,0,1);C(1,7,1,3);F();
		C(5,9,1,4);C(0,10,0,3);F();
		C(19,65,1,3);F();
		B();C(1,12,1,4);F();
		C(0,14,1,3);C(8,20,1,5);F();
		C(0,7,0,1);C(8,12,1,4);C(6,13,0,1);C(3,8,0,3);C(0,6,1,5);F();
		C(13,27,1,1);C(0,14,1,2);F();
		C(9,18,1,4);F();
		C(0,10,1,1);C(0,7,0,4);F();
		C(0,23,1,3);F();
		B();C(0,4,0,3);F();
		C(18,24,1,4);F();
		B();C(24,31,1,8);C(10,28,1,7);F();
		C(0,18,0,2);F();
		C(22,40,0,2);F();
		C(20,64,1,3);F();
		C(17,26,0,6);F();
		C(41,50,1,2);F();
		C(37,72,1,4);F();
		C(0,16,1,1);C(0,17,1,4);F();
		C(0,17,1,6);C(10,15,0,6);C(0,6,1,2);B();C(0,4,0,8);F();
		C(7,41,0,7);F();
		C(0,2,1,1);C(1,15,1,8);F();
		C(0,2,2,6);F();
		return 2.316805;
	}

	double compute_indices_34 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(2,20,0,1);C(0,16,0,8);C(29,33,1,4);C(1,12,0,1);B();F();
		C(0,7,0,2);C(0,21,1,4);C(0,2,2,5);C(61,78,0,1);C(0,47,0,1);F();
		C(0,2,0,1);C(8,16,1,5);C(0,275,0,5);F();
		C(0,2,0,4);B();C(0,18,0,4);C(0,4,1,4);F();
		B();C(0,32,1,7);C(22,47,1,3);F();
		C(1,12,2,6);C(0,6,0,8);C(18,24,0,3);B();C(11,30,0,3);F();
		C(0,10,0,3);F();
		B();C(108,141,0,6);F();
		C(0,8,1,1);F();
		C(10,11,1,5);C(14,105,0,6);C(22,47,1,3);F();
		C(0,255,0,1);C(9,10,0,8);C(51,70,0,2);C(0,7,0,2);C(0,2,0,8);B();C(29,33,1,7);F();
		B();F();
		C(8,20,0,4);C(2,4,0,2);C(3,8,0,5);C(0,6,0,8);F();
		C(15,16,1,4);C(8,12,1,4);C(5,69,0,1);C(0,13,1,6);C(0,37,1,1);C(0,256,0,4);C(12,24,1,2);F();
		C(0,4,1,5);C(15,43,0,4);F();
		C(0,40,1,6);C(5,24,0,2);F();
		C(0,10,1,1);C(8,718,1,4);F();
		C(3,8,0,5);C(4,6,0,4);C(6,28,0,4);C(61,78,0,3);C(0,47,0,1);F();
		C(5,13,1,4);C(6,27,0,4);C(3,69,0,5);F();
		C(67,264,1,3);C(3,15,1,1);C(0,60,1,4);C(0,65,1,6);C(57,63,2,3);F();
		C(15,70,0,7);B();C(5,28,0,6);C(15,70,0,7);F();
		C(17,29,0,3);C(0,15,1,2);C(0,1,1,2);F();
		C(9,22,1,2);C(11,20,0,1);C(3,89,0,8);C(60,69,1,8);F();
		C(25,46,1,2);C(0,40,0,4);C(0,251,1,2);F();
		C(6,14,0,4);C(0,47,0,1);F();
		C(0,58,0,2);C(31,37,0,2);C(0,10,0,8);C(3,8,1,3);F();
		C(39,85,0,2);B();C(15,25,0,4);F();
		C(0,10,1,4);C(10,22,0,3);F();
		C(1,3,1,2);C(8,15,1,4);C(0,116,0,3);C(3,6,1,3);C(11,30,0,4);F();
		C(61,78,0,3);C(0,31,0,5);C(0,208,1,5);C(7,8,1,4);C(1,6,1,2);C(0,4,1,7);C(10,31,1,1);F();
		B();C(77,264,0,3);C(0,12,1,7);C(46,147,1,1);F();
		C(0,36,0,7);F();
		return 2.329719;
	}

	double compute_indices_35 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,5,0,4);F();
		C(3,4,1,6);C(0,14,0,5);C(5,36,1,8);C(5,36,0,2);F();
		C(0,2,0,1);C(2,55,0,1);C(0,3,1,3);C(9,16,0,3);F();
		C(0,1,1,1);B();C(0,4,1,4);C(82,434,1,4);C(0,6,1,7);C(0,4,1,3);C(19,30,1,4);C(56,86,1,5);F();
		C(0,13,1,5);C(9,22,0,3);C(20,246,1,6);C(0,7,1,4);C(10,71,1,2);C(64,267,1,4);F();
		C(76,417,1,1);C(0,28,0,8);F();
		C(2,55,0,8);C(21,102,0,1);C(0,3,1,8);C(4,11,1,5);F();
		C(0,12,0,5);F();
		C(2,9,0,2);F();
		C(22,63,0,1);C(0,18,1,3);C(10,26,1,1);F();
		C(14,21,0,2);C(0,12,0,6);F();
		B();C(21,102,0,1);F();
		C(0,14,0,5);C(20,246,1,6);C(11,22,1,7);C(22,31,0,4);C(0,208,1,6);F();
		C(37,57,0,6);F();
		C(8,20,1,5);C(0,6,0,5);C(10,11,1,2);F();
		B();C(1,2,0,6);C(0,18,1,8);C(0,7,1,4);B();F();
		B();C(18,24,0,5);C(0,3,1,1);C(0,2,2,4);F();
		C(0,32,1,8);F();
		C(0,2,0,2);C(6,27,1,4);C(7,19,0,6);C(0,1,0,4);F();
		C(0,9,2,4);F();
		B();C(24,31,0,6);C(2,8,0,4);F();
		C(2,36,1,5);C(48,63,1,5);C(20,22,2,8);C(120,154,1,6);F();
		C(0,3,0,3);C(15,164,1,8);C(6,21,0,5);C(8,15,1,7);C(17,22,1,2);B();C(9,17,0,1);C(1,12,1,4);C(114,657,1,7);F();
		C(30,90,1,2);C(0,40,0,4);C(3,8,0,4);B();C(11,46,0,8);C(0,1,0,5);C(8,20,1,1);C(1,7,1,4);C(0,27,0,8);C(14,21,0,2);C(0,14,1,4);F();
		C(0,14,1,3);C(0,14,1,3);F();
		C(22,63,0,1);C(56,86,0,5);C(99,142,0,2);B();F();
		C(0,60,0,3);C(0,14,1,6);C(0,17,0,8);C(39,85,0,2);C(2,9,1,4);F();
		C(0,2,1,3);B();F();
		C(0,116,0,1);B();F();
		C(0,54,0,3);C(0,16,1,8);C(6,712,1,5);F();
		C(2,13,1,6);C(60,142,1,4);B();C(58,80,0,4);C(82,264,0,4);C(0,4,0,3);C(0,4,0,6);F();
		C(11,30,1,4);C(11,30,0,4);F();
		return 3.057748;
	}

	double compute_indices_36 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,17,0,5);C(2,25,1,3);C(13,22,0,1);C(0,13,1,4);C(12,18,1,2);C(3,6,1,1);C(8,39,1,1);C(4,5,0,8);F();
		C(8,20,0,4);C(2,4,0,2);C(3,8,0,5);C(0,6,0,8);F();
		C(2,9,1,4);C(0,8,0,1);F();
		C(103,135,1,6);F();
		C(2,5,2,3);C(8,39,0,4);F();
		C(0,2,1,5);C(0,6,0,5);C(0,1,0,1);C(57,68,1,7);F();
		C(0,5,1,8);C(0,4,0,6);C(0,12,1,5);F();
		B();C(99,107,1,1);C(4,6,0,4);F();
		C(1,21,1,3);F();
		C(2,55,0,1);F();
		C(7,19,0,2);C(0,12,1,2);C(11,30,1,1);B();C(0,16,0,4);C(4,10,1,8);C(8,26,1,1);F();
		C(111,144,1,6);F();
		C(7,13,0,6);C(0,16,1,1);C(4,7,1,4);F();
		C(64,70,0,3);F();
		C(53,61,0,4);F();
		C(63,74,1,5);F();
		C(73,82,1,7);F();
		C(69,81,1,2);F();
		C(5,13,0,3);C(6,27,0,6);C(1,11,0,1);F();
		C(0,14,0,4);C(23,31,0,6);B();C(0,11,1,4);C(0,47,0,1);F();
		C(0,7,0,8);B();C(2,9,0,2);C(2,9,1,4);F();
		C(24,31,1,6);C(57,63,2,3);F();
		C(0,2,0,3);C(0,7,1,1);F();
		C(0,40,0,4);C(0,40,0,2);C(0,3,1,8);C(2,55,1,6);C(14,21,0,3);C(0,15,0,8);F();
		C(62,91,1,3);F();
		C(31,64,1,5);F();
		C(14,36,0,2);C(9,12,0,4);C(1,7,0,5);F();
		C(0,8,0,1);C(2,13,0,1);C(14,30,0,1);C(3,9,1,4);C(9,14,0,1);F();
		C(44,103,1,8);F();
		C(0,37,1,3);C(0,22,1,2);F();
		C(27,41,1,1);C(0,30,1,7);F();
		C(2,8,1,8);C(34,42,0,2);C(36,48,0,6);C(5,13,1,4);C(1,24,0,2);F();
		return 3.913464;
	}

	double compute_indices_37 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(1,6,1,4);C(4,5,0,8);F();
		C(0,10,0,2);C(6,8,1,2);F();
		C(0,15,1,1);B();C(0,4,1,1);C(1,12,1,7);C(0,3,1,7);C(0,11,1,2);C(3,8,1,2);F();
		C(0,15,0,6);C(0,10,1,5);F();
		C(0,18,0,1);F();
		C(0,23,0,2);F();
		C(0,23,1,1);F();
		C(5,13,1,4);C(6,27,0,4);C(0,13,1,4);C(0,3,0,1);F();
		C(29,33,1,5);B();C(22,31,0,7);C(0,5,1,8);F();
		C(4,19,0,3);C(25,46,1,2);C(0,40,0,4);C(0,6,1,7);C(0,40,1,4);C(11,44,1,5);F();
		B();C(4,26,0,5);C(2,9,0,2);C(36,48,0,1);C(0,11,1,1);F();
		C(2,9,1,4);C(32,55,0,2);F();
		C(22,58,0,3);F();
		C(0,6,1,7);C(22,63,0,1);F();
		C(5,69,0,4);B();C(0,4,1,3);C(2,4,1,2);C(22,63,0,1);F();
		B();C(31,37,0,7);C(31,38,2,8);C(0,11,0,2);C(8,20,0,7);C(0,9,0,4);C(3,67,0,5);F();
		C(0,5,1,2);B();C(48,69,0,1);C(11,44,2,3);C(0,8,0,8);F();
		C(42,62,0,2);C(0,28,0,1);C(61,78,0,8);C(53,61,1,8);C(4,38,1,1);C(6,52,0,8);C(0,18,1,5);C(41,53,1,1);C(0,1,0,3);C(0,5,1,3);F();
		C(66,79,0,4);C(0,16,0,5);C(0,17,1,4);C(34,42,1,7);C(22,28,1,7);F();
		C(2,55,0,1);C(0,5,1,5);C(0,14,0,3);C(35,81,1,2);F();
		C(39,85,0,2);F();
		C(0,2,1,4);C(6,19,1,2);C(24,32,1,4);C(3,89,0,8);C(16,20,1,1);C(0,1,1,1);F();
		C(46,77,1,5);C(18,36,1,6);C(6,98,0,3);C(0,9,0,4);C(2,6,1,8);C(19,47,0,4);C(9,25,1,8);F();
		C(17,26,1,3);C(1,6,0,7);C(0,102,0,3);C(0,18,0,7);C(0,6,1,1);B();F();
		C(25,46,1,8);C(0,2,0,2);B();C(0,8,0,3);C(0,116,0,3);C(27,71,1,8);C(0,15,0,5);F();
		B();C(0,24,1,2);C(53,61,0,8);C(4,26,1,1);C(112,130,0,2);F();
		C(33,90,0,2);F();
		C(4,6,1,4);C(3,23,1,1);C(99,142,0,3);F();
		C(46,147,0,6);C(56,86,0,5);C(7,14,1,3);C(7,14,1,1);C(4,86,1,1);C(31,75,1,2);F();
		C(46,147,1,1);C(7,41,1,7);C(0,5,1,7);C(0,3,1,4);C(0,14,0,5);C(3,7,1,3);F();
		C(7,9,1,7);C(46,147,1,1);C(29,33,1,4);F();
		C(19,65,0,3);F();
		return 2.536136;
	}

	double compute_indices_38 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(14,21,0,3);F();
		C(71,884,1,2);F();
		B();C(0,10,0,1);F();
		C(0,3,0,4);C(0,11,1,3);C(0,4,1,4);F();
		C(51,134,0,3);F();
		C(23,31,0,4);F();
		C(24,31,0,6);C(2,8,0,4);C(59,146,0,3);F();
		C(6,33,1,6);C(7,19,0,2);C(5,13,1,6);C(0,11,1,6);C(12,17,1,8);F();
		C(5,58,0,7);C(0,6,0,3);B();F();
		C(3,69,0,5);C(2,6,1,1);F();
		C(0,16,1,8);C(0,36,0,7);C(3,89,0,8);F();
		C(21,102,0,1);C(0,73,1,8);F();
		C(99,142,0,2);C(34,42,0,8);C(39,85,0,2);C(0,8,0,1);C(0,44,1,8);C(0,8,1,2);F();
		C(2,20,0,1);C(0,8,0,5);C(4,144,1,7);C(9,16,1,3);C(2,8,0,3);F();
		B();C(62,64,0,5);C(46,147,0,1);F();
		B();C(100,162,0,3);C(52,95,0,5);B();C(6,14,0,2);F();
		C(30,33,0,1);C(6,192,0,5);F();
		C(64,70,0,3);C(4,201,1,6);C(0,125,0,3);F();
		C(2,191,0,6);F();
		C(34,44,1,2);C(0,17,0,1);C(0,255,0,2);C(0,255,0,1);F();
		C(8,21,0,1);C(14,258,0,1);F();
		C(14,258,0,1);F();
		C(22,31,0,4);C(12,24,0,2);C(0,38,1,6);C(5,43,0,1);B();C(82,264,0,5);F();
		C(5,99,0,3);F();
		B();C(18,106,0,3);C(99,142,1,2);C(0,3,1,6);C(0,275,0,5);C(9,57,1,3);F();
		C(11,240,1,8);C(0,15,1,6);C(8,302,0,6);C(0,14,1,8);F();
		C(0,5,2,2);C(0,6,0,7);C(45,351,0,2);F();
		C(0,116,0,3);C(59,711,0,2);C(0,60,0,3);F();
		C(59,711,1,2);C(2,6,0,7);C(67,264,1,3);F();
		C(0,54,0,3);C(0,9,1,2);C(6,712,1,5);C(0,54,0,3);F();
		C(2,718,1,2);C(0,8,1,1);C(16,40,0,8);F();
		C(8,718,0,2);F();
		return 2.734156;
	}

	double compute_indices_39 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,21,1,7);C(0,16,1,7);C(0,6,0,7);C(9,12,0,8);C(5,22,0,4);C(0,29,1,4);C(0,12,2,5);F();
		C(99,142,0,2);C(2,25,0,3);C(0,7,1,4);F();
		C(0,6,1,1);C(0,94,0,7);C(5,33,1,3);C(0,4,0,1);C(6,28,1,4);F();
		C(60,365,1,3);F();
		C(0,10,0,6);C(0,17,0,4);C(5,9,1,6);C(0,42,0,5);C(5,10,1,2);B();F();
		C(5,52,1,2);F();
		C(31,70,0,3);C(0,14,1,4);F();
		B();C(5,19,2,4);C(99,142,0,2);C(4,26,1,4);F();
		C(0,18,0,5);C(2,9,0,2);C(0,10,1,1);C(5,46,1,7);F();
		C(0,42,1,1);F();
		C(42,62,0,2);C(0,18,1,5);C(61,78,0,8);C(41,53,1,1);C(0,69,1,3);F();
		C(5,25,0,2);F();
		C(0,1,0,5);C(3,8,0,5);C(5,13,1,4);C(7,19,0,2);F();
		C(0,60,0,3);C(56,346,1,3);C(0,15,1,8);C(20,41,1,1);F();
		C(8,20,1,2);C(37,57,1,8);C(3,69,1,3);C(0,9,1,5);F();
		C(10,24,0,2);C(42,51,1,8);C(9,25,1,8);C(0,2,1,2);B();C(36,52,0,3);C(0,17,1,8);B();C(99,142,0,2);C(3,9,1,3);C(0,42,0,8);C(0,6,1,4);F();
		C(8,29,0,4);C(0,14,0,1);C(0,18,0,3);C(0,58,1,2);F();
		C(2,8,0,3);F();
		C(0,5,0,4);F();
		C(23,31,0,1);C(2,20,1,1);C(0,2,1,2);C(1,6,0,1);C(37,57,1,5);C(2,5,0,4);C(0,10,1,4);F();
		C(50,155,0,5);C(37,72,0,4);C(24,31,1,6);F();
		C(0,4,0,1);F();
		C(11,25,0,1);C(82,434,1,5);C(2,3,0,1);F();
		C(0,4,1,1);B();C(0,40,0,4);C(0,13,1,8);C(0,4,1,4);F();
		C(2,55,0,1);C(18,106,0,3);C(0,89,1,3);F();
		C(52,79,1,1);C(22,63,0,1);C(2,4,1,2);C(0,12,2,5);F();
		C(0,2,0,4);C(21,102,0,1);C(0,16,1,3);F();
		C(0,3,1,1);F();
		C(0,34,0,5);B();C(0,116,0,2);C(0,15,1,7);F();
		C(8,101,0,3);F();
		C(0,1,0,3);F();
		C(0,208,1,5);C(0,42,1,1);C(0,28,0,1);C(1,11,0,1);F();
		return 2.788430;
	}

	double compute_indices_40 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(0,60,0,3);C(2,20,0,1);C(0,16,1,8);F();
		B();C(0,2,2,5);F();
		C(0,2,0,1);F();
		C(0,2,0,4);B();C(0,1,1,1);C(0,4,1,4);F();
		C(0,5,1,2);C(0,4,0,5);F();
		B();B();C(0,6,0,8);F();
		C(0,3,1,8);C(0,8,1,2);F();
		C(3,8,0,5);B();C(99,142,0,2);C(4,6,0,4);F();
		B();C(2,9,0,2);C(2,9,1,4);F();
		C(5,9,1,4);C(1,11,0,7);F();
		C(0,15,1,6);B();F();
		B();C(1,12,1,4);F();
		C(0,14,0,5);F();
		C(9,16,0,3);C(8,12,1,4);C(15,16,1,4);F();
		C(8,20,1,2);C(0,4,1,5);F();
		B();C(0,17,0,8);B();C(14,21,0,2);C(0,7,1,4);B();C(0,18,1,3);F();
		C(18,24,0,4);C(0,3,1,1);F();
		C(1,6,1,4);B();C(17,26,1,3);B();C(0,4,1,1);B();F();
		C(5,13,1,4);C(6,27,0,4);C(7,19,0,2);F();
		C(23,31,0,1);C(2,5,0,4);B();F();
		B();C(24,31,1,6);F();
		B();C(31,37,0,2);C(31,38,1,3);F();
		C(34,42,1,2);B();F();
		C(25,46,1,2);C(0,40,0,4);C(0,19,0,4);F();
		C(8,39,1,4);C(2,55,0,1);F();
		C(56,86,1,5);C(2,4,1,2);C(22,63,0,1);F();
		B();C(39,85,0,2);F();
		B();F();
		B();C(0,17,1,6);B();B();C(0,116,0,3);F();
		C(0,208,1,5);C(0,14,1,4);C(1,6,1,2);C(0,4,1,7);F();
		C(82,264,1,3);C(46,147,1,1);B();C(62,64,0,5);F();
		C(114,650,1,2);C(6,712,1,5);C(82,434,1,5);C(0,153,1,6);C(4,706,1,4);F();
		return 2.284233;
	}

	void check_signature (unsigned int pc) {
		switch (pc) {
		case 0x40591c:
		case 0x415aa4:
		case 0x40d86d:
		case 0x4221de:
		case 0x437b22:
		case 0x40f54f:
		case 0x417de9:
		case 0x40fcf2:
		case 0x413d5d:
		case 0x40a75f:
		case 0x22e36:
		case 0x16eb0b:
		case 0x402ba5:
		case 0x15c7f7:
		case 0x43e82f:
		case 0xa0a5df2:
		case 0xfc117b:
		case 0x141c69:
		case 0x703272a:
		case 0x68675b6:
		case 0x4822c8:
		case 0x401857:
		case 0x499aed:
		case 0x9fcfd71:
		case 0x40279e:
		case 0x4832a8:
		case 0xc413dcc:
		case 0x40cc85:
		case 0x567199:
		case 0x43fc18:
		case 0x411dc5:
		case 0x45b612:
		case 0x40b033:
		case 0x411715:
		case 0x40196c:
		case 0x5f890a4:
		case 0x457229:
		case 0x407336:
		case 0x7459eb:
		case 0x404ba9:
		signature = pc; break;
		default:;
		}
	}

	double compute_indices (unsigned int pc, sampler_info *u) {
          fprintf(stderr,"compute_indices: %x",pc);
          return compute_indices_0(pc,u);
          /**
		switch (signature) {
		case 0x40591c: return compute_indices_0 (pc, u);
		case 0x415aa4: return compute_indices_1 (pc, u);
		case 0x40d86d: return compute_indices_2 (pc, u);
		case 0x4221de: return compute_indices_3 (pc, u);
		case 0x437b22: return compute_indices_4 (pc, u);
		case 0x40f54f: return compute_indices_5 (pc, u);
		case 0x417de9: return compute_indices_6 (pc, u);
		case 0x40fcf2: return compute_indices_7 (pc, u);
		case 0x413d5d: return compute_indices_8 (pc, u);
		case 0x40a75f: return compute_indices_9 (pc, u);
		case 0x22e36: return compute_indices_10 (pc, u);
		case 0x16eb0b: return compute_indices_11 (pc, u);
		case 0x402ba5: return compute_indices_12 (pc, u);
		case 0x15c7f7: return compute_indices_13 (pc, u);
		case 0x43e82f: return compute_indices_14 (pc, u);
		case 0xa0a5df2: return compute_indices_15 (pc, u);
		case 0xfc117b: return compute_indices_16 (pc, u);
		case 0x141c69: return compute_indices_17 (pc, u);
		case 0x703272a: return compute_indices_18 (pc, u);
		case 0x68675b6: return compute_indices_19 (pc, u);
		case 0x4822c8: return compute_indices_20 (pc, u);
		case 0x401857: return compute_indices_21 (pc, u);
		case 0x499aed: return compute_indices_22 (pc, u);
		case 0x9fcfd71: return compute_indices_23 (pc, u);
		case 0x40279e: return compute_indices_24 (pc, u);
		case 0x4832a8: return compute_indices_25 (pc, u);
		case 0xc413dcc: return compute_indices_26 (pc, u);
		case 0x40cc85: return compute_indices_27 (pc, u);
		case 0x567199: return compute_indices_28 (pc, u);
		case 0x43fc18: return compute_indices_29 (pc, u);
		case 0x411dc5: return compute_indices_30 (pc, u);
		case 0x45b612: return compute_indices_31 (pc, u);
		case 0x40b033: return compute_indices_32 (pc, u);
		case 0x411715: return compute_indices_33 (pc, u);
		case 0x40196c: return compute_indices_34 (pc, u);
		case 0x5f890a4: return compute_indices_35 (pc, u);
		case 0x457229: return compute_indices_36 (pc, u);
		case 0x407336: return compute_indices_37 (pc, u);
		case 0x7459eb: return compute_indices_38 (pc, u);
		case 0x404ba9: return compute_indices_39 (pc, u);
		default: return compute_indices_40 (pc, u);
		}
          */
	}


        /**
         * Takes pc and tries to predict u->prediction.
         */
	void compute_sums (unsigned int pc, sampler_info *u) {
          
          // initialize indices
          /**
           * For each table we set indices to 0
           */
          for (int i = 0; i < num_tables; i++)
            u->indices[i] = 0;

          // for each sample, accumulate a partial index

          /**
           *
           */
          histories[0] = global_history;
          histories[1] = path_history;
          histories[2] = &callstack_history;
          
          // TODO 
          // Maybe hook in here
          // Just pass in global_history,path_history,callstack_history,local_history
          // so for each pc we will create a string with these three variables
          // and desire branch outcome.
          // create a convolution network 
          // allow it to train on the trace.
          // graph the predictions with accuracy. ?
          // what should it be , what is it?
          // this will not qualify with any of the entries.
          // final result should be yes/no decision
          // why do we need to give it the PC ? -> just decide based on local-hostry,path_history and call_stack history
          
          
          /**
           * This seems like a completely magical quantity. I have no
           * idea what is going on here ??
           */
          double coeff = compute_indices (pc, u);
          
          // Final result is some double value
          
          /**
           * Q: What are the indices supposed bo be doing ? 
           */          
          // now we have all the indices. use them to compute
          // the weighted and unweighted sums


          /** 
           * Since we always keep the list of local histories.
           */
          if (local_table_size) {
            
            u->local_index = get_local_history (pc);
            int x = local_pht[u->local_index];
            
            /**
             * Based on the coeff computed for this pc 
             * we decide how much weight we are going to assign to the
             * local histry from the local pht
             * 
             * We also consider distinct values of coeff_local and coeff.
             *
             */            
            // it seems we can just multipythe value into x which is the value of local_pht ??
            u->sum += (int) (coeff * x);
            u->weighted_sum += (int) (coeff_local * x);
            
          }
	}


        /**
         * Look up a prediction in the predictor
         */
	branch_info *lookup (unsigned int pc, bool, bool) { 
          sampler_info *u = &p;
          u->pc = pc;
          if (!signature) {
            check_signature (pc);
            if (signature) reset_tables ();
          }

          // initialize two sums

          u->sum = 0.0;
          u->weighted_sum = 0.0;

          // an index into the filter

          unsigned int idx;
          bool new_branch = false;
          if (filter_size) {
            idx = hash (u->pc, 0) % filter_size;

            // remember whether this is the first time we are seeing this branch

            new_branch = !(filter[0][idx] || filter[1][idx]);
          }

          // if the branch is in exactly one filter...

          if (filter_size && (filter[0][idx] != filter[1][idx])) {

            // ...predict the branch will do what its been observed doing before

            /**
             * Here we catch and inter out trivial branches which have
             * so far never switched in behavior
             */
            if (filter[0][idx]) {
              fprintf(stderr,"Trivial False: %x\n",pc);
              u->prediction (false);
              u->weighted_sum = min_weight;
              u->sum = min_weight;
              
            } else if (filter[1][idx]) { 
              fprintf(stderr,"Trivial True: %x\n",pc);
              u->prediction (true);
              u->weighted_sum = max_weight;
              u->sum = max_weight;
              
            }
            
          } else {  
            /**
             * If the branch has be been seen to switch state then it
             * uses non trivial behavior which needs to be predicted.
             */
            
            // it's in both filters; have to predict it with
            // perceptron predictor
            fprintf(stderr,"Perceptron Predict: %x\n",pc);
            compute_sums (pc, u);
          }
          
          if (new_branch) {
            // if this is a new branch, predict it statically
                  
            /**
             * Since we disable static prediction i dont know
             * what this is about
             */
            u->prediction (static_prediction);

          } else {
            // give the prediction with the kind of sum that seems do
            // be doing well
            bool prediction;
                        
            if (psel >= 0) {
              prediction = u->weighted_sum >= 0.0;
            } else {
              prediction = u->sum >= 0.0;
            }
            u->prediction (prediction);
          }
          return u;
	}

	// update an executed branch

	void update (branch_info *hu, bool taken) {
		sampler_info *u = (sampler_info *) hu;


                /**
                 * Variables used to inhibit behavior of trivial
                 * branches from polluting our pattern history table.
                 */
		bool never_taken = false, never_untaken = false;

		// an index into the filters
		unsigned int idx = 0;

		// we don't know that we need to update yet
		bool need_to_update = false;
                
		if (filter_size) {

                  // get the index into the filters
                  
                  // Get an index into our simple behavior inhibition fitlers
                  idx = hash (u->pc, 0) % filter_size;
                  

                  // the first time we encounter a branch, we will update
                  // no matter what to prime the predictor
                  
                  /**
                   * If we see the branch as being taken in either
                   * filter[0] or filter[1] we will consider the, and
                   * no filter bit has been set in either one then we
                   * will request an update.
                   */
                  
                  /**
                   * 1. both filter[0] and filter[1] are zero  -> update
                   * 2. filter[1] are different filter[0]      -> no update
                   * 3. filter[0] and filter[1] are one        -> no update
                   */
                  need_to_update = !(filter[0][idx] || filter[1][idx]);

                  /**
                   * Update the bit in either all-zero-bit filter i.e
                   * fitler[0] or in all-one-bit filter ie. filter[1].
                   * Acknowledge that we have seen it in this sense.
                   * taken == 1 -> filter[1] <- true
                   * taken == 0 -> filter[0] <- true
                   */
                  
                  // We have seen this branch with this sense at least once now
                  filter[!!taken][idx] = true;


                  /**
                   * filter[0] -> true  => never_untaken <- 0 # we have seen it as untaken at least once
                   * filter[1] -> true  => never_take   <-  0 # we have seen it as taken at least once
                   */
                  // see if the branch has never been taken or never been not taken
                  
                  never_untaken = !filter[0][idx];
                  never_taken = !filter[1][idx];

		}

		// we need to update if there is no filter or if the filter
		// has the branch with both senses
                
                /**
                 * Assuming not in simple cases :
                 *    - Trivial cases : First Time, Zero Filter size
                 *    - Non-Trivial Cases : If we are in non-trivial 
                 *       - If both filter[0] and filter[1] -> branch has been seen in both taken and not taken senses
                 *       - We need to update.
                 */
		need_to_update |= !filter_size || (filter[0][idx] && filter[1][idx]);
                
                // Allow the branch decistion to hit the perceptron tables                
		if (need_to_update) {
                  
                  // was the prediction from the weighted sum correct?
                  
                  bool weighted_correct = (u->weighted_sum >= 0) == taken;

                  // was the prediction from the non-weighted sum correct?

                  bool correct = (u->sum >= 0) == taken;

                  // Keep track of which one of those techniques is performing better
                  
                  /**
                   * We compare the validity of weigthed and non weighted predictions
                   * psel is a saturating metric of confidence. 
                   *
                   * where positive values indicate picking
                   * weighted_sum and negative values mean use un
                   * weighted sums.
                   */
                  
                  if (weighted_correct && !correct) {
                    if (psel < 511)
                      psel++;
                  } else if (!weighted_correct && correct) {
                    if (psel > -512)
                      psel--;
                  }

                  // update the coefficients
                  /**
                   * Iterate through each table 
                   */
                  for (int i = 0; i < num_tables; i++) {

                    // see what the sum would have been without this table
                    // pick table and corresponding indices
                    
                    /**
                     * 1. remove table's weight from the total sum
                     * 2. check if the sum would have been correct 
                     */
                    int sum = u->sum - table[i][u->indices[i]];
                    

                    // would the prediction have been correct?

                    bool this_correct = (sum >= 0) == taken;
                    if (correct) {// if branch was correctly predicted
                      // this table helped; increase its coefficient

                      if (!this_correct)
                        if (coeffs[i] < (max_exp-1)) coeffs[i]++;
                      
                    } else { // branch was not currect 
                      // this table hurt; decrease its coefficient
                      
                      if (this_correct)
                        if (coeffs[i] > -max_exp) coeffs[i]--;
                    }
                  }

                  // Get the magnitude of the sum times a fudge factor
                  
                  /**
                   * We look at magnited, deviation from 0 of sum from 
                   */
                  int a = abs ((int) (u->sum * coeff_train));

                  // get a random number between 0 and 1

                  double p = (rand_r (&my_seed) % 1000000) / 1000000.0;

                  // if the branch was predicted incorrectly according
                  // to the unweighted sum, if if the magnitude of the
                  // some does not exceed some random value near theta,
                  // then we must update the predictor
                  
                  /**
                   * Train if 
                   * 1. Incorrect prediction
                   * 2. If magnitude of deviation from zero within random error factor
                   *    is theta. That is we have not reached training saturation
                   */
                  bool do_train = !correct || (a - (p/2 * theta_fuzz)) < theta;
                  
                  if (do_train) {
                    // train the global tables

                    for (int i=0; i<num_tables; i++)                                  
                      table[i][u->indices[i]] = 
                        satincdec(table[i][u->indices[i]], taken, max_weight, min_weight);
                          

                    // train the local table                                
                    /**
                     * Index into the local_pht table using some sort of hash of the pc
                     * Saturating  increment the pht entry.
                     */
                    if (local_table_size) {

                      int x = local_pht[u->local_index];
                      x = satincdec (x, taken, max_weight, min_weight);
                      local_pht[u->local_index] = x;
                      
                    }
                    
                    
                  }

                  // adjust theta
                  /**
                   * This is part of dynamic threshold setting in
                   * branch prediction schemes where we want to adjust
                   * speed of convergence on incorrenct branches.
                   */
                  threshold_setting (u, correct, a);
		}


                /**
                 * Not sure why the pc is taken into 
                 */
                
		// update global, path, and local histories
                
                /**
                 * Q: I still dont undertand why we care about the
                 * fourth bit of this pc while putting it in the
                 * global history path. ??
                 */
		shift_history(global_history, taken ^ !!(u->pc & 4));
		shift_history(path_history, (u->pc >> path_bit) & 1);                
                
		if (local_table_size) {
                  
                  // don't record into a local history if this branch has trivial behavior
                  
                  bool inhibit_local = never_taken || never_untaken;
                  
                  if (!inhibit_local) {
                    local_histories[hash(u->pc,0)%local_num_histories] <<= 1;
                    local_histories[hash(u->pc,0)%local_num_histories] |= taken;
                  }
		}
	}


	// record extra information provided by the infrastructure

	void info (unsigned int pc, unsigned int optype) {

		// maintain the callstack history

		if (optype == 3) {
			// push
			callstack_history <<= 1;
			callstack_history |= !!(pc & 4);
		} else if (optype == 4) {
			// pop 
			callstack_history >>= 1;
		}

		// based on a mask, include or don't include certain bits in the histories

		if (mask & (1<<optype)) {
			if (mask & 0x40000000)
				shift_history (path_history, (pc >> path_bit) & 1);
			if (mask & 0x80000000)
				shift_history (global_history, 1);
		}
	}
};

class PREDICTOR{

	branch_predictor *pred;
 public:

  // The interface to the four functions below CAN NOT be changed

	PREDICTOR(void) {

	pred = new sampler (
		(32768+(1024/8))-376, 		// number of bytes allocated to predictor
		6, 		// bit width
		32, 		// number of tables
		24, 		// folding parameter, i.e. hash function width
		16384,		 // filter size
		24, 		// initial theta
		6, 		// shift shift
		7, 		// adaptive threshold learning speed
		2.25, 		// local predictor fudge factor
		2048, 		// local predictor table size
		7, 		// local predictor history length
		256, 		// local predictor number of histories
		8.72, 		// theta taper
		1.004, 		// fudge factor 2
		1.000025800, 		// fudge factor 3
		131072, 	// maximum exponent for coefficient learning
		true,
		0x40000018,
		false, 
		5);
        fprintf(stderr,"Perceptron Num Tables: %d",32);
	}

	branch_info *u;

        bool GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN){
          u = pred->lookup(PC & 0x0fffffff, false, false);
          //fprintf(stderr,"PERCEPTRON:GetPrediction 0x%2x Prediction:%d \n",PC,u->prediction());
          return u->prediction();
	}

        void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN){
          // fprintf(stderr,"PERCEPTRON:UpdatePredictor 0x%2x Prediction:%d Actual:%d Success:%d \n", PC, predDir, resolveDir, predDir == resolveDir);
          pred->update (u, resolveDir);
	}

        void TrackOtherInst(UINT64 PC, OpType opType, bool branchDir, UINT64 branchTarget) {
		switch (opType) {                  
                           // case OPTYPE_CALL_DIRECT:
                           // case OPTYPE_RET:
                           // case OPTYPE_BRANCH_UNCOND:
                           // case OPTYPE_INDIRECT_BR_CALL:
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
                  pred->info (PC & 0x0fffffff, opType); break;
                default: ;
		}
	}

  // Contestants can define their own functions below

};
  
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// END PERCEPTRON
////////////////////////////////////////////////////////////////////////////////////////////////////
#define LOOPPREDICTOR	//loop predictor enable
#define NNN 1			// number of entries allocated on a TAGE misprediction

// number of tagged tables
/**
 * Hard coded value for the number of tag tables .
 */
#ifndef REALISTIC
#define NHIST 15
#else 
#define NHIST 12
#endif

#define HYSTSHIFT 2 // bimodal hysteresis shared by 4 entries
/**
 * Number of entries in bi-modal predictor = 2^LOGB = 16,384
 */
#define LOGB 14 // log of number of entries in bimodal predictor 



#define PERCWIDTH 6 //Statistical coorector maximum counter width 

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
#define LOGL 5
#define WIDTHNBITERLOOP 10	// we predict only loops with less than 1K iterations
#define LOOPTAG 10		//tag width in the loop predictor


//update threshold for the statistical corrector
#ifdef REALISTIC
#define LOGSIZEUP 0
#else
#define LOGSIZEUP 5
#endif
int Pupdatethreshold[(1 << LOGSIZEUP)];	//size is fixed by LOGSIZEUP
#define INDUPD (PC & ((1 << LOGSIZEUP) - 1))

// The three counters used to choose between TAGE ang SC on High Conf TAGE/Low Conf SC
int8_t  FirstH, SecondH, ThirdH;
#define CONFWIDTH 7	//for the counters in the choser


#define PHISTWIDTH 27		// width of the path history used in TAGE




#define UWIDTH 2   // u counter width on TAGE

#define CWIDTH 3   // predictor counter width on the TAGE tagged tables

#define HISTBUFFERLENGTH 4096	// we use a 4K entries history buffer to store the branch history


//the counter(s) to chose between longest match and alternate prediction on TAGE when weak counters

#ifdef REALISTIC
#define LOGSIZEUSEALT 0
#else
#define LOGSIZEUSEALT 8 
#endif
#define SIZEUSEALT  (1<<(LOGSIZEUSEALT))
#define INDUSEALT (PC & (SIZEUSEALT -1))
int8_t use_alt_on_na[SIZEUSEALT][2];



long long GHIST;


//The two BIAS tables in the SC component
#define LOGBIAS 7
int8_t Bias[(1<<(LOGBIAS+1))];
#define INDBIAS (((PC<<1) + pred_inter) & ((1<<(LOGBIAS+1)) -1))
int8_t BiasSK[(1<<(LOGBIAS+1))];
#define INDBIASSK ((((PC^(PC>>LOGBIAS))<<1) + pred_inter) & ((1<<(LOGBIAS+1)) -1))

bool HighConf;
int LSUM;
int8_t BIM;



// utility class for index computation
// this is the cyclic shift register for folding 
// a long global history into a smaller number of bits;
// see P. Michaud's PPM-like predictor at CBP-1
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
class lentry			//loop predictor entry
{
public:
  uint16_t NbIter;		//10 bits
  uint8_t confid;		// 4bits
  uint16_t CurrentIter;		// 10 bits

  uint16_t TAG;			// 10 bits
  uint8_t age;			// 4 bits
  bool dir;			// 1 bit

  //39 bits per entry    
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
   * u - something 
   */
  int8_t u;

  /**
   * Counter, Tag and  u start off empty
   */
  gentry () {
    ctr = 0;
    tag = 0;
    u = 0;
  }
  
};




int TICK;// for the reset of the u counter


uint8_t ghist[HISTBUFFERLENGTH];
int ptghist;

// Path history
long long phist;

// Utility for computing TAGE indices
folded_history ch_i[NHIST + 1];	

// Utility for computing TAGE tags
folded_history ch_t[2][NHIST + 1];	

// For the TAGE predictor
bentry *btable;			//bimodal TAGE table
gentry *gtable[NHIST + 1];	// tagged TAGE tables

/**
 * Using perceptron predictor as alternative to bimodal table
 */
PERCEPTRON::PREDICTOR* perceptron_predictor;

#ifdef REALISTIC

/**
 * History lengths for different tables.
 */
int m[NHIST+1]={0,8,12,18,27,40,60,90,135,203,305,459,690};

/**
 * Tag width for different tables
 */
int tag_width[NHIST + 1]  ={0,8, 9, 9,10,10,11,11,12,12,13,13,14};

/**
 * Log of number of entries in different tables
 */
int logg[NHIST + 1]={0,10,10,10,10,10,10,10,10,10,10,10,10};

#else

/**
 * History lenghts for different tag tables.
 */
int m[NHIST+1]={0,6,10,18,25,35,55,69,105,155,230,354,479,642,1012,1347};

/**
 * Width of tags for different tables
 */
int tag_width[NHIST + 1]={0,7,9,9,9,10,11,11,12,12,12,13,14,15,15,15};

/**
 * Log of number of entries in different tables
 */
int logg[NHIST + 1]={0,10,10,10,11,10,10,10,10,10,9,9,9,8,7,7};	
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
 * Contains the bank number for the longest bank where with matching history.
 */
int HitBank;	


/**
 * Contains thebank number for next longest bank with matching tag.
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
int LHIT;			//hitting way in the loop predictor
int LTAG;			//tag on the loop predictor
bool LVALID;			// validity of the loop predictor prediction
int8_t WITHLOOP;		// counter to monitor whether or not loop prediction is beneficial
#endif


/**
 * Computes and prints the size of the bi-modal table.
 */
int predictorsize () {
  
  int storage_size = 0;
  int inter = 0;
  
  for (int i = 1; i <= NHIST; i += 1) {
    storage_size += (1 << (logg[i])) * (CWIDTH + UWIDTH + tag_width[i]);
  }
   
  storage_size += 2 * (SIZEUSEALT) * 4;
  storage_size += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));
  storage_size += m[NHIST];
  storage_size += PHISTWIDTH;
  storage_size += 10 ; //the TICK counter

  fprintf(stderr, " (TAGE %d) ", storage_size);  

#ifdef LOOPPREDICTOR
  inter= (1 << LOGL) * (2 * WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1);fprintf (stderr, " (LOOP %d) ", inter); 
  storage_size+= inter;  
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
  inter += 4; // the history stack pointer
  inter += 16; //global histories for SC  
  inter += NSECLOCAL * (Sm[0]+Tm[0]);
#endif
  
  inter += NLOCAL * Lm[0];
  inter += 3*CONFWIDTH; //the 3 counters in the choser
  storage_size+= inter;
  fprintf (stderr, " (SC %d) ", inter);

#ifdef PRINTSIZE
  fprintf (stderr, " (TOTAL %d) ", storage_size);
#endif
  return (storage_size);
}




class PREDICTOR
{
public:
  PREDICTOR (void)
  {

    reinit ();
#ifdef PRINTSIZE    
    predictorsize ();
#endif
  }


  void reinit ()
  {

#ifdef LOOPPREDICTOR
    ltable = new lentry[1 << (LOGL)];
#endif

    // creates a 32 KB perceptron predictor.
    perceptron_predictor = new PERCEPTRON::PREDICTOR();
    
    /**
     * For each history length 
     */
    for (int i = 1; i <= NHIST; i++) {      
      gtable[i] = new gentry[1 << (logg[i])];
    }

    /**
     * Initialized the bimodal table with 2^LOGB entries
     */
    btable = new bentry[1 << LOGB];

    for (int i = 1; i <= NHIST; i++)
      {
	ch_i[i].init (m[i], (logg[i]), i - 1);
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
      for (int j = 0; j < ((1 << LOGGNB) - 1); j++){
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
     * Initialize the base bi-modal table with 2^LOGB entries with
     * hysteresis 1 and prediction 0
     */
    for (int i = 0; i < (1 << LOGB); i++)
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

    for (int i = 0; i < SIZEUSEALT; i++) {
	use_alt_on_na[i][0] = 0;
	use_alt_on_na[i][1] = 0;
    }

    TICK = 0;
    ptghist = 0;
    phist = 0;
    
  }


  
  /**
   * Simply select the last LOGB number of bits of PC
   * - index function for the bimodal table
   */

  int bindex(uint32_t PC)
  {
    return ((PC) & ((1 << (LOGB)) - 1));
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
  int index_functions_for_tables (long long A, int size, int bank)
  {
    int A1, A2;
    
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << logg[bank]) - 1));
    A2 = (A >> logg[bank]);
    A2 = ((A2 << bank) & ((1 << logg[bank]) - 1)) + (A2 >> (logg[bank] - bank));
    A = A1 ^ A2;
    A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));
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
    
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    
    // PC XOR 
    index = PC ^ (PC >> (abs (logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^ index_functions_for_tables (hist, M, bank);
    
    return (index & ((1 << (logg[bank])) - 1));
    
  }

  /**
   * Takes two folded histories and the bank number we need the tag
   * for and returns tag for comparison
   */
  uint16_t gtag (unsigned int PC, int bank, 
                 folded_history * ch0,
		 folded_history * ch1)
  {
    int tag = PC ^ ch0[bank].comp ^ (ch1[bank].comp << 1);
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
  int lindex (uint32_t PC)
  {
    return ((PC & ((1 << (LOGL - 2)) - 1)) << 2);
  }


// loop prediction: only used if high confidence
// skewed associative 4-way
// At fetch time: speculative
#define CONFLOOP 15

  bool getloop (uint32_t PC)
  {
    LHIT = -1;

    LI = lindex (PC);
    LIB = ((PC >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1));
    LTAG = (PC >> (LOGL - 2)) & ((1 << 2 * LOOPTAG) - 1);
    LTAG ^= (LTAG >> LOOPTAG);
    LTAG = (LTAG & ((1 << LOOPTAG) - 1));

    for (int i = 0; i < 4; i++)
      {
	int index = (LI ^ ((LIB >> i) << 2)) + i;

	if (ltable[index].TAG == LTAG)
	  {
	    LHIT = i;
	    LVALID = ((ltable[index].confid == CONFLOOP)
		      || (ltable[index].confid * ltable[index].NbIter > 128));
	    if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
	      return (!(ltable[index].dir));
	    else
	      return ((ltable[index].dir));
	  }
      }

    LVALID = false;
    return (false);

  }


  void loopupdate (uint32_t PC, bool Taken, bool ALLOC)
  {
    if (LHIT >= 0) {
      int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
      //already a hit 
      if (LVALID) {
        if (Taken != predloop) {
          // free the entry
          ltable[index].NbIter = 0;
          ltable[index].age = 0;
          ltable[index].confid = 0;
          ltable[index].CurrentIter = 0;
          return;
        }
        else if ((predloop != tage_pred) || ((MYRANDOM () & 7) == 0))
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
		if (ltable[index].confid < CONFLOOP)
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

  /**
   * Instead of bimodal predictor what if we use a perceptron predictor.
   */
  bool get_bimodal_prediction ()
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
  int MYRANDOM ()
  {
    Seed++;
    Seed ^= phist;
    Seed = (Seed >> 21) + (Seed << 11);
    return (Seed);
  };


  //  TAGE PREDICTION: same code at fetch or retire time but the index and tags must recomputed
  void Tagepred(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN) {
    // (UINT32 PC) {
    
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
     * Index into bi-modal table is simple last LOGB bits of PC
     */
    BI = PC & ((1 << LOGB) - 1);


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
	if (gtable[i][GI[i]].tag == GTAG[i])
	  {
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
    for (int i = HitBank - 1; i > 0; i--)
      {
	if (gtable[i][GI[i]].tag == GTAG[i])
	  {

	    AltBank = i;
	    break;
	  }
      }

    // Computes the prediction and the alternate prediction

    if (HitBank > 0) // was a hit
      {
	if (AltBank > 0) // had alternate hit
	  alttaken = (gtable[AltBank][GI[AltBank]].ctr >= 0);
	else{

          alttaken = get_bimodal_prediction ();
          // override bimodal prediction
          alttaken = perceptron_predictor->GetPrediction(PC,btbANSF,btbATSF,btbDYN);
        }

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
	alttaken = get_bimodal_prediction ();
        // override bimodal prediction
        alttaken = perceptron_predictor->GetPrediction(PC,btbANSF,btbATSF,btbDYN);
	tage_pred = alttaken;
	LongestMatchPred = alttaken;
      }


  }
//compute the prediction

/*
  bool GetPrediction (UINT32 PC)
  {
*/
  bool GetPrediction(UINT64 PC, bool btbANSF, bool btbATSF, bool btbDYN) {
    // computes the TAGE table addresses and the partial tags
    
    Tagepred(PC,btbANSF,btbATSF,btbDYN);
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
	    if ((abs (LSUM) <
		 Pupdatethreshold[INDUPD] / 3))
	      pred_taken = (FirstH < 0) ? SCPRED : pred_inter;

	    else
	      if ((abs (LSUM) <
		   2 * Pupdatethreshold[INDUPD] / 3))
	      pred_taken = (SecondH < 0) ? SCPRED : pred_inter;
	    else
	      if ((abs (LSUM) <
		   Pupdatethreshold[INDUPD]))
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

  void    UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget, bool btbANSF, bool btbATSF, bool btbDYN){
    /*
    void UpdatePredictor (UINT32 PC, bool resolveDir, bool predDir,
			UINT32 branchTarget)
  {
    */
    
    perceptron_predictor->UpdatePredictor(PC,opType,resolveDir,predDir,branchTarget,btbANSF,btbATSF,btbDYN);
#ifdef LOOPPREDICTOR
    if (LVALID)
      {


	if (pred_taken != predloop)
	  ctrupdate (WITHLOOP, (predloop == resolveDir), 7);

      }

    loopupdate (PC, resolveDir, (pred_taken != resolveDir));


#endif

    bool  SCPRED = (LSUM >= 0);    
    if (pred_inter != SCPRED)
	{
     	if ((abs (LSUM) <
             Pupdatethreshold[INDUPD]))        if ((HighConf))
	    {

	      if ((abs (LSUM) <
		   Pupdatethreshold[INDUPD] / 3))
		ctrupdate (FirstH, (pred_inter == resolveDir), CONFWIDTH);
	      else
		if ((abs (LSUM) <
		     2 * Pupdatethreshold[INDUPD] / 3))
		ctrupdate (SecondH, (pred_inter == resolveDir), CONFWIDTH);
	      else
		if ((abs (LSUM) <
		     Pupdatethreshold[INDUPD]))
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
	if ((MYRANDOM () & 31) != 0)
	  ALLOC = false;
      //do not allocate too often if the overall prediction is correct 

	if (HitBank > 0)	  {
          // Manage the selection between longest matching and alternate matching
          // for "ps`eudo"-newly allocated longest matching entry
            
	    bool PseudoNewAlloc =
	      (abs (2 * gtable[HitBank][GI[HitBank]].ctr + 1) <= 1);
// an entry is considered as newly allocated if its prediction counter is weak
	    if (PseudoNewAlloc)
	      {
		if (LongestMatchPred == resolveDir)
		  ALLOC = false;
// if it was delivering the correct prediction, no need to allocate a new entry
//even if the overall prediction was false
		if (LongestMatchPred != alttaken)
		  {
		    int index = (INDUSEALT) ^ LongestMatchPred;
		    ctrupdate (use_alt_on_na[index]
			       [HitBank > (NHIST / 3)],
			       (alttaken == resolveDir), 4);

		  }

	      }
	  }
      
 
     if (ALLOC)
	{

	  int T = NNN;
          
	  int A = 1;
	  if ((MYRANDOM () & 127) < 32)
	    A = 2;
	  int Penalty = 0;
	  int NA = 0;
          for (int i = HitBank + A; i <= NHIST; i += 1)
	    {
	      if (gtable[i][GI[i]].u == 0)
		{
		  gtable[i][GI[i]].tag = GTAG[i];
		  gtable[i][GI[i]].ctr = (resolveDir) ? 0 : -1;
		  NA++;
		  if (T <= 0)
		    {
		      break;
		    }
                  i +=  1;
		  T -= 1;
		}
	      else
		{
		  Penalty++;
		}
	    }
	  TICK += (Penalty -NA );
//just the best formula for the Championship
	  if (TICK < 0)
	    TICK = 0;
 	  if (TICK > 1023)
	    {
	      for (int i = 1; i <= NHIST; i++)
		for (int j = 0; j <= (1 << logg[i]) - 1; j++)
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
  int Gpredict (UINT32 PC, long long BHIST, int *length, int8_t ** tab,
		int NBR, int logs)
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
      
      HistoryUpdate (PC, opType, taken, branchTarget, phist,
                     ptghist, ch_i,
                     ch_t[0], ch_t[1],
                     L_shist[INDLOCAL], 
                     S_slhist[INDSLOCAL], T_slhist[INDTLOCAL],HSTACK[pthstack], GHIST);
      break;


    default:;
    }  

  }

};



#endif
