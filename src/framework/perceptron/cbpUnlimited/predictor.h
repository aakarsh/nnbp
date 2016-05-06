#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include "utils.h"
#include "tracer.h"
#include <math.h>
#include <string.h>


// search for "class sampler_info" to see my predictor code. preceding it is
// some stuff from my infrastructure.

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

#include <math.h>

typedef char byte;

// this class holds information learned or computed in the prediction phase
// that we would like to remember for the update phase. it does not count
// against the hardware budget because it can be recomputed; it is a programming 
// convenience rather than a necessity.

#define MAXIMUM_HISTORY_LENGTH  896

class sampler_info : public branch_info {
public:

	// the perceptron output for this prediction

	float
		sum, 
		weighted_sum; 

	// the pc of the branch being predicted

	unsigned int 
		pc;

	// table indices

	unsigned int 
		*indices, 
		local_index;

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

	bool index_history (unsigned long long int *v, unsigned int index) {
		unsigned int word_index = index / (sizeof (unsigned long long int) * 8);
		if (word_index >= (unsigned int) history_size) return false; // too far
		unsigned int word_offset = index % (sizeof (unsigned long long int) * 8);
		return (v[word_index] >> word_offset) & 1;
	}

	// fold a history of 'on' bits into 'cn' bits, skipping 'st'
	// bits. inspired by Seznec's history folding function

	unsigned int fold_history (unsigned long long int *v, int start, int on, int cn, int st) {
		unsigned int r = 0;

		// first part: go up to the highest multiple of 'cn' less than 'on'

		int lim = (on / cn) * cn;
		int i;
		unsigned int q;
		for (i=0; i<lim; i+=cn) {
			q = 0;
			for (int j=i; j<i+cn; j+=st) {
				q <<= 1;
				q |= index_history (v, start+j);
			}
			r ^= q;
		}

		// second part: peel off last iteration and check we don't go over 'on'

		q = 0;
		for (int j=i; j<on; j++) {
			q <<= 1;
			q |= index_history (v, start+j);
		}
		r ^= q;
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
		int _bits = 6, 				// bit width of a weight
		int _num_tables = 32, 			// number of perceptron weights tables
		int _folding = 24, 			// width of hash function domain
		int _filter_size = 16384, 		// number of entries in filter
		int _theta = 24, 			// initial value of threshold
		int _path_bit = 6, 			// which bit of the PC to use for global path history
		int _speed = 7, 			// speed for dynamic threshold fitting
		double _coeff_local = 2.25,	 	// coefficient for local predictor weight
		int _local_table_size = 2048, 		// size of local predictor pattern history table
		int _local_history_length = 7, 		// history length for local predictor
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

#define N 1000000000
#define VERBOSE
		my_heap = new unsigned char[N];
		memset (my_heap, 0, N);
		heap_end = &my_heap[N];
		bump_pointer = my_heap;
		signature = 0;

		// we have this many total bits to work with

		int total_bits = bytes * 8;
#ifdef VERBOSE
		printf ("start with %d bits\n", total_bits);
#endif

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

		total_bits -= 64;

		// initialize weighted versus non-weighted selector and account for its 10 bit width

		psel = 0;
		total_bits -= 10;

		// coefficients are 18 bits each

		total_bits -= 18 * num_tables;
		coeffs = (int *) my_malloc (sizeof (int) * num_tables);
		for (int i=0; i<num_tables; i++) coeffs[i] = 0;

		// set up and account for local predictor

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

		p.indices = (unsigned int *) my_malloc (num_tables * sizeof (unsigned int));

		// compute maximum and minimum weights

		max_weight = (1<<bits)/2-1;
		min_weight = -(1<<bits)/2;

		// initialize and accont for adaptive threshold fitting counter (it is never more than 8 bits)

		tc = 0;
		total_bits -= 8;

		// initialize and account for theta (it is never more than 9 bits)

		total_bits -= 9;
		theta = _theta;

		// global history 

		global_history = (unsigned long long int *) my_malloc (sizeof (unsigned long long int) * (history_size+1));

		// path history

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
		// don't care for unlimited
		// assert (total_bits >= 0);
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

	void shift_history (unsigned long long int *v, bool t) {
		for (int i=0; i<=history_size; i++) {
			bool nextt = !!(v[i] & (1ull<<63));
			v[i] <<= 1;
			v[i] |= t;
			t = nextt;
		}
	}

	// saturating increment/decrement

	byte satincdec (byte weight, bool taken, int max, int min) {
		if (taken) {
			return (weight < max) ? weight + 1 : weight;
		} else {
			return (weight > min) ? weight - 1 : weight;
		}
	}

	// dynamic threshold updating per Seznec

	void threshold_setting (sampler_info *u, bool correct, int a) {
		if (!correct) {
			tc++;
			if (tc >= speed) {
				theta++;
				tc = 0;
			}
		}
		if (correct && a < theta) {
			tc--;
			if (tc <= -speed) {
				theta--;
				tc = 0;
			}
		}
	}

	// return a local history for this pc

	unsigned int get_local_history (unsigned int pc) {
		unsigned int l = local_histories[hash(pc,0) % local_num_histories];
		l &= (1<<local_history_length)-1;
		l <<= local_shift;
		return (l ^ pc) % local_table_size;
	}

	void compute_partial_index (unsigned int *index, unsigned int pc, int start, int end, int kind, int which, int stride) {
		unsigned int partial_index = hash (pc, which);
		partial_index ^= fold_history (histories[kind], start, end-start, folding, stride);
		*index = *index * 2 + partial_index;
	}

	void finish_index (unsigned int *index, int this_table, sampler_info *u) {
		*index %= table_size;
		int x = table[this_table][*index];
		u->sum += x;
		u->weighted_sum += pow (coeff_base, coeffs[this_table]) * x;
		u->indices[this_table] = *index;
		*index = 0;
	}

#define C(a, b, c, d) compute_partial_index (&j, pc, a, b, c, t, d);
#define B() compute_partial_index (&j, pc, 0, 0, 0, t, 0);
#define F() finish_index (&j, t++, u);

	double compute_indices_0 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,3,0,6);C(0,3,0,1);F();
		C(0,7,1,4);F();
		B();C(0,14,1,5);F();
		C(0,18,0,3);F();
		B();C(0,3,2,2);C(9,18,1,4);F();
		B();C(0,3,1,5);C(8,20,1,2);F();
		C(2,9,0,2);C(14,26,1,4);F();
		C(0,1,1,3);C(0,2,0,4);C(0,17,1,5);C(11,30,1,1);C(11,30,1,1);F();
		C(0,11,1,2);C(8,12,1,4);C(0,4,0,8);C(7,30,1,4);F();
		C(1,42,0,7);C(46,147,1,4);F();
		C(0,52,0,4);C(25,46,1,2);C(0,18,0,3);C(14,21,1,6);F();
		C(1,6,0,4);C(2,55,0,1);C(8,39,0,4);F();
		C(2,9,0,2);B();C(0,3,0,2);C(36,38,0,2);C(49,57,1,5);C(21,22,1,3);C(2,5,2,3);F();
		C(49,57,1,5);C(36,60,0,8);F();
		C(7,11,0,1);C(15,53,0,4);C(13,58,0,4);C(13,20,1,1);C(0,3,1,6);F();
		C(0,4,0,2);C(0,16,1,8);C(0,65,1,2);F();
		C(11,20,1,1);C(3,89,0,8);F();
		C(2,7,0,5);C(66,139,1,1);C(9,16,0,3);F();
		B();C(99,142,0,2);C(0,9,0,3);F();
		C(1,24,1,8);C(27,71,1,8);F();
		C(6,10,1,3);C(8,163,1,2);F();
		C(9,221,1,6);F();
		C(0,31,1,2);C(0,253,1,8);F();
		C(1,12,1,4);C(0,255,1,8);B();F();
		C(105,671,1,7);F();
		C(7,36,0,4);C(26,34,1,3);C(10,264,1,2);F();
		C(0,7,0,1);C(45,351,1,2);C(0,2,0,1);C(6,27,1,4);C(0,46,0,8);C(6,14,0,3);C(5,33,0,3);F();
		C(65,403,1,2);C(0,16,0,1);C(0,16,1,8);C(0,65,1,2);F();
		C(2,20,0,1);C(0,8,0,4);C(76,417,1,5);C(14,258,0,1);C(7,14,1,1);C(0,26,1,6);C(5,9,1,3);F();
		C(82,434,1,5);C(19,54,1,7);C(4,253,1,8);F();
		C(105,671,1,7);C(0,31,0,4);C(10,57,0,2);C(2,18,1,3);F();
		C(1,11,1,7);C(0,153,1,6);C(8,718,1,4);C(6,712,1,5);F();
		return 2.256304;
	}

	double compute_indices_1 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,8,2,2);F();
		C(11,19,1,5);F();
		C(0,7,1,5);C(0,5,0,8);B();C(6,14,0,8);B();C(6,14,0,8);F();
		C(8,18,0,6);C(9,125,1,3);F();
		C(0,5,0,2);C(0,2,1,5);C(0,5,1,4);C(0,4,0,5);C(31,37,0,2);C(2,5,0,7);C(0,71,1,4);F();
		C(1,3,1,4);C(0,15,2,3);F();
		C(1,4,0,8);C(0,3,1,8);C(0,7,0,1);C(4,6,1,6);C(0,4,0,7);F();
		C(12,24,1,2);C(0,4,2,4);F();
		C(2,9,1,3);C(2,9,0,6);B();F();
		C(1,24,0,2);C(34,42,0,2);C(36,48,0,6);C(5,13,1,4);F();
		C(10,74,0,3);F();
		C(0,60,0,3);C(34,42,1,5);F();
		C(7,19,1,7);C(0,32,0,7);F();
		C(8,79,0,2);F();
		C(0,1,0,4);B();C(37,57,1,2);F();
		B();C(27,34,1,7);C(0,7,0,4);C(0,18,1,8);C(0,17,0,1);C(0,29,0,1);B();F();
		C(0,12,1,4);C(0,7,1,5);F();
		C(0,5,0,2);F();
		C(7,21,0,6);C(7,40,0,5);C(6,14,0,4);C(16,18,1,2);C(16,18,1,2);F();
		C(0,19,0,2);C(2,5,1,2);F();
		C(5,28,0,6);C(15,70,0,7);B();C(16,50,1,3);F();
		B();B();C(0,6,1,7);C(93,143,0,2);C(1,6,1,4);C(0,3,0,1);C(17,26,1,1);F();
		C(0,7,1,8);C(0,2,0,6);C(8,12,1,2);C(5,7,2,1);C(0,40,1,8);C(0,3,1,3);C(5,25,0,2);C(4,6,0,4);C(0,7,1,7);C(2,8,1,8);F();
		C(31,38,2,8);C(20,113,0,3);C(10,37,1,6);B();F();
		C(31,38,0,3);B();C(8,163,0,2);C(0,6,1,7);C(2,20,1,5);C(0,60,0,3);C(2,54,1,2);C(0,16,1,8);C(61,78,0,3);C(8,12,0,2);F();
		C(7,21,1,8);C(8,16,1,7);C(0,116,0,3);F();
		C(0,9,1,7);C(25,46,1,2);B();C(2,4,1,2);C(22,63,0,1);C(0,4,1,3);F();
		C(2,3,0,6);C(5,52,1,7);F();
		C(0,16,0,6);C(57,68,1,7);F();
		C(0,5,0,8);C(14,16,1,2);C(14,25,0,2);F();
		C(6,27,1,4);C(0,12,1,6);C(3,69,0,5);F();
		B();C(24,37,0,1);C(18,106,0,3);C(0,42,0,1);C(10,19,0,4);C(18,24,0,4);C(14,16,1,4);F();
		return 1.749574;
	}

	double compute_indices_2 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(83,225,1,2);F();
		C(16,24,1,6);B();C(8,96,0,4);F();
		C(37,51,0,4);C(3,6,0,3);B();F();
		C(0,49,1,1);C(0,27,1,7);C(36,60,1,8);C(60,63,2,6);C(0,2,0,1);C(1,6,0,4);C(17,26,1,3);C(0,4,0,5);B();C(0,40,0,7);C(0,7,0,2);C(0,1,1,8);B();C(0,4,1,7);C(15,17,0,6);F();
		C(0,6,0,1);B();F();
		C(31,38,1,1);F();
		C(0,1,0,5);C(29,33,1,5);C(10,28,0,7);F();
		B();C(0,22,1,2);C(5,23,0,4);F();
		C(0,10,1,1);C(0,14,0,5);C(21,42,0,1);C(10,28,1,7);C(0,5,0,4);F();
		C(14,36,1,6);C(21,28,0,4);C(0,1,0,4);B();C(0,15,1,8);C(16,159,1,7);F();
		C(1,11,0,7);C(30,164,0,6);F();
		C(0,155,1,5);C(33,161,0,1);F();
		C(25,46,0,2);F();
		C(2,162,1,2);C(6,7,1,3);F();
		C(0,9,1,3);C(0,13,1,6);C(0,3,0,2);C(83,225,1,5);B();C(24,31,1,7);C(7,30,1,4);C(0,9,1,3);C(24,31,1,7);C(7,30,1,4);F();
		C(3,10,0,3);C(0,19,1,4);C(29,53,1,3);C(0,1,0,2);C(0,14,1,3);F();
		C(56,86,1,5);C(2,4,1,2);C(22,63,1,6);C(0,3,0,8);F();
		C(54,74,0,8);C(14,258,0,1);F();
		C(55,87,0,8);C(0,13,1,6);C(0,38,0,6);C(0,153,1,6);F();
		C(2,5,0,7);B();C(23,31,1,1);F();
		C(24,31,1,6);C(3,4,1,3);C(6,14,1,3);F();
		C(21,32,0,3);C(0,26,0,7);F();
		B();C(0,9,0,3);F();
		C(27,83,1,1);C(19,30,1,2);F();
		C(0,98,0,7);C(2,5,1,4);C(0,1,1,3);C(66,139,1,3);C(2,20,0,1);C(8,15,0,4);B();F();
		C(0,15,0,5);C(34,42,0,2);C(0,1,1,6);C(0,15,0,5);F();
		C(0,26,1,1);C(38,52,1,8);F();
		C(0,12,1,7);C(32,69,0,7);F();
		C(2,11,1,2);C(64,101,1,7);F();
		C(59,75,0,7);C(4,5,0,7);F();
		C(11,30,1,1);C(0,255,1,1);F();
		C(21,42,0,1);C(8,11,0,2);C(0,10,1,4);C(9,57,1,3);C(2,5,0,4);C(2,56,1,8);B();F();
		return 2.985028;
	}

	double compute_indices_3 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,66,0,2);C(2,20,0,1);C(2,4,1,2);F();
		B();C(0,91,1,8);F();
		C(0,2,0,1);C(17,38,0,2);C(2,4,1,2);C(0,2,1,5);F();
		B();C(0,4,1,5);C(0,1,1,1);C(7,40,0,4);C(8,9,0,4);C(1,8,0,7);C(82,434,0,5);C(111,147,0,3);C(0,255,0,8);C(8,12,1,4);F();
		C(15,103,0,1);C(0,4,0,5);C(3,15,1,1);C(12,18,0,3);B();C(0,12,1,7);C(46,147,1,1);C(72,260,1,3);F();
		C(16,21,1,2);C(97,844,1,4);C(129,653,1,2);F();
		C(83,225,1,2);F();
		C(3,8,1,2);C(20,113,0,3);C(5,7,2,1);C(4,6,0,4);C(0,7,1,7);C(5,7,2,1);F();
		C(2,9,0,2);C(2,8,0,8);F();
		C(93,141,0,3);C(5,9,1,4);C(0,20,0,7);C(29,53,1,3);C(0,1,0,2);C(3,10,1,5);C(77,264,1,6);F();
		C(6,27,0,5);C(9,91,0,4);F();
		C(0,10,1,4);C(8,163,1,3);F();
		C(0,259,1,8);C(4,6,0,4);F();
		C(0,2,2,4);F();
		C(3,7,0,5);C(0,11,1,6);C(9,221,1,6);C(0,11,1,3);C(2,10,0,5);F();
		C(1,61,0,2);C(0,10,1,4);C(3,8,0,5);C(57,503,1,2);C(1,11,1,3);F();
		C(0,20,1,5);C(48,76,1,5);C(27,51,1,4);C(85,420,1,5);F();
		C(2,13,1,3);C(0,2,0,2);B();B();C(11,22,1,7);F();
		C(0,15,0,8);C(48,63,2,5);C(11,57,0,3);F();
		C(2,5,0,4);C(50,54,2,5);C(0,65,1,6);C(118,645,1,4);C(0,10,1,8);F();
		C(59,711,1,2);C(6,24,1,1);F();
		C(0,18,1,7);C(0,275,1,5);C(6,47,0,8);F();
		C(60,142,1,4);C(0,42,1,7);C(17,26,1,3);C(0,2,1,5);C(0,102,0,3);C(0,18,0,5);B();B();F();
		C(2,55,0,1);F();
		C(61,78,0,3);C(0,3,0,8);C(0,4,0,2);F();
		C(10,12,1,4);C(0,5,0,2);C(56,86,1,5);C(22,63,0,6);C(8,163,1,2);C(6,10,1,3);C(0,8,1,2);C(0,3,1,5);C(0,5,0,7);C(90,826,0,8);C(10,20,0,5);C(5,9,1,4);C(6,29,0,2);F();
		B();C(39,85,1,2);B();C(1,5,0,4);C(24,31,1,6);C(46,147,1,1);F();
		C(14,30,0,7);C(9,18,0,5);C(1,6,0,7);F();
		C(0,8,2,7);C(0,3,0,3);C(15,164,1,8);C(6,21,0,5);C(8,15,0,7);C(17,22,1,2);B();C(9,17,0,1);C(1,12,1,4);C(114,657,1,7);F();
		C(46,147,1,1);C(62,64,0,5);C(82,264,1,3);C(18,712,1,4);F();
		C(46,147,1,1);C(62,64,0,5);C(84,263,1,2);C(6,11,1,4);C(7,20,0,4);C(0,36,0,7);C(0,75,1,4);C(62,64,0,5);C(0,13,1,3);F();
		C(0,40,1,2);F();
		return 3.703292;
	}

	double compute_indices_4 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(4,37,1,3);F();
		C(2,7,1,7);F();
		C(3,110,1,2);F();
		C(0,10,0,2);B();F();
		C(0,14,0,2);C(0,1,0,8);C(0,2,1,5);F();
		C(2,16,1,8);F();
		C(0,17,0,8);C(9,18,0,3);C(0,7,1,7);F();
		C(11,20,1,5);F();
		C(0,5,0,4);C(17,28,1,5);F();
		C(0,5,0,1);C(2,9,0,2);C(11,16,1,3);C(8,19,0,8);C(13,39,0,3);C(0,3,0,5);F();
		B();F();
		C(0,6,0,1);F();
		C(14,27,1,3);C(0,7,0,7);F();
		C(0,6,0,7);F();
		C(0,6,1,6);C(25,104,1,3);C(102,863,0,1);F();
		C(18,24,0,4);C(18,24,0,4);C(12,105,0,5);C(18,24,0,4);C(0,40,0,7);C(121,785,1,5);F();
		C(38,46,0,2);C(18,36,1,6);C(0,253,0,4);F();
		B();C(0,260,0,1);C(7,10,0,4);C(11,46,0,3);F();
		C(0,3,1,2);C(31,75,1,8);C(0,281,0,6);C(0,275,1,5);C(43,142,1,1);F();
		C(19,28,0,4);F();
		C(36,48,1,7);C(82,434,1,5);F();
		C(31,71,0,4);F();
		C(1,715,0,5);C(37,57,0,6);F();
		C(0,10,1,8);C(8,718,0,4);C(42,43,1,3);C(0,6,1,2);C(6,14,0,3);C(0,6,1,2);C(121,785,1,5);C(6,14,0,8);C(6,11,0,8);F();
		C(6,29,0,2);C(0,5,0,4);C(70,338,1,3);C(5,9,1,4);C(0,6,0,4);C(70,338,1,3);C(114,806,0,8);C(0,5,0,7);C(10,20,0,5);F();
		C(14,17,1,1);C(37,57,1,3);F();
		C(0,204,1,5);C(105,808,1,6);C(6,11,0,2);C(0,204,1,5);C(0,206,1,5);C(17,26,0,3);C(0,4,1,1);C(2,5,2,3);C(8,163,1,8);C(34,42,0,2);C(0,24,0,8);F();
		C(3,8,1,4);C(97,844,0,5);C(4,5,1,8);C(12,18,0,2);B();C(0,5,0,5);C(13,23,0,4);B();C(18,25,0,4);F();
		C(59,84,0,5);C(97,844,0,2);F();
		C(3,8,1,3);C(3,8,0,1);C(97,844,1,4);C(31,37,0,2);C(0,10,0,8);C(31,37,0,2);C(3,8,1,3);C(78,159,0,2);C(2,5,0,4);C(0,42,0,8);B();F();
		C(97,844,0,5);C(0,4,1,7);C(0,4,0,1);C(0,109,0,3);F();
		C(31,38,0,3);C(31,38,0,3);C(0,2,0,5);C(106,872,0,1);F();
		return 6.362405;
	}

	double compute_indices_5 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,70,0,3);C(0,16,1,8);C(24,30,1,2);C(8,79,0,2);C(0,43,1,7);C(2,20,0,1);C(7,14,1,1);F();
		B();C(0,2,1,5);C(0,208,1,6);F();
		C(60,365,1,3);F();
		C(2,16,0,8);F();
		C(0,5,1,2);C(0,4,0,5);F();
		B();C(20,108,1,3);C(0,6,1,7);B();C(0,255,1,4);F();
		B();F();
		C(3,8,1,5);C(9,16,0,7);B();B();C(99,142,0,2);F();
		B();C(0,8,0,6);F();
		C(0,1,1,5);F();
		B();C(5,43,1,3);F();
		C(82,264,1,3);C(2,20,0,1);C(0,43,1,7);C(1,12,1,4);C(0,70,0,3);C(0,1,1,8);C(0,16,1,8);C(8,79,0,2);C(24,30,1,2);C(7,14,1,1);C(0,55,0,3);F();
		C(2,5,2,3);F();
		C(8,163,1,8);C(34,42,0,4);F();
		C(23,31,0,4);C(76,417,1,1);F();
		B();C(15,24,1,8);C(0,28,1,7);C(14,21,0,1);B();C(0,18,1,3);F();
		C(0,275,1,5);C(0,255,1,6);C(0,15,1,3);F();
		C(0,3,1,8);C(1,6,1,4);B();C(17,26,1,3);C(0,4,1,1);C(0,14,1,4);C(0,249,0,7);C(0,2,0,4);F();
		C(8,14,1,3);C(5,13,1,4);C(6,27,0,4);C(82,434,1,5);C(52,63,1,7);C(7,19,1,2);F();
		C(0,79,0,4);C(97,844,1,1);F();
		C(59,711,1,2);C(0,3,1,8);C(0,8,1,2);C(0,116,0,6);F();
		C(0,12,0,7);C(0,40,0,4);C(48,63,1,5);C(66,139,0,3);C(10,54,1,2);C(4,6,0,4);C(0,14,1,8);F();
		C(0,2,2,7);F();
		C(80,498,1,2);F();
		C(2,55,0,1);C(0,44,1,7);C(41,63,1,7);F();
		C(36,60,1,8);C(62,64,0,5);C(2,4,0,2);C(9,18,1,8);F();
		C(39,85,0,2);C(0,1,1,4);F();
		B();C(18,106,0,7);F();
		C(29,33,1,1);C(2,9,0,1);C(12,119,0,3);C(61,69,0,2);B();B();C(0,17,1,6);F();
		C(0,208,0,5);C(0,14,1,4);C(1,6,1,2);C(0,14,1,4);C(3,9,1,4);C(0,255,1,8);F();
		B();B();C(62,64,0,1);C(11,23,1,1);C(46,147,0,1);F();
		C(6,12,1,4);C(9,221,1,6);F();
		return 4.162816;
	}

	double compute_indices_6 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,7,1,4);C(0,16,1,8);C(61,78,0,3);C(15,27,1,7);C(8,12,0,2);C(1,22,1,2);B();B();C(61,78,0,3);C(0,3,0,3);C(15,164,1,8);C(6,21,0,5);C(8,15,0,7);B();C(9,17,0,1);C(1,12,1,4);C(114,657,1,7);F();
		C(2,16,1,1);C(30,67,1,4);C(0,6,1,1);C(0,4,0,3);C(3,214,1,2);C(10,12,0,6);C(58,702,1,2);C(56,346,1,4);F();
		B();C(0,2,0,1);C(0,4,0,1);C(36,54,0,2);B();F();
		C(114,657,1,7);C(0,4,0,5);C(71,90,0,7);F();
		C(17,150,1,2);F();
		C(1,264,0,2);C(1,264,0,2);C(12,268,1,2);F();
		C(4,19,0,7);C(14,21,0,2);C(7,44,1,2);C(0,11,0,1);C(0,11,0,1);C(82,434,0,5);F();
		B();C(12,95,0,2);C(0,6,1,7);C(17,26,1,3);C(0,2,1,5);C(1,6,0,7);C(0,18,0,5);B();F();
		C(0,243,0,2);C(0,253,0,8);C(0,4,0,7);C(2,15,1,8);C(0,6,0,6);F();
		B();C(46,147,0,8);C(82,434,1,5);F();
		C(0,260,0,1);F();
		C(0,260,0,1);C(80,868,1,2);F();
		C(14,26,1,5);C(0,7,0,6);C(0,3,1,5);C(0,8,1,2);C(0,5,0,7);C(90,826,0,8);C(0,16,1,3);C(5,9,1,4);C(6,29,0,2);F();
		C(6,27,0,3);C(20,28,0,2);F();
		C(0,36,0,6);C(18,712,1,4);F();
		B();C(0,24,1,3);C(0,61,0,6);C(0,1,0,8);C(0,1,1,6);C(13,38,0,7);C(53,61,0,4);C(24,37,1,1);C(40,42,1,1);F();
		C(123,635,0,1);C(0,4,1,4);B();C(24,31,1,6);F();
		C(0,8,0,8);F();
		C(11,30,0,4);C(121,785,0,2);C(56,346,1,3);F();
		C(0,698,0,1);B();C(7,11,0,3);F();
		C(0,698,0,1);C(38,65,1,2);F();
		C(106,872,0,1);F();
		C(0,14,0,8);C(0,7,0,3);C(1,8,0,4);F();
		C(0,116,0,1);F();
		C(13,15,0,3);C(0,105,1,6);C(12,268,1,2);F();
		C(8,14,0,2);C(15,74,0,1);C(0,14,0,8);F();
		C(0,6,0,3);C(0,6,0,3);C(33,90,0,2);C(0,16,0,1);F();
		C(2,6,1,1);B();C(0,40,0,8);C(1,12,1,4);C(22,25,1,6);F();
		B();C(0,2,0,1);C(0,119,1,3);C(0,2,2,3);C(0,5,1,7);C(80,498,1,2);C(8,163,1,3);C(46,147,1,1);F();
		C(8,33,0,8);C(0,4,1,7);C(61,78,0,3);C(0,31,0,5);C(0,208,1,5);C(0,14,1,4);C(1,6,1,2);C(29,30,1,8);F();
		C(0,14,1,7);C(46,61,0,4);B();C(31,37,0,2);C(103,266,0,3);C(25,31,1,2);C(8,20,0,7);C(85,420,1,6);F();
		C(0,17,0,6);C(2,6,1,1);C(114,650,0,2);C(114,650,0,2);C(0,153,1,6);C(73,445,1,5);F();
		return 1.225147;
	}

	double compute_indices_7 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		B();B();F();
		C(3,9,0,4);B();F();
		C(3,9,0,6);B();F();
		C(0,15,0,7);F();
		C(0,13,0,5);C(0,17,0,5);F();
		C(0,2,1,1);C(14,21,1,3);F();
		C(3,21,0,5);C(14,34,1,2);F();
		C(0,13,0,7);C(10,26,1,1);C(0,2,0,4);C(8,718,0,4);F();
		C(7,31,0,4);F();
		C(31,38,1,7);F();
		C(5,13,2,5);C(15,16,1,4);C(1,8,0,1);C(0,4,1,3);B();C(0,17,1,4);C(1,36,1,8);F();
		C(1,9,1,4);F();
		B();C(0,44,1,7);C(0,26,1,1);F();
		C(0,4,1,3);C(0,1,0,1);C(0,1,1,4);B();F();
		C(8,12,0,5);F();
		C(9,16,1,3);C(39,85,0,2);C(0,8,0,8);F();
		B();B();C(93,143,0,2);C(0,4,1,6);C(17,26,1,1);F();
		C(0,208,1,5);F();
		C(15,63,1,6);C(43,154,0,1);C(6,12,1,6);F();
		C(0,40,1,4);C(30,40,0,2);C(0,40,0,4);C(2,7,1,1);C(0,40,1,4);B();C(2,55,1,6);C(0,16,0,8);C(37,60,0,2);C(2,162,1,1);C(40,54,0,4);F();
		C(0,55,0,3);C(0,208,0,5);C(1,6,1,2);C(2,6,0,5);C(0,4,1,7);F();
		C(83,225,0,5);F();
		C(0,230,0,4);F();
		C(0,246,0,5);F();
		C(0,4,0,2);C(7,41,1,3);B();C(2,252,1,2);C(2,252,1,1);C(0,43,0,8);C(0,232,1,1);F();
		C(57,82,0,1);C(2,9,1,5);C(0,253,1,4);C(0,4,0,7);C(2,15,1,8);F();
		C(25,46,0,2);C(56,86,1,5);C(8,22,1,2);C(22,63,0,6);C(14,258,0,1);C(0,38,0,8);C(22,35,0,5);C(62,66,0,7);C(0,2,1,6);C(0,14,1,2);C(0,2,0,8);C(6,98,1,3);C(0,41,1,3);B();C(0,17,0,4);C(7,14,1,1);C(14,46,0,2);C(25,37,1,4);F();
		C(22,24,0,7);B();C(62,64,0,5);C(82,264,1,3);C(0,1,1,1);C(16,17,0,1);F();
		C(0,22,0,5);C(0,12,0,8);C(0,16,1,8);C(0,65,1,2);C(65,403,1,2);F();
		C(0,6,0,8);C(111,651,1,2);C(35,55,2,7);C(0,2,1,2);F();
		C(0,153,1,3);C(114,650,1,2);C(6,712,1,5);F();
		C(82,434,1,5);C(114,650,1,2);C(0,153,1,6);C(16,159,1,7);F();
		return 6.014423;
	}

	double compute_indices_8 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,9,1,3);F();
		C(3,11,0,7);F();
		C(9,23,1,5);F();
		B();C(1,24,1,4);C(7,11,0,3);C(0,17,1,8);F();
		C(0,4,1,7);C(6,13,1,2);C(5,9,1,3);C(8,26,0,1);C(0,4,0,3);F();
		C(0,14,0,5);C(6,27,1,4);C(2,8,0,4);B();C(7,19,0,2);C(6,29,0,7);C(6,11,1,4);C(0,13,1,4);F();
		C(17,32,1,6);F();
		C(29,33,1,1);F();
		C(0,8,1,7);B();C(0,36,0,7);F();
		C(0,8,0,3);C(0,38,1,8);C(3,37,1,4);C(0,9,1,4);F();
		C(0,40,0,6);F();
		C(22,47,0,3);C(0,24,1,2);C(5,17,1,6);F();
		C(0,6,1,4);C(9,51,0,3);C(9,18,0,5);C(0,16,1,7);F();
		B();C(0,6,0,8);B();C(22,63,1,1);F();
		C(5,65,0,1);F();
		C(0,65,1,6);C(0,6,2,7);C(9,15,0,4);F();
		C(0,15,1,6);C(0,7,1,8);C(7,14,0,1);C(52,70,1,5);C(8,17,0,4);C(40,42,0,5);F();
		B();C(0,5,1,4);C(0,19,0,2);C(31,70,0,3);C(0,25,1,3);C(22,28,1,7);C(0,4,0,2);C(0,55,0,3);C(0,54,0,4);F();
		C(7,20,1,2);C(31,38,1,3);C(0,75,0,1);F();
		C(2,5,0,4);C(56,86,1,5);B();C(23,31,0,1);C(5,43,0,2);F();
		C(0,20,1,3);C(18,106,0,3);C(6,7,1,2);B();F();
		C(0,14,1,2);C(46,147,1,1);C(0,16,0,1);F();
		C(0,260,0,1);F();
		C(1,264,0,8);C(0,13,0,5);F();
		C(0,30,1,1);C(2,55,0,7);C(0,2,0,5);C(82,264,0,4);C(57,83,0,2);C(0,21,1,1);C(18,24,0,1);C(0,2,2,8);C(6,19,1,7);C(0,42,0,5);C(8,163,1,2);F();
		B();B();C(8,101,0,3);C(0,17,1,6);C(5,43,0,3);C(10,264,0,1);F();
		C(0,253,1,4);C(0,14,0,4);C(82,264,1,3);F();
		C(82,264,0,3);C(8,12,1,1);C(62,64,0,5);C(7,19,0,3);F();
		C(9,53,1,6);C(0,2,0,2);C(0,15,1,5);C(2,9,1,4);C(0,24,0,5);C(14,266,0,6);F();
		C(9,17,0,7);C(119,806,0,7);C(10,11,0,4);F();
		C(0,281,1,5);C(97,844,0,5);F();
		C(97,844,0,2);B();C(31,38,1,3);C(0,11,1,8);C(0,7,0,6);C(0,21,0,1);C(41,52,1,4);C(0,30,1,2);C(6,14,1,1);C(2,21,0,7);C(4,35,1,3);F();
		return 5.395354;
	}

	double compute_indices_9 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,65,1,2);C(0,13,0,6);C(0,60,0,3);F();
		C(58,82,1,7);C(8,39,0,4);C(2,55,1,1);C(22,31,1,4);F();
		C(0,2,0,1);C(41,57,1,8);C(37,57,1,2);C(0,26,0,6);C(46,147,0,6);F();
		C(0,68,1,4);F();
		C(0,260,1,1);F();
		B();C(5,9,0,6);F();
		C(0,8,1,2);C(0,3,1,8);C(0,15,1,3);F();
		C(0,68,1,8);F();
		C(24,37,0,1);C(18,106,0,3);C(0,42,0,1);C(10,19,0,4);C(18,24,0,4);B();C(0,255,0,1);F();
		C(0,16,1,8);F();
		C(0,7,0,8);F();
		C(0,12,1,2);F();
		C(3,9,0,8);F();
		C(8,12,1,4);C(14,16,1,4);C(9,16,1,3);F();
		C(0,2,1,3);C(38,65,0,2);C(11,20,1,1);C(3,89,0,8);F();
		C(16,20,0,2);C(0,18,1,3);B();C(25,46,0,2);C(0,1,0,4);C(4,38,1,3);C(0,14,1,5);C(1,4,0,4);F();
		C(4,8,0,1);C(0,4,1,6);C(0,3,0,3);F();
		C(0,14,1,5);C(8,79,0,2);F();
		C(0,116,1,3);C(13,23,0,4);F();
		C(2,55,1,6);C(0,5,1,1);C(0,1,0,1);C(8,39,1,1);F();
		C(0,78,0,8);C(0,17,1,1);C(0,54,1,3);C(5,8,0,2);C(0,2,2,4);F();
		C(34,42,1,2);F();
		B();C(6,712,1,5);C(43,62,0,3);F();
		C(0,28,1,4);C(23,40,0,7);C(0,34,1,4);F();
		C(1,11,1,4);C(3,8,1,6);F();
		C(54,73,1,7);C(22,63,0,1);C(89,253,0,3);C(0,4,0,2);C(0,16,1,8);C(0,65,1,2);C(11,24,1,2);F();
		C(3,8,1,5);C(39,85,0,2);C(9,24,0,1);C(0,116,0,3);C(14,21,0,6);F();
		C(48,183,1,7);C(1,3,1,6);C(41,53,1,1);C(61,78,0,8);C(0,1,1,6);C(42,62,0,2);C(0,11,1,6);C(1,41,1,1);C(0,20,1,6);C(36,100,0,5);C(36,48,0,2);F();
		B();C(0,19,1,7);C(0,17,1,6);C(0,2,0,6);F();
		C(0,14,1,8);C(0,208,1,5);C(1,6,1,2);C(0,4,1,7);F();
		C(46,147,1,1);B();C(62,64,0,5);F();
		C(114,650,1,2);C(0,153,1,6);C(82,434,1,5);C(0,156,1,8);C(85,420,1,5);C(0,11,0,6);C(0,30,1,1);C(0,9,0,4);C(8,12,0,1);C(0,6,1,8);C(0,255,1,2);F();
		return 5.192714;
	}

	double compute_indices_10 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(9,18,0,7);C(48,63,2,5);F();
		B();C(0,2,2,3);C(9,25,1,4);F();
		C(2,9,1,1);F();
		C(0,3,0,4);F();
		C(0,5,1,2);C(0,14,0,6);C(0,17,1,3);F();
		C(0,6,0,4);F();
		C(0,15,1,5);C(0,2,0,4);C(0,4,0,5);C(0,1,1,6);F();
		C(18,19,1,7);C(0,37,1,7);F();
		C(2,9,0,2);B();C(0,8,1,5);C(0,3,0,2);F();
		C(1,22,0,7);C(16,25,1,5);F();
		C(9,16,1,1);B();C(7,8,0,3);C(0,19,0,1);C(0,5,1,1);C(7,10,0,4);C(3,19,1,1);C(6,15,1,3);F();
		C(5,7,0,5);C(1,12,1,5);C(0,4,1,1);C(21,27,1,4);F();
		C(0,17,0,8);C(0,14,0,5);C(0,21,1,1);C(0,1,0,1);F();
		C(13,23,1,4);C(15,16,0,2);F();
		C(36,48,0,7);F();
		C(0,18,1,3);C(0,3,0,4);C(0,7,1,8);C(0,6,0,3);C(0,2,0,5);C(0,4,1,7);F();
		C(9,13,0,7);C(1,12,1,4);F();
		C(0,1,1,1);C(0,4,1,4);C(17,26,0,3);B();C(0,4,1,1);F();
		C(0,60,1,3);F();
		C(64,71,0,7);F();
		C(24,31,1,7);F();
		C(31,38,1,3);B();C(31,37,0,8);F();
		C(34,42,1,2);F();
		C(0,2,0,1);C(31,38,1,3);C(11,20,1,3);F();
		C(0,3,0,5);C(0,36,0,1);F();
		C(5,10,1,2);F();
		C(16,17,0,6);C(6,28,0,4);C(8,20,1,6);F();
		C(30,40,0,2);F();
		B();C(21,27,1,6);C(22,28,0,7);F();
		C(64,71,1,2);C(1,6,1,3);F();
		C(50,55,1,1);B();F();
		C(7,13,0,5);C(0,16,1,8);F();
		return 3.152694;
	}

	double compute_indices_11 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(14,31,0,4);F();
		B();F();
		B();C(0,2,2,8);C(0,2,1,4);F();
		C(0,4,1,4);C(0,2,0,3);C(0,2,0,3);F();
		C(0,4,1,8);C(6,8,1,1);C(1,12,1,4);C(8,88,1,3);F();
		C(0,15,0,8);B();F();
		C(8,12,1,8);C(9,16,0,3);C(15,16,1,3);C(9,16,1,5);F();
		C(6,20,1,7);C(2,18,1,2);C(0,7,0,2);C(9,21,0,7);B();F();
		C(3,19,1,5);C(0,16,0,1);C(1,6,1,4);B();C(17,26,0,3);C(0,5,1,7);B();C(2,20,1,1);F();
		C(11,28,1,6);C(0,2,1,5);B();B();C(0,6,1,7);F();
		C(11,246,0,8);F();
		C(9,18,0,7);C(1,11,1,3);C(24,31,0,1);B();C(0,12,1,2);C(16,20,1,1);F();
		B();C(18,24,0,4);C(4,20,1,4);F();
		C(31,37,0,4);C(0,2,1,7);C(31,38,1,3);C(0,8,0,6);F();
		C(20,27,0,4);C(23,31,1,1);F();
		C(5,43,0,1);F();
		C(28,40,1,4);F();
		C(0,46,0,8);C(4,5,0,8);F();
		C(6,47,1,1);F();
		C(0,9,0,2);F();
		C(0,4,0,1);C(37,49,1,5);F();
		C(22,52,0,1);C(2,4,1,1);C(22,52,0,1);C(2,4,1,1);F();
		C(46,54,0,8);C(19,42,1,2);F();
		C(71,90,0,7);F();
		C(8,20,0,2);C(41,63,2,7);C(8,18,1,2);C(0,281,1,6);F();
		B();C(6,25,1,2);C(0,4,1,5);C(0,65,1,2);C(0,11,1,4);C(0,18,1,8);C(0,6,1,5);F();
		C(49,69,1,7);F();
		C(10,71,1,2);B();F();
		C(0,36,1,2);C(0,71,0,1);C(0,2,0,6);C(0,60,0,3);C(9,19,1,1);C(0,16,1,8);F();
		C(46,54,1,8);F();
		C(0,14,0,5);C(25,44,0,3);F();
		C(68,808,1,5);C(57,503,1,4);C(27,38,1,7);C(76,417,1,1);C(11,729,0,2);C(0,4,1,5);C(2,13,1,5);C(2,13,0,5);F();
		return 2.879463;
	}

	double compute_indices_12 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(106,872,0,1);F();
		C(0,1,0,1);F();
		C(0,4,1,4);C(0,2,0,4);F();
		C(0,4,0,5);F();
		C(13,16,1,6);C(51,60,0,8);F();
		C(5,8,0,2);F();
		C(2,12,1,1);C(2,9,0,2);C(2,9,1,4);C(21,56,0,6);F();
		B();C(6,11,1,4);F();
		C(0,16,1,8);C(2,7,0,8);C(10,11,1,2);F();
		C(0,18,0,2);F();
		C(13,22,1,4);C(7,19,0,4);C(1,9,1,3);C(9,16,0,3);F();
		C(18,24,0,4);C(0,3,1,1);F();
		C(1,10,0,6);C(17,26,1,7);C(0,15,1,4);C(0,15,1,4);C(0,13,1,6);C(0,13,1,3);F();
		C(4,6,0,7);C(5,13,1,4);C(7,19,0,2);F();
		B();C(6,27,0,4);C(27,30,1,6);F();
		C(27,38,1,2);C(32,48,1,2);C(31,38,1,3);B();C(14,31,0,7);C(6,11,1,3);F();
		C(34,42,1,2);B();F();
		C(0,44,0,1);C(0,5,1,2);F();
		B();C(8,39,1,4);C(2,55,0,1);C(0,28,0,8);F();
		C(25,46,1,2);C(0,40,1,4);C(0,40,0,4);C(21,32,0,2);C(55,63,2,2);F();
		C(39,85,0,2);B();F();
		C(52,100,0,5);C(4,9,0,2);C(22,63,0,1);F();
		C(0,17,1,6);C(0,116,0,3);C(0,12,0,2);B();C(0,22,1,4);F();
		C(7,22,0,3);C(3,8,0,5);C(88,136,1,2);F();
		C(0,14,0,5);C(0,14,0,1);C(59,146,0,3);F();
		C(7,41,0,1);F();
		C(53,61,1,1);F();
		C(1,6,1,2);C(0,14,0,1);C(0,6,1,4);C(0,208,1,5);C(0,14,1,4);C(0,4,1,7);C(0,14,0,5);C(59,146,0,3);C(0,5,0,5);C(15,25,0,7);F();
		C(2,19,0,5);C(0,287,0,6);C(0,244,0,5);C(0,242,1,2);F();
		C(0,281,0,6);C(0,12,1,1);C(0,18,1,3);C(0,23,1,3);F();
		C(85,420,1,5);F();
		C(82,434,0,5);C(119,663,1,2);C(6,712,1,5);C(8,718,1,4);C(1,15,0,8);C(11,240,1,6);C(9,221,1,6);C(0,4,1,7);C(0,12,1,1);F();
		return 4.625642;
	}

	double compute_indices_13 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		B();C(3,37,1,4);C(82,434,1,2);B();F();
		C(59,711,1,2);C(6,24,1,1);C(14,25,0,4);F();
		C(0,5,0,2);C(111,144,1,6);C(0,208,1,6);C(0,15,1,4);C(76,417,1,5);C(0,1,0,4);C(0,11,1,1);C(8,163,1,3);F();
		C(38,350,1,2);F();
		C(82,434,1,5);C(0,710,1,1);C(0,153,1,6);C(114,650,1,2);F();
		C(97,844,1,2);F();
		C(22,63,0,1);C(36,60,1,8);C(60,63,2,6);C(0,233,1,2);C(11,20,1,2);C(4,20,1,4);C(9,11,1,1);C(105,851,1,3);F();
		C(0,5,1,7);C(80,498,1,2);C(46,147,1,1);C(0,14,1,1);F();
		C(5,46,0,7);C(1,252,0,1);C(12,268,1,2);C(45,351,1,2);C(0,2,0,1);C(0,3,1,1);F();
		C(14,259,1,6);B();C(97,844,1,3);C(1,12,1,4);C(111,646,1,7);C(0,6,1,3);F();
		C(14,726,1,5);F();
		C(1,6,1,4);C(0,4,1,7);B();B();C(76,417,1,5);C(9,91,1,5);C(0,9,0,2);C(0,59,1,1);C(36,48,0,4);C(1,41,1,1);C(0,20,1,6);C(36,100,0,5);C(36,48,0,2);F();
		C(17,26,0,6);C(6,142,1,4);C(0,253,1,7);C(76,417,1,1);F();
		C(0,153,1,3);C(114,650,0,2);C(8,718,1,4);C(6,712,1,5);C(0,21,1,1);C(8,12,1,1);C(0,7,0,3);C(0,15,0,8);C(0,64,0,8);F();
		C(12,253,1,4);C(12,18,0,2);C(0,9,1,6);C(42,62,0,2);C(0,18,1,5);C(1,844,1,3);C(8,15,0,4);F();
		C(82,434,1,5);C(8,13,1,8);C(18,176,1,8);C(9,16,1,3);C(18,255,1,5);C(3,246,1,8);C(5,10,0,8);C(3,89,0,8);C(8,12,1,4);C(18,19,1,5);C(80,498,0,2);F();
		C(12,31,0,4);C(0,3,1,1);C(120,838,1,6);C(22,31,0,4);C(0,4,2,6);C(0,208,1,6);F();
		C(0,6,0,8);B();F();
		C(15,17,1,7);C(22,253,1,4);C(2,5,2,3);C(0,230,0,4);C(0,18,1,1);C(0,43,1,5);B();C(51,60,0,8);C(76,417,1,1);F();
		C(79,415,0,1);B();C(57,503,1,2);F();
		C(22,24,1,8);C(31,38,1,7);C(0,715,1,3);F();
		C(0,3,1,7);C(0,54,1,2);C(111,144,1,6);C(82,434,1,2);C(5,13,0,3);F();
		C(34,42,0,4);C(0,3,0,4);C(15,21,1,2);C(0,2,0,2);C(83,225,1,2);C(2,6,1,7);F();
		C(0,255,1,5);C(0,31,1,1);C(0,255,1,5);C(2,5,2,3);C(0,255,1,5);C(0,253,1,8);C(3,246,0,4);C(1,8,1,1);C(56,346,1,3);C(6,25,0,4);F();
		C(12,27,1,4);C(18,21,0,8);C(10,711,0,7);C(8,163,0,3);F();
		C(64,101,1,7);C(114,650,1,2);F();
		C(0,31,1,4);C(54,704,1,2);C(3,214,1,2);C(56,346,1,4);C(0,18,1,3);F();
		C(0,259,1,3);C(97,844,1,5);F();
		C(61,69,0,5);C(0,59,0,1);C(31,38,0,8);B();C(31,37,0,2);C(2,6,0,7);F();
		C(97,844,1,3);C(5,8,0,1);C(9,266,1,8);C(5,8,0,1);C(0,8,2,4);C(123,646,1,5);C(81,429,1,4);F();
		C(0,3,1,8);C(0,1,0,1);C(46,147,0,1);B();C(2,55,1,6);C(0,5,1,1);C(8,39,1,1);C(0,5,0,2);C(34,42,1,2);F();
		C(80,498,0,2);F();
		return 1.344119;
	}

	double compute_indices_14 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(9,12,0,3);C(20,26,0,6);C(0,7,0,2);C(0,69,1,3);C(61,78,0,3);F();
		C(0,3,1,8);C(0,255,1,2);C(0,5,1,3);C(14,25,0,4);C(7,14,1,6);F();
		B();C(99,142,0,2);C(6,12,1,5);C(0,2,0,1);F();
		B();C(0,4,1,8);C(10,38,0,5);C(14,24,1,3);C(0,44,0,3);C(0,8,0,1);C(0,12,1,6);B();C(12,24,0,3);C(0,6,0,7);C(8,20,0,3);C(11,19,1,5);F();
		C(5,16,1,6);C(0,4,0,6);B();C(10,26,1,6);C(12,18,0,5);C(0,6,0,5);C(2,6,1,1);F();
		C(0,5,2,8);C(0,12,0,8);C(6,14,1,1);F();
		C(64,264,1,5);C(76,417,1,5);B();B();C(8,12,0,4);C(0,17,0,6);B();C(0,116,1,3);C(105,851,1,3);F();
		C(0,5,0,2);C(57,503,1,2);C(1,11,1,3);F();
		C(0,11,0,6);C(0,6,1,6);F();
		C(0,14,1,7);C(15,63,1,6);C(0,44,1,7);C(10,22,1,4);C(71,884,1,2);C(0,2,0,1);C(0,2,0,1);C(2,4,1,2);F();
		C(8,12,1,4);C(15,16,0,7);C(0,249,1,2);F();
		C(1,12,1,6);C(0,20,0,4);C(36,48,0,7);C(46,147,1,1);F();
		C(0,14,1,6);C(0,40,1,2);F();
		C(9,18,0,3);C(14,56,1,2);C(8,163,0,8);C(0,6,1,7);C(0,2,1,4);C(0,7,1,1);C(0,28,0,1);F();
		C(0,28,0,1);C(61,78,0,8);C(53,61,1,8);C(6,52,1,8);C(42,62,0,2);C(0,18,1,5);C(41,53,1,1);C(4,38,1,1);C(1,844,1,3);F();
		C(0,232,1,1);C(59,146,0,3);C(27,51,1,4);F();
		C(35,81,0,2);C(29,33,1,5);C(22,31,0,7);B();F();
		C(9,16,1,8);C(46,147,1,5);C(10,74,0,3);C(2,20,0,1);C(0,16,1,8);C(7,21,1,7);F();
		C(0,10,1,1);C(7,41,0,1);F();
		C(2,5,1,5);C(0,15,1,5);B();C(0,9,1,6);C(0,20,0,2);C(57,503,1,2);F();
		C(24,31,1,6);B();C(24,31,1,6);B();C(0,232,1,1);C(0,40,0,6);C(0,5,1,8);C(0,255,0,1);C(0,10,1,4);C(2,4,0,2);C(46,77,0,5);F();
		C(31,38,0,3);C(0,21,1,1);C(28,66,0,6);C(0,17,1,6);B();B();C(0,42,0,8);F();
		C(0,18,1,8);C(0,63,0,2);F();
		C(7,19,1,4);C(27,78,1,4);B();C(0,32,1,5);C(22,47,1,3);C(0,4,0,8);F();
		C(0,14,0,8);C(0,259,0,8);C(7,41,0,1);C(0,93,1,8);F();
		C(2,4,0,2);C(23,66,0,1);C(56,86,1,5);C(26,34,1,2);B();C(0,27,0,7);C(0,7,0,2);B();C(0,4,1,7);F();
		B();C(1,6,1,2);B();B();B();F();
		C(2,8,0,4);B();C(0,1,1,3);C(24,31,0,6);C(7,19,1,1);C(0,3,0,3);F();
		C(0,262,0,5);F();
		C(0,3,0,7);C(5,13,0,4);C(6,27,0,5);C(101,141,0,2);C(0,17,1,5);F();
		C(2,29,0,2);C(9,13,0,3);F();
		C(0,255,1,2);F();
		return 3.281745;
	}

	double compute_indices_15 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(37,60,0,6);C(4,253,1,8);C(18,19,1,7);F();
		C(0,64,0,8);F();
		C(0,3,0,5);C(5,8,1,8);F();
		C(23,177,0,2);F();
		C(0,3,0,6);C(6,8,2,6);C(0,15,0,5);F();
		C(0,7,2,8);F();
		C(0,9,0,6);F();
		C(0,1,1,7);C(0,4,1,4);C(2,14,1,5);C(0,10,0,4);C(13,58,0,4);F();
		C(2,21,0,2);C(1,11,0,3);F();
		C(0,23,1,3);F();
		C(9,31,0,4);C(1,6,1,2);C(0,8,0,8);C(0,14,1,4);C(4,8,0,4);C(0,4,1,7);F();
		C(0,41,0,3);C(3,7,0,4);C(9,16,0,3);C(121,785,0,7);F();
		C(8,33,0,1);F();
		C(6,712,0,3);F();
		C(18,24,1,4);C(2,20,0,1);C(0,65,1,2);C(0,16,0,1);C(0,60,0,1);C(59,84,0,2);F();
		C(20,108,1,3);C(49,57,1,5);C(11,41,1,5);C(4,5,1,1);C(8,20,0,2);F();
		C(0,7,0,4);C(23,147,1,6);C(93,143,0,2);B();C(0,4,1,6);C(0,3,0,1);C(6,52,0,7);C(0,2,1,4);C(1,6,1,4);F();
		C(0,42,1,1);C(78,159,0,2);C(2,5,1,6);B();C(0,15,0,2);F();
		C(33,161,1,1);C(10,26,1,7);C(0,2,0,7);C(15,20,0,3);C(0,11,0,3);F();
		C(0,208,0,5);F();
		C(9,16,1,7);C(0,54,1,8);F();
		C(57,67,1,8);F();
		B();C(46,147,0,1);C(64,264,1,5);B();C(0,38,0,6);C(7,41,1,7);C(37,60,0,6);C(15,268,1,8);F();
		C(103,266,0,3);C(46,61,0,4);B();C(34,42,1,2);B();C(1,11,1,3);F();
		C(5,36,0,4);C(76,417,1,1);C(0,4,0,5);B();C(51,60,0,8);C(17,22,0,1);C(0,2,0,4);C(0,1,1,1);B();C(2,5,2,3);C(0,103,0,7);F();
		C(0,19,0,8);C(0,17,1,2);C(9,14,1,7);F();
		C(17,24,1,2);C(6,712,1,5);C(15,16,0,1);F();
		C(0,38,1,8);C(0,38,1,8);C(0,3,1,8);C(51,70,0,2);C(51,70,0,2);C(0,19,1,1);C(8,718,1,4);C(31,38,1,7);B();C(0,3,0,8);C(6,9,1,6);C(24,31,0,1);F();
		C(0,8,0,8);C(80,284,1,7);C(0,153,1,6);C(0,4,1,5);C(8,26,1,7);C(6,712,1,5);C(82,434,1,5);C(8,718,1,4);C(114,650,1,2);F();
		B();C(8,723,0,6);C(3,246,0,4);C(88,154,1,2);B();C(10,31,0,7);C(17,26,1,3);F();
		C(6,712,1,6);C(2,18,1,3);F();
		B();C(0,2,0,5);C(106,872,0,1);F();
		return 1.142361;
	}

	double compute_indices_16 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,230,0,4);F();
		C(0,4,2,3);F();
		C(1,7,1,6);C(0,5,0,2);F();
		C(0,8,0,5);C(0,2,1,8);C(0,7,1,2);C(0,2,0,6);F();
		C(0,11,0,7);C(0,17,0,1);F();
		C(10,12,0,3);C(10,12,1,4);C(20,22,1,1);F();
		C(12,24,1,2);C(3,4,0,3);C(0,20,0,2);F();
		C(0,5,2,5);B();C(22,28,0,5);F();
		C(7,19,1,2);B();C(0,14,1,5);C(6,27,1,4);C(2,8,0,4);C(6,29,0,2);C(6,11,1,4);F();
		C(0,5,0,2);C(7,41,0,7);C(3,10,0,6);C(97,844,0,2);F();
		C(0,7,0,4);C(1,34,1,4);C(1,12,1,5);C(4,26,0,5);C(2,9,0,2);C(36,48,0,1);C(0,11,1,1);F();
		C(34,42,1,8);C(11,54,1,6);C(14,36,1,6);C(16,18,1,1);C(0,15,0,3);C(31,38,1,3);C(0,32,0,4);F();
		C(0,5,1,7);C(0,6,0,5);C(10,26,1,6);B();C(12,18,0,5);C(14,62,1,4);F();
		C(53,61,0,4);C(56,63,2,8);C(29,33,1,5);C(4,20,1,5);F();
		C(0,65,1,3);F();
		C(2,76,1,2);C(2,20,0,1);C(0,10,2,4);B();C(31,37,0,2);C(31,38,1,3);F();
		C(49,83,0,8);C(2,4,1,6);C(6,14,1,7);F();
		C(49,74,0,4);C(8,39,1,4);C(0,116,0,1);C(10,11,0,6);F();
		C(6,11,1,4);C(24,31,1,6);C(2,9,1,1);B();C(46,147,1,1);F();
		C(12,233,1,2);C(2,162,1,2);C(18,106,0,7);C(4,12,1,4);F();
		B();C(4,253,0,1);C(0,14,0,7);C(1,2,1,2);F();
		C(0,260,0,1);C(1,5,0,4);C(0,23,0,5);F();
		B();C(0,261,1,5);C(2,20,1,1);C(1,2,1,6);F();
		C(2,55,1,1);C(3,7,1,8);C(8,17,0,4);C(88,411,1,5);C(18,24,0,7);C(67,422,0,6);F();
		C(0,2,1,5);C(0,1,1,2);C(10,23,1,1);C(80,421,0,5);C(0,7,0,1);F();
		C(4,253,1,8);C(82,434,1,5);C(3,61,1,7);F();
		C(82,434,1,5);C(2,3,0,1);B();F();
		C(31,38,1,7);C(0,65,1,7);C(6,18,1,3);C(3,7,1,2);C(1,6,0,3);C(0,706,1,7);C(29,33,0,8);F();
		B();C(0,11,1,1);C(24,31,1,6);C(16,709,0,4);C(0,266,0,8);F();
		C(8,718,0,4);B();C(8,39,1,1);C(3,4,1,4);F();
		C(0,16,0,5);C(0,15,0,6);C(8,718,1,1);F();
		C(121,785,0,7);C(0,261,0,5);F();
		return 1.532565;
	}

	double compute_indices_17 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,6,1,5);C(0,6,1,3);C(1,6,0,3);F();
		C(8,10,1,8);C(3,5,1,3);C(0,12,0,7);F();
		C(0,5,1,4);C(10,22,0,3);B();F();
		C(90,826,0,1);F();
		C(24,31,1,6);C(1,6,1,5);C(0,22,0,4);C(0,14,0,2);B();F();
		C(0,38,0,6);F();
		C(1,19,0,5);C(6,11,0,3);C(5,65,0,1);C(0,23,1,4);C(0,23,0,3);C(0,16,0,8);C(0,18,0,2);F();
		C(0,15,1,8);C(39,85,0,4);F();
		C(0,94,0,7);F();
		C(0,43,0,1);C(2,55,1,6);C(0,111,1,3);F();
		C(82,434,1,6);F();
		C(0,116,0,3);F();
		C(111,144,1,6);C(22,47,1,3);C(32,107,1,1);F();
		C(5,25,1,6);C(13,58,1,4);F();
		C(0,2,0,4);C(5,13,1,4);C(0,1,1,1);C(0,4,1,4);C(0,208,1,7);C(83,851,1,6);F();
		C(0,1,1,5);C(57,82,1,3);C(9,221,1,6);F();
		C(20,46,0,8);C(0,65,1,8);C(0,40,0,4);C(0,243,0,8);C(25,46,1,2);C(21,24,1,2);C(0,64,0,4);C(0,8,0,3);C(29,33,1,7);C(2,4,1,7);C(38,65,0,2);C(30,40,0,2);F();
		C(0,246,0,5);C(8,39,0,1);C(2,5,1,4);C(0,98,0,7);C(0,1,1,5);C(66,139,0,3);F();
		C(17,58,0,1);B();C(1,11,0,5);C(0,15,0,8);C(0,4,0,8);C(32,107,1,1);C(22,47,1,3);C(111,144,1,6);C(9,10,1,7);C(6,254,1,4);C(3,37,0,4);C(2,5,2,3);C(0,18,1,1);C(0,230,0,4);C(10,19,0,4);C(6,21,0,8);F();
		C(0,13,1,6);C(0,38,1,1);C(0,256,0,4);C(12,24,1,2);F();
		C(37,57,0,1);C(0,259,0,8);C(0,1,0,3);C(17,26,0,8);C(0,4,0,4);C(11,23,0,1);C(7,8,0,8);F();
		C(0,6,0,8);C(29,52,0,1);C(0,5,1,5);C(12,68,0,1);C(12,68,0,1);C(0,259,0,8);F();
		C(7,20,0,4);C(26,29,1,1);B();C(0,51,1,3);C(10,33,1,7);C(62,64,0,5);C(46,147,1,1);C(84,263,1,2);C(6,11,1,4);C(0,36,0,7);C(0,75,0,2);C(0,3,0,2);C(37,57,1,2);F();
		C(108,266,0,2);C(2,262,1,5);F();
		C(22,63,0,1);C(0,260,1,8);C(0,54,0,5);C(76,417,1,1);F();
		C(1,4,0,1);C(83,225,1,5);C(45,351,1,2);C(76,417,1,5);C(14,21,0,3);C(31,75,1,8);F();
		C(88,154,1,2);C(105,415,1,2);C(17,30,1,8);F();
		C(0,230,0,4);C(53,61,0,2);C(82,434,1,1);F();
		C(0,9,1,4);C(54,704,1,2);C(3,8,1,5);C(0,60,1,6);C(49,69,0,7);F();
		C(114,650,1,2);C(82,434,1,5);C(0,153,1,6);C(8,718,1,4);C(6,712,1,5);F();
		C(8,163,0,7);C(55,725,1,2);F();
		C(0,8,1,7);C(8,39,1,5);B();C(97,844,1,4);C(1,10,0,7);C(46,147,1,1);F();
		return 1.704065;
	}

	double compute_indices_18 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,13,0,5);C(0,31,0,5);F();
		C(6,14,0,4);C(0,208,1,1);F();
		C(0,8,2,4);F();
		C(10,57,0,2);F();
		C(0,11,0,4);C(2,7,0,6);C(1,12,1,4);C(12,18,0,5);F();
		C(1,22,1,1);C(11,41,0,8);C(4,7,0,2);F();
		C(3,15,1,6);C(0,2,0,5);C(0,4,0,4);F();
		C(1,7,1,8);B();C(0,5,0,8);B();F();
		C(17,26,0,6);F();
		C(23,33,2,4);C(0,5,0,4);B();F();
		C(1,30,0,2);C(12,20,0,4);C(0,16,0,5);C(31,38,1,2);C(0,11,1,4);B();F();
		C(27,51,1,1);C(19,50,1,1);C(6,40,0,5);C(3,12,1,3);C(0,6,1,5);C(17,26,1,7);C(13,24,1,8);F();
		C(97,844,1,3);F();
		C(1,61,0,2);F();
		C(3,7,1,4);C(0,36,0,3);C(0,16,1,3);C(0,68,0,1);C(35,81,1,2);F();
		C(32,69,1,8);C(14,56,0,7);C(0,259,1,5);F();
		C(111,144,1,6);C(0,15,1,5);F();
		B();C(39,85,0,2);F();
		C(13,215,0,3);F();
		C(19,30,1,4);C(24,103,1,1);C(5,36,0,2);F();
		C(0,14,0,5);C(0,14,0,1);C(59,146,0,3);F();
		C(64,267,1,4);F();
		C(0,18,0,4);C(0,199,1,8);C(10,18,1,5);F();
		C(0,19,0,3);F();
		C(8,718,0,1);F();
		C(14,21,1,7);C(0,18,0,1);F();
		C(0,10,0,3);F();
		C(0,275,0,5);C(14,21,0,8);C(0,259,1,5);C(1,34,1,2);C(1,48,1,1);C(0,4,0,3);C(6,20,1,7);C(2,18,1,2);B();C(34,42,1,2);C(1,11,0,3);C(6,8,0,8);C(9,17,1,1);C(6,11,0,8);F();
		C(45,351,1,2);B();C(24,31,1,6);C(0,5,1,1);C(2,9,0,1);F();
		B();C(82,434,0,5);C(0,255,0,4);C(13,32,1,3);C(111,147,0,3);C(2,21,0,4);C(1,6,1,2);C(4,16,0,1);C(7,12,1,3);C(0,4,1,1);C(0,6,1,4);C(111,147,0,3);C(82,434,0,5);C(0,255,0,1);C(8,12,1,4);F();
		C(46,147,1,4);F();
		C(8,718,1,4);C(0,12,1,3);C(7,19,0,2);F();
		return 1.066677;
	}

	double compute_indices_19 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,4,0,1);B();F();
		C(0,1,1,2);B();C(1,12,0,4);F();
		B();C(2,13,1,1);F();
		C(5,13,1,4);C(0,15,0,6);B();C(3,13,1,1);C(2,6,1,5);C(4,5,1,7);F();
		C(0,18,0,3);C(3,7,1,2);C(0,4,1,7);F();
		C(9,21,0,2);B();B();B();F();
		C(12,18,0,2);B();C(0,11,0,5);C(9,10,1,8);C(0,4,1,4);C(8,26,1,2);C(0,11,2,8);C(7,10,0,8);B();C(10,33,0,4);F();
		C(0,1,1,1);C(7,32,0,2);C(12,25,1,4);F();
		B();C(0,18,1,3);C(0,15,1,6);C(8,39,1,4);F();
		C(4,37,0,1);C(3,11,0,4);C(0,26,1,3);C(1,2,1,2);C(0,40,0,4);C(25,44,0,3);C(6,21,1,5);C(0,8,0,5);C(4,38,1,1);C(1,13,1,6);F();
		C(50,54,2,5);C(0,8,2,2);F();
		C(19,65,1,3);C(23,35,0,3);C(0,2,1,3);F();
		B();C(49,69,0,7);C(0,24,1,2);C(53,61,0,8);C(4,26,1,4);C(3,8,1,5);C(0,60,1,6);F();
		C(20,24,0,2);C(66,71,0,5);C(0,1,0,2);C(0,10,0,4);F();
		C(25,71,1,2);B();C(0,6,1,1);C(2,4,1,2);C(0,13,1,1);C(56,86,1,7);F();
		C(0,2,1,4);C(6,19,1,2);C(24,32,0,7);C(3,89,0,8);C(16,20,1,1);F();
		C(3,8,1,4);C(82,132,0,3);C(0,4,0,5);C(24,31,1,5);C(0,14,0,1);F();
		C(66,139,0,2);F();
		C(5,13,1,4);C(6,27,0,4);C(8,163,1,2);C(5,13,1,4);C(1,8,1,2);F();
		C(0,5,1,2);C(61,69,0,5);C(1,6,1,2);C(0,4,1,7);C(0,208,1,4);C(0,14,1,4);C(8,20,1,2);C(28,66,1,2);C(29,33,1,4);C(14,21,0,2);C(80,142,1,7);C(0,6,1,7);B();B();C(0,5,0,8);C(0,7,1,1);B();C(0,17,0,8);C(1,2,0,6);B();C(0,18,1,8);C(0,7,1,4);F();
		B();C(0,210,0,5);C(2,6,1,2);C(0,14,1,4);C(1,6,1,2);C(0,4,1,5);C(0,1,1,7);F();
		C(0,1,0,1);C(0,13,0,2);C(0,23,1,3);B();C(0,232,1,1);F();
		B();C(9,221,1,5);C(9,14,1,5);C(0,2,1,4);B();C(0,24,1,8);C(0,253,1,4);F();
		C(0,10,0,1);C(0,255,0,7);C(36,48,0,6);F();
		C(0,259,1,5);F();
		C(0,14,1,8);C(0,15,1,6);C(11,240,1,8);C(8,302,0,6);C(0,68,1,8);C(74,146,0,3);F();
		C(0,21,1,1);C(24,31,1,7);C(99,142,0,2);C(3,9,1,3);C(45,351,1,2);F();
		C(0,275,1,2);C(0,5,0,5);C(10,57,0,2);C(0,255,0,1);C(0,21,1,1);C(45,351,1,2);C(0,3,0,3);C(0,3,0,3);F();
		C(76,417,1,5);C(76,417,1,5);C(0,3,1,3);C(0,6,0,1);C(0,12,1,1);F();
		C(68,425,1,1);C(2,9,0,5);C(5,24,0,2);F();
		C(5,33,0,5);C(0,28,0,1);C(0,54,1,2);C(111,144,1,6);C(82,434,1,2);C(0,3,0,7);C(46,77,1,5);C(2,6,1,8);C(23,34,1,1);C(114,650,1,2);F();
		C(0,44,1,3);C(59,711,1,2);C(0,17,1,8);C(2,76,1,2);F();
		return 0.913017;
	}

	double compute_indices_20 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,2,1,6);C(0,8,2,4);F();
		C(0,3,0,5);C(2,9,1,2);C(0,7,0,1);F();
		C(0,6,0,6);B();B();C(0,7,1,1);C(0,5,0,8);C(1,4,0,4);C(0,11,2,8);C(0,6,0,8);B();F();
		C(1,7,1,3);C(0,21,1,6);C(6,13,1,7);F();
		C(2,9,0,2);C(15,25,1,3);F();
		C(6,28,0,6);F();
		C(24,31,1,6);C(57,63,2,3);C(24,37,0,1);C(11,12,1,3);F();
		C(0,14,1,5);C(18,25,1,6);C(0,11,0,1);C(0,13,0,1);C(0,36,0,7);C(6,14,0,4);C(9,51,0,3);F();
		C(20,75,1,1);B();F();
		C(0,5,1,4);C(0,22,1,5);C(5,6,1,4);C(0,4,0,1);C(37,57,1,1);C(8,112,0,3);C(80,142,1,7);C(0,6,1,7);B();B();C(0,5,0,8);C(0,7,1,7);F();
		C(0,73,1,3);C(0,14,1,6);C(10,52,1,3);C(9,18,0,4);C(5,20,0,4);C(99,142,0,2);C(59,711,0,2);F();
		C(111,144,1,6);C(0,43,1,5);F();
		C(9,24,0,1);C(0,5,0,8);C(8,163,1,2);C(0,5,0,8);F();
		C(0,59,1,1);C(0,208,1,5);C(0,4,1,7);C(0,6,1,6);C(57,813,1,6);F();
		C(0,14,1,4);C(9,27,1,4);C(1,6,1,2);C(0,208,0,5);C(5,13,0,3);C(8,12,1,4);C(33,62,0,4);F();
		C(0,247,1,5);C(4,7,1,8);C(8,20,1,2);C(18,25,0,6);C(9,24,1,4);C(0,27,0,8);C(14,21,0,2);C(0,14,1,4);C(82,132,0,3);C(62,96,0,3);F();
		C(1,3,0,7);C(0,10,1,7);C(0,10,1,6);C(0,257,1,8);C(0,255,1,1);C(53,150,0,2);C(14,259,0,6);F();
		C(61,363,1,3);C(8,33,0,8);F();
		C(0,5,0,8);C(1,24,0,2);B();C(0,246,1,5);C(0,262,0,5);C(0,255,0,5);C(0,247,1,5);C(2,8,1,8);C(36,48,0,6);C(6,9,0,4);C(34,42,0,2);F();
		C(1,6,1,7);C(85,268,0,3);C(47,72,0,5);C(8,26,1,1);C(0,60,0,4);C(71,123,0,7);F();
		C(10,245,1,7);C(0,2,2,3);C(0,14,0,8);C(0,125,0,3);C(8,302,0,6);C(29,33,1,1);F();
		C(10,16,1,1);C(0,5,1,1);C(45,351,1,2);C(3,8,1,7);C(2,4,1,7);C(21,27,1,4);F();
		C(0,1,0,3);C(0,5,0,2);C(0,208,1,6);C(0,15,0,3);C(8,163,1,3);C(76,417,1,5);C(0,11,1,1);F();
		C(6,17,0,8);C(46,147,0,8);C(82,434,1,5);F();
		C(0,2,1,7);C(10,16,1,7);C(3,8,0,7);C(82,434,1,6);C(0,31,1,1);C(61,78,0,5);C(7,19,1,2);B();C(0,17,0,8);B();C(14,21,0,2);C(0,255,1,5);C(6,19,1,8);C(1,11,0,3);C(9,27,0,4);C(11,24,1,1);B();C(8,10,1,6);C(0,7,1,4);F();
		C(8,718,0,1);F();
		C(53,61,0,4);C(0,71,0,5);C(76,417,1,1);C(0,6,1,7);C(81,449,0,6);C(38,57,0,2);C(5,46,0,4);C(14,31,0,7);F();
		C(0,11,2,3);C(25,46,1,2);C(0,3,0,5);C(80,498,1,2);F();
		C(113,640,0,7);B();C(4,91,1,8);F();
		C(64,101,1,7);C(114,650,1,2);F();
		C(8,39,0,8);C(46,147,0,8);C(0,10,1,1);C(2,54,1,1);C(114,650,1,2);F();
		C(0,5,0,4);B();B();C(0,4,1,1);C(97,844,1,1);C(39,85,0,2);C(57,138,1,2);F();
		return 2.185707;
	}

	double compute_indices_21 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,8,0,5);B();F();
		C(0,10,0,7);C(2,11,1,1);C(0,6,0,3);C(2,13,1,8);C(0,3,0,1);C(1,6,1,4);C(0,6,0,3);C(2,13,1,8);C(1,6,1,4);C(2,11,1,1);F();
		C(0,2,1,7);C(0,16,1,4);C(0,6,0,7);B();F();
		C(0,4,1,8);C(8,20,1,2);C(0,4,1,8);C(0,3,1,5);C(9,14,1,3);F();
		C(11,35,1,1);F();
		B();C(9,15,0,3);C(0,12,0,8);C(12,24,0,3);C(0,6,0,7);C(8,20,0,3);F();
		C(17,26,1,3);F();
		C(23,34,1,2);F();
		C(0,12,1,3);C(15,24,1,6);C(14,21,0,6);C(0,6,0,1);C(0,18,1,3);F();
		C(21,37,1,6);C(3,9,1,8);F();
		B();C(0,1,1,3);C(0,4,1,4);C(3,23,1,5);C(0,10,0,4);C(0,38,0,6);C(0,9,0,3);F();
		C(34,42,1,5);F();
		C(1,6,1,4);C(0,35,1,3);C(25,46,1,2);C(0,21,1,2);C(13,14,1,7);F();
		C(12,27,0,1);C(11,46,0,8);F();
		C(0,46,1,2);F();
		C(10,54,1,2);C(0,3,1,8);F();
		C(0,6,0,1);C(37,57,1,2);F();
		B();C(0,8,1,2);C(4,6,0,4);C(0,3,1,8);C(7,8,1,4);C(43,62,1,3);F();
		C(48,63,2,5);F();
		B();C(12,13,1,2);C(52,63,1,8);C(9,51,1,4);F();
		C(3,8,1,5);C(5,6,1,1);C(0,4,0,6);C(0,1,1,8);C(49,69,0,7);F();
		C(27,71,1,8);C(6,9,1,4);C(0,1,1,5);B();C(5,43,1,3);F();
		C(10,72,1,2);C(9,16,1,3);C(6,15,1,6);C(19,22,1,4);F();
		C(9,18,0,5);C(0,11,0,6);C(5,9,1,3);C(1,12,0,4);C(0,76,1,2);C(7,15,1,2);F();
		C(3,8,0,3);C(2,4,0,2);C(0,15,1,6);B();C(56,86,1,5);C(26,34,1,2);C(22,63,0,1);B();F();
		C(0,90,0,7);F();
		C(0,60,1,1);C(0,109,0,3);C(8,12,1,8);F();
		C(26,31,0,4);C(0,116,0,3);C(1,3,0,2);C(3,6,0,3);C(31,38,0,3);F();
		C(111,144,1,6);B();B();C(10,22,1,3);C(0,66,1,7);C(3,61,1,7);F();
		C(11,24,1,6);C(0,26,0,4);C(7,21,0,1);C(8,163,1,1);C(17,26,0,3);F();
		C(22,63,0,4);C(6,14,1,7);F();
		C(0,259,1,8);C(4,6,0,4);F();
		return 4.658991;
	}

	double compute_indices_22 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		B();C(8,163,1,3);C(39,85,0,2);B();C(15,25,0,4);C(5,7,0,1);F();
		C(5,31,0,3);C(0,2,2,8);C(8,718,1,1);C(121,785,0,7);C(1,7,0,7);F();
		C(0,35,1,7);C(21,79,1,4);C(0,255,1,1);F();
		C(0,281,0,6);F();
		C(0,4,1,3);C(56,86,0,5);C(28,66,1,6);C(28,66,1,6);F();
		C(0,20,0,3);C(17,27,1,8);C(104,652,1,2);C(4,15,1,3);F();
		C(0,3,0,8);C(83,439,0,5);C(9,20,1,8);F();
		C(4,6,0,4);C(3,5,0,8);C(0,7,0,8);C(2,5,0,4);C(0,21,1,1);C(25,75,0,1);F();
		C(22,63,0,1);C(59,711,0,2);C(0,11,0,8);C(26,34,1,2);C(2,4,0,2);C(56,86,1,5);F();
		C(0,3,0,1);C(5,9,1,3);C(0,3,1,2);C(6,39,0,1);C(0,43,1,7);C(1,11,0,7);B();C(24,31,1,6);F();
		C(11,22,0,3);C(0,6,1,4);F();
		C(0,253,0,7);C(0,8,1,3);C(0,47,0,4);C(11,30,0,3);C(7,15,1,6);B();F();
		B();C(8,33,0,1);C(0,5,1,2);C(82,434,1,1);C(15,18,0,2);C(0,3,1,1);F();
		C(0,281,1,6);C(0,244,0,5);C(13,15,0,3);B();C(0,42,0,5);C(0,43,1,7);C(20,36,1,5);C(0,16,1,4);C(0,38,0,8);C(114,657,1,7);C(13,17,1,5);C(3,24,1,5);C(14,27,1,7);C(23,28,1,2);F();
		C(2,55,1,3);C(4,38,0,8);C(2,36,1,2);C(87,424,1,5);F();
		C(0,262,0,5);F();
		C(0,698,0,1);F();
		C(0,253,1,2);C(6,24,1,2);B();C(0,153,1,6);C(22,47,1,3);C(28,66,0,6);F();
		C(2,17,1,1);C(18,19,1,1);C(103,135,1,6);F();
		C(17,20,0,4);C(0,249,1,2);F();
		B();C(0,6,0,6);C(0,8,1,2);F();
		C(31,37,0,2);C(21,25,0,8);C(0,235,1,7);F();
		C(97,844,0,2);C(4,20,1,6);F();
		C(11,36,1,7);C(25,46,0,2);C(13,20,1,6);C(9,67,1,5);F();
		C(59,711,1,2);C(6,24,1,1);F();
		C(99,142,0,2);C(0,2,0,1);F();
		C(0,11,1,5);C(0,4,0,6);C(41,73,1,4);C(31,70,0,3);C(0,14,1,4);C(0,7,1,8);C(0,14,1,6);F();
		C(0,4,1,1);C(0,89,0,3);C(5,46,0,7);C(0,68,0,4);C(0,1,2,5);C(31,70,0,6);F();
		C(0,264,0,5);C(0,9,0,4);C(76,502,0,2);F();
		C(0,1,1,6);C(0,5,1,7);C(0,3,1,4);C(0,14,0,5);F();
		C(57,71,0,5);C(8,15,0,3);C(82,264,0,3);C(3,7,1,2);F();
		C(7,12,1,1);C(2,701,1,5);C(82,434,1,5);F();
		return 2.763273;
	}

	double compute_indices_23 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,4,0,7);C(0,4,1,7);F();
		C(1,29,1,3);C(20,22,2,8);B();F();
		C(0,12,0,3);C(29,33,1,2);C(6,27,0,4);F();
		C(6,47,1,1);C(0,3,1,8);B();C(0,264,0,5);F();
		C(2,55,1,6);C(8,39,1,1);C(25,46,1,5);F();
		C(0,4,0,4);C(0,2,2,8);C(5,103,0,1);C(0,14,0,2);C(5,46,1,5);C(4,189,0,3);F();
		C(7,27,0,2);C(66,139,1,3);B();C(8,18,0,8);F();
		C(8,20,0,4);F();
		C(0,7,0,1);B();C(0,54,1,3);C(46,147,1,1);C(0,8,0,3);F();
		C(0,239,1,7);C(0,8,1,3);C(0,68,1,8);F();
		C(54,74,0,8);C(14,258,0,1);C(0,246,0,5);C(18,21,0,8);F();
		C(0,260,0,1);C(0,5,2,2);F();
		C(0,31,1,7);C(22,260,0,1);C(11,46,0,3);C(11,46,0,8);C(19,30,1,4);F();
		C(15,25,0,3);C(0,260,0,1);C(0,30,0,4);C(0,8,2,4);C(5,13,1,4);C(1,11,1,8);C(0,6,1,5);C(13,23,1,6);C(0,27,1,7);F();
		C(0,263,1,1);C(14,33,1,5);C(0,1,1,8);C(0,1,2,5);C(0,28,0,3);C(26,260,0,1);C(13,27,0,2);B();F();
		C(23,31,1,2);C(0,275,0,5);C(33,161,0,1);C(2,4,0,1);C(0,262,0,2);C(0,5,1,1);C(0,239,0,7);C(0,6,0,3);F();
		C(56,346,0,3);C(56,346,1,3);C(0,15,0,1);C(11,30,0,1);C(6,39,0,1);C(13,23,0,4);C(0,4,2,3);C(0,6,1,4);C(0,7,0,7);B();C(0,10,0,8);C(7,15,1,1);B();B();C(56,63,2,8);C(53,150,0,2);F();
		C(0,255,0,5);C(0,9,0,4);C(76,502,0,2);C(0,3,1,7);C(6,21,0,1);C(15,16,1,4);C(0,20,0,4);C(0,12,1,7);C(1,11,0,3);C(0,70,0,3);C(2,20,0,1);C(0,1,1,8);C(0,16,1,8);C(8,79,0,2);C(0,43,1,7);C(24,30,1,2);C(7,14,1,1);F();
		C(118,645,0,6);C(2,287,0,6);C(0,3,0,4);C(18,24,0,4);C(0,5,0,8);C(53,61,0,4);C(82,434,1,5);C(24,37,1,1);C(40,42,0,1);F();
		C(0,6,0,3);C(0,8,0,5);C(3,35,0,3);C(0,22,1,3);C(0,4,0,4);C(5,46,0,4);C(0,14,0,5);C(0,43,1,7);C(72,144,0,1);C(0,5,1,8);C(0,9,1,5);C(15,53,0,4);C(2,23,1,2);C(0,255,0,1);C(0,10,1,4);C(2,4,0,2);C(46,77,0,5);C(59,711,0,2);F();
		C(0,116,0,3);C(59,711,0,2);C(0,60,0,3);F();
		C(0,255,0,1);C(0,255,0,1);C(5,9,1,5);C(7,18,0,2);C(59,711,0,2);F();
		C(6,712,0,3);C(14,62,1,4);C(14,258,0,1);C(39,42,0,1);C(0,2,1,1);C(0,255,1,7);F();
		C(10,38,0,1);C(8,718,0,4);C(5,43,0,1);C(53,61,1,5);C(0,255,0,1);F();
		B();C(3,246,0,4);C(0,15,1,5);C(10,31,0,7);C(17,26,1,3);C(8,723,0,6);B();C(17,26,1,3);F();
		C(9,18,1,5);C(4,728,0,2);C(14,31,0,7);C(0,9,1,3);F();
		C(0,1,0,1);C(37,57,0,5);C(0,4,2,1);C(9,42,0,6);C(120,838,1,6);F();
		C(97,844,0,2);C(10,19,1,8);C(1,257,0,1);C(0,17,1,3);C(18,38,0,3);F();
		C(82,264,1,3);C(0,15,1,8);C(1,12,1,4);C(0,7,0,7);C(111,646,1,7);C(0,6,1,3);C(97,844,0,3);C(14,259,1,6);C(8,70,1,3);F();
		C(2,55,0,7);C(82,264,0,4);C(43,80,1,4);C(0,21,1,1);C(18,24,0,1);C(2,55,0,7);C(0,2,2,8);C(6,19,1,7);C(0,42,0,5);C(8,163,1,2);C(97,844,0,5);C(105,217,0,2);F();
		C(82,434,0,5);C(111,147,0,6);C(0,255,0,1);C(8,12,1,4);C(42,62,0,2);C(97,844,0,5);F();
		C(0,85,0,7);C(106,872,0,1);C(92,840,1,5);F();
		return 2.000905;
	}

	double compute_indices_24 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,4,1,6);C(0,4,0,1);C(0,3,1,5);B();C(0,2,2,8);F();
		C(0,2,1,1);C(0,16,1,8);F();
		C(10,28,0,7);F();
		C(7,22,0,3);C(0,17,0,3);C(47,59,2,6);C(0,36,0,8);C(0,6,0,5);C(19,47,0,4);C(0,8,1,1);C(0,18,1,3);C(0,7,0,4);C(2,16,0,4);C(18,19,1,8);C(15,25,0,7);C(5,9,1,3);C(23,31,0,4);F();
		C(0,60,1,4);B();C(22,63,0,1);F();
		C(7,13,1,5);C(0,3,1,1);C(0,2,1,6);C(0,20,0,2);C(0,68,0,4);C(2,6,0,1);C(8,12,1,4);C(0,9,1,2);F();
		C(10,26,1,1);C(11,57,0,3);C(0,4,1,3);B();C(39,85,1,2);C(1,15,1,1);F();
		C(0,10,0,8);C(1,9,0,1);C(0,4,1,5);C(42,86,0,2);F();
		C(5,22,0,4);C(0,10,1,3);C(5,22,0,4);C(0,89,0,3);C(5,46,0,7);F();
		C(6,98,0,3);F();
		C(0,7,1,3);C(2,128,0,3);B();C(0,11,1,5);F();
		C(5,47,1,3);C(0,2,1,1);C(60,140,0,1);F();
		C(46,147,1,2);C(0,113,0,4);F();
		C(7,14,0,2);C(5,46,0,4);C(108,149,0,2);F();
		C(8,163,0,8);C(11,19,1,5);C(0,6,1,7);C(0,2,1,4);C(8,12,0,1);C(0,28,0,1);C(103,135,1,6);C(0,11,1,8);C(9,18,0,3);F();
		C(48,69,0,5);C(10,11,1,4);C(30,164,0,6);F();
		C(0,60,0,3);C(0,16,1,8);C(2,20,0,2);C(83,225,1,5);B();C(0,9,1,3);C(24,31,0,6);C(7,30,1,4);F();
		C(1,8,0,8);C(0,3,1,8);C(15,25,0,3);C(0,17,0,1);B();B();C(14,21,0,2);C(34,42,0,4);C(0,1,1,4);C(15,21,1,2);C(0,2,0,2);C(83,225,1,2);F();
		C(15,41,1,2);C(62,64,0,4);B();C(82,241,0,3);C(62,64,0,5);F();
		C(102,215,0,5);C(0,47,1,2);C(0,253,1,4);F();
		C(2,34,1,4);C(105,217,0,2);C(0,43,1,2);C(0,253,1,4);F();
		C(0,14,1,1);C(16,93,0,3);C(9,11,1,1);C(7,21,0,1);C(0,259,1,5);F();
		C(8,39,0,7);C(0,11,1,6);B();C(0,5,1,4);C(0,4,0,2);C(1,12,1,4);C(0,10,1,8);C(0,27,0,8);C(0,259,0,8);C(53,61,1,5);F();
		C(0,94,0,1);C(10,264,0,2);C(63,73,1,3);C(0,15,0,4);C(38,46,0,4);C(0,14,0,1);C(27,38,1,2);F();
		C(0,14,0,7);C(0,3,1,4);C(0,1,0,1);C(61,402,1,2);F();
		C(0,5,1,2);C(0,4,1,1);C(82,434,1,1);C(15,18,0,2);B();F();
		C(112,639,1,7);C(0,94,0,7);B();C(21,102,0,6);C(22,47,0,2);F();
		C(56,86,1,5);C(83,272,1,3);C(118,645,0,6);C(14,21,1,6);F();
		C(0,10,1,1);C(2,54,1,1);C(114,650,1,2);C(0,12,1,1);C(111,653,1,2);C(0,255,1,2);C(15,279,1,8);C(0,255,1,3);C(0,255,1,2);C(15,230,1,7);C(14,266,1,5);F();
		C(6,712,1,5);C(19,30,1,8);C(82,434,1,2);B();C(2,5,2,5);C(0,4,2,3);F();
		C(0,9,1,8);C(99,142,1,2);C(102,653,1,7);C(3,8,0,5);C(4,6,0,4);C(97,844,1,3);C(7,10,0,4);C(14,259,1,6);C(0,6,1,3);C(0,1,1,1);F();
		C(106,872,1,1);F();
		return 1.741118;
	}

	double compute_indices_25 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,8,1,1);F();
		C(0,5,0,3);C(1,8,0,8);C(0,5,0,4);C(2,9,1,4);C(2,9,0,1);F();
		C(12,17,0,2);C(0,7,0,4);C(10,11,1,8);F();
		C(13,23,0,1);F();
		C(0,30,1,4);C(6,27,0,3);C(0,3,0,5);F();
		C(0,8,0,2);C(0,3,1,1);C(2,5,0,4);C(5,36,0,2);C(0,14,1,2);C(2,3,1,8);C(11,14,0,3);F();
		C(10,31,1,1);C(6,27,0,5);C(0,1,1,1);C(0,2,2,5);C(32,39,1,5);F();
		C(24,63,0,1);F();
		C(5,46,1,7);C(2,11,1,6);C(1,21,0,7);C(1,5,0,4);C(19,33,0,3);F();
		C(25,46,1,8);C(22,50,0,1);F();
		C(18,43,1,1);B();C(0,11,1,4);C(13,48,0,6);C(0,3,0,6);C(8,39,0,1);F();
		B();C(0,8,0,6);C(0,15,0,8);C(0,42,0,8);C(41,57,1,8);C(0,26,0,6);C(1,12,0,4);F();
		C(10,14,0,5);C(0,6,1,1);C(37,57,1,4);C(0,4,0,3);F();
		C(0,15,1,8);B();C(39,51,1,7);C(3,37,1,4);C(22,63,0,1);C(0,18,0,4);C(10,26,1,1);F();
		C(8,12,0,4);C(20,36,1,5);C(0,11,1,3);C(0,3,1,6);B();C(0,65,0,6);C(13,23,0,1);C(0,16,0,4);F();
		C(34,42,1,2);C(28,66,1,6);C(24,31,1,6);C(14,25,1,7);C(15,25,0,4);F();
		C(16,26,0,4);C(0,68,1,4);C(11,14,0,3);B();C(0,17,0,1);C(0,18,0,2);C(0,18,0,3);F();
		C(0,22,1,5);C(0,17,1,5);C(20,69,0,2);F();
		C(49,69,1,7);C(7,14,1,2);C(0,11,0,2);C(6,15,1,7);F();
		B();C(0,7,1,8);C(10,74,0,3);C(0,5,0,8);F();
		C(0,10,0,8);C(16,18,0,6);C(0,76,1,2);C(18,24,0,6);F();
		C(0,17,1,2);C(4,38,0,3);C(0,76,1,2);C(12,27,1,4);F();
		C(4,77,0,1);B();C(1,36,1,8);C(0,8,1,7);C(0,4,0,2);F();
		C(0,1,0,1);C(8,39,0,1);C(61,78,0,3);F();
		C(0,8,1,2);C(2,5,0,4);C(0,9,0,1);C(0,44,1,6);C(58,78,1,8);C(0,6,1,1);F();
		B();C(0,1,1,4);C(42,62,0,2);C(0,18,1,5);C(61,78,0,8);C(41,53,1,1);C(0,69,1,3);C(0,18,1,5);C(61,78,0,8);C(41,53,1,1);C(3,77,1,3);F();
		C(8,96,0,4);C(31,38,1,7);F();
		C(19,26,0,3);C(0,28,0,5);C(9,42,1,4);C(5,103,0,1);C(18,106,0,7);F();
		C(4,38,1,7);C(5,33,0,3);C(0,7,0,7);C(24,37,1,1);C(0,4,0,4);C(0,2,1,1);C(0,9,0,8);C(0,17,1,6);C(0,125,0,3);F();
		C(0,208,0,5);C(0,4,1,7);C(0,17,1,4);C(14,16,1,2);C(99,142,0,2);C(0,2,2,5);C(8,20,0,2);C(0,1,1,4);C(0,2,1,8);C(7,40,0,5);F();
		B();C(14,21,0,6);C(0,208,0,5);C(18,21,1,5);F();
		C(34,42,0,4);C(2,5,0,8);C(10,16,1,6);C(0,13,1,4);C(12,18,1,2);B();C(8,39,1,1);C(4,5,1,8);F();
		return 6.453201;
	}

	double compute_indices_26 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(1,3,1,2);C(1,6,0,4);B();F();
		B();C(0,4,1,1);B();C(0,9,1,4);B();F();
		C(0,36,0,3);F();
		C(4,19,1,7);C(5,9,1,4);F();
		C(1,8,1,5);F();
		C(6,14,0,4);C(0,14,0,2);C(3,19,0,6);F();
		C(0,15,1,6);B();F();
		C(8,12,1,4);C(15,16,1,2);C(0,4,1,7);F();
		C(8,16,0,5);F();
		C(11,20,1,4);C(15,17,0,6);F();
		B();C(14,21,0,2);F();
		C(10,19,1,8);F();
		C(27,34,0,4);F();
		C(9,41,0,1);F();
		B();C(0,18,0,3);F();
		C(21,42,0,1);F();
		C(8,20,1,7);C(0,4,1,2);C(0,44,0,7);F();
		C(0,2,2,8);C(5,13,1,8);C(0,47,0,1);F();
		B();C(4,6,0,4);C(8,39,1,4);C(48,53,1,6);F();
		C(2,9,0,3);C(57,63,2,3);F();
		C(0,17,1,6);B();B();C(0,116,0,3);C(0,17,0,4);F();
		C(78,159,0,2);C(2,5,0,4);C(0,42,0,8);B();F();
		C(0,59,1,1);C(0,208,1,5);C(0,4,1,7);F();
		C(0,2,2,5);C(0,246,0,5);C(21,159,0,7);F();
		B();C(7,30,1,4);C(0,11,1,2);C(8,12,1,4);C(2,252,0,3);F();
		C(0,40,0,4);C(10,264,0,1);C(20,40,1,3);C(0,40,0,4);C(10,11,1,2);C(39,85,1,2);C(10,28,1,7);C(52,70,1,5);F();
		C(0,12,1,1);C(0,14,0,5);C(9,16,0,3);C(0,65,1,2);C(0,2,0,1);C(8,16,0,5);C(0,275,0,5);F();
		B();C(0,78,1,2);C(2,20,0,1);C(0,60,0,3);C(16,17,1,8);B();B();C(65,403,1,2);B();B();F();
		C(0,3,0,5);C(76,417,1,5);C(0,3,0,5);F();
		C(0,11,1,1);B();C(41,55,1,6);C(6,712,1,5);B();F();
		C(55,79,0,5);C(82,264,1,3);C(46,147,1,1);F();
		B();F();
		return 2.229774;
	}

	double compute_indices_27 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,6,1,8);C(0,8,0,2);F();
		C(0,11,0,1);C(0,4,0,4);C(0,2,2,4);F();
		C(0,15,2,3);F();
		C(0,19,0,3);C(3,11,1,1);C(0,2,2,5);C(0,9,1,7);C(0,7,1,2);C(0,3,1,2);F();
		C(0,6,0,3);C(0,3,1,1);C(0,20,1,6);C(8,15,0,4);F();
		C(10,24,0,6);F();
		C(24,37,0,1);F();
		C(0,3,1,6);B();C(40,42,1,1);C(8,16,1,2);C(18,24,1,4);C(0,37,1,8);F();
		C(22,63,1,1);F();
		C(13,28,0,7);C(4,7,0,3);C(0,7,1,1);C(0,8,1,7);C(0,44,0,1);C(7,14,0,1);F();
		B();C(19,28,0,4);C(0,60,0,3);C(0,1,0,4);C(0,14,1,5);C(1,11,0,5);C(3,9,1,4);F();
		C(1,12,0,4);C(0,2,2,5);C(0,14,1,6);C(31,70,0,6);C(11,22,1,1);F();
		C(10,74,0,3);C(9,16,0,4);C(46,147,1,5);F();
		C(0,31,1,6);C(1,60,0,3);C(48,78,0,5);C(0,19,1,3);C(2,4,1,2);C(22,63,0,1);C(0,2,0,6);F();
		C(0,7,0,4);C(2,9,1,4);C(0,5,0,1);C(2,9,0,2);C(22,88,0,5);C(57,90,1,5);C(0,17,0,4);F();
		C(0,11,1,1);C(0,4,1,4);C(0,92,0,7);C(4,253,1,8);F();
		C(16,93,0,3);C(0,2,1,1);C(5,47,1,3);C(0,1,1,4);F();
		C(7,12,1,5);C(44,103,1,8);C(42,62,0,2);B();F();
		C(0,13,1,4);C(31,75,1,8);C(0,3,0,2);C(7,9,0,1);C(44,103,1,8);C(10,11,0,1);C(39,85,1,2);C(0,8,0,4);F();
		C(1,6,0,1);C(0,116,0,1);C(42,125,0,6);F();
		C(1,32,0,7);C(57,138,1,2);C(1,12,2,6);C(0,6,0,8);C(18,24,0,3);B();F();
		C(59,146,0,3);F();
		C(59,146,0,3);C(68,76,1,2);C(1,13,0,4);C(0,3,1,3);C(0,4,0,4);F();
		C(2,11,1,8);C(48,53,1,6);C(0,208,1,6);C(0,208,0,6);C(39,85,0,2);C(0,27,0,1);F();
		C(0,6,0,7);C(0,13,1,4);C(0,89,0,8);B();C(8,12,1,5);C(0,253,1,4);C(64,70,0,5);F();
		C(1,12,1,5);C(0,8,2,7);F();
		C(54,74,0,8);C(1,27,0,3);C(0,251,0,5);C(14,258,0,1);C(18,21,0,8);F();
		C(57,138,1,2);C(25,46,1,2);C(0,40,0,4);C(0,40,0,4);C(10,264,0,1);C(10,11,1,2);C(23,31,1,2);C(0,2,1,3);F();
		C(37,51,0,4);F();
		C(17,26,1,3);C(8,11,1,4);C(0,7,1,4);B();B();C(14,21,1,1);C(10,15,1,6);B();C(0,4,1,1);B();C(1,6,1,4);C(15,268,1,8);C(2,55,0,7);C(14,31,0,4);F();
		C(10,16,1,6);B();B();C(0,6,1,5);C(82,434,1,5);C(3,61,1,7);C(36,46,0,6);F();
		C(0,7,0,4);B();C(0,25,0,3);C(19,108,1,3);C(5,6,0,4);C(114,650,1,2);C(2,9,0,1);C(28,57,1,3);F();
		return 1.571859;
	}

	double compute_indices_28 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(111,653,1,2);F();
		C(104,652,1,2);F();
		B();C(0,10,1,7);F();
		C(4,38,1,1);C(5,13,1,4);C(0,4,1,7);C(1,5,0,4);C(8,18,0,6);C(0,3,0,4);F();
		C(3,24,1,5);C(6,14,0,4);C(0,14,0,8);C(0,14,1,5);C(14,27,1,7);C(45,52,1,1);C(0,17,0,1);C(8,31,1,3);C(9,42,1,4);C(13,17,1,5);C(23,28,0,2);F();
		C(67,70,0,8);C(2,13,1,8);C(0,5,1,7);F();
		B();C(2,4,1,2);C(22,63,0,1);C(0,6,0,7);F();
		C(35,59,0,1);F();
		C(15,53,0,4);C(73,82,1,7);F();
		B();C(5,8,1,3);C(39,85,0,4);C(0,1,0,7);C(24,31,1,6);C(0,3,0,8);C(1,5,0,4);C(6,29,0,2);F();
		C(56,86,1,7);C(22,63,1,3);F();
		C(31,98,0,6);F();
		C(0,6,1,4);C(20,108,0,3);F();
		C(101,136,1,8);C(4,6,0,4);C(3,8,0,5);C(19,30,0,6);F();
		C(0,8,1,4);C(99,142,0,2);C(4,6,0,4);B();F();
		C(111,144,1,6);F();
		C(0,153,1,6);C(0,153,1,6);C(9,31,1,4);C(0,1,0,4);F();
		C(6,10,1,3);C(8,163,1,2);C(7,15,1,4);F();
		C(11,20,1,4);F();
		B();C(41,53,1,3);C(0,208,1,6);C(7,19,0,8);C(0,9,0,2);C(8,14,1,5);F();
		C(10,22,0,7);C(0,232,1,6);C(6,29,0,6);F();
		C(5,36,1,2);C(39,45,0,6);C(0,242,1,2);C(0,40,0,7);C(6,12,1,6);C(0,7,1,8);C(0,7,1,1);F();
		C(48,63,2,5);C(4,253,1,8);F();
		C(0,4,0,8);C(0,255,1,2);B();C(8,163,1,8);C(1,15,0,8);C(3,11,0,7);C(2,20,0,1);C(0,65,1,2);C(27,51,0,6);C(55,79,0,2);F();
		C(31,109,1,1);C(5,36,1,8);C(1,264,0,2);F();
		C(58,80,1,2);C(0,5,0,5);C(0,21,1,1);C(10,57,0,2);C(0,255,0,3);C(0,275,1,2);C(45,351,1,2);C(0,21,1,1);F();
		C(18,24,1,1);C(2,9,0,1);C(45,355,0,2);C(0,17,1,6);C(2,4,1,3);F();
		C(3,4,2,5);C(0,12,0,6);C(82,434,1,5);F();
		C(0,18,0,1);C(0,255,1,3);C(0,255,1,2);C(111,653,1,2);C(15,265,1,8);C(15,230,0,8);C(0,253,0,4);F();
		C(114,657,1,7);F();
		C(114,650,1,2);C(19,30,1,4);C(6,712,1,5);C(2,27,1,4);F();
		C(6,712,1,5);C(0,147,1,6);C(19,717,1,8);C(1,48,1,1);C(0,255,1,2);C(99,142,1,4);C(0,255,0,2);C(82,434,1,5);C(46,54,1,8);F();
		return 3.411512;
	}

	double compute_indices_29 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		B();C(0,4,2,5);C(2,5,2,5);F();
		C(0,4,1,1);C(0,10,1,5);C(0,7,0,2);F();
		C(17,26,1,3);F();
		C(22,28,0,7);C(14,17,1,7);F();
		C(5,36,0,2);C(0,44,1,3);C(0,12,0,8);C(21,47,0,5);C(17,26,1,3);C(0,17,0,5);C(0,1,0,4);F();
		C(5,13,1,7);B();C(0,17,1,1);C(13,58,0,4);C(8,12,0,4);C(4,38,1,8);C(2,13,1,8);F();
		C(0,47,0,2);C(0,60,0,3);C(34,42,0,5);C(0,202,1,8);F();
		C(3,72,0,3);C(0,44,0,1);C(6,19,1,3);F();
		C(72,91,1,5);C(1,2,0,8);C(0,23,0,6);C(7,19,0,2);C(8,70,0,1);C(23,31,1,7);F();
		C(5,46,0,7);C(8,96,0,4);C(0,5,1,1);C(0,7,0,2);B();C(0,2,0,2);C(0,4,1,7);C(0,12,1,1);F();
		B();B();C(2,9,0,2);C(19,112,0,1);C(21,27,0,5);C(11,20,1,5);C(2,55,1,1);F();
		C(18,57,1,4);C(0,20,0,1);C(0,116,0,1);B();C(16,19,1,7);C(22,47,1,3);C(10,15,1,6);F();
		C(0,116,0,3);B();C(20,108,1,3);B();C(4,38,0,4);C(16,18,1,1);C(7,16,0,8);F();
		C(13,16,1,6);C(45,136,0,3);C(0,47,0,1);C(13,16,1,6);F();
		C(0,17,0,7);B();C(0,16,1,6);C(7,41,0,7);C(0,10,0,1);C(2,55,0,1);C(0,1,0,3);C(66,139,0,3);C(5,13,1,4);C(6,21,1,1);C(44,103,1,8);F();
		C(0,23,1,4);C(1,19,0,5);C(35,67,0,8);C(0,6,0,3);C(0,14,0,5);C(0,14,0,1);C(59,146,0,3);F();
		C(90,153,0,2);C(15,38,0,7);C(0,8,1,1);C(2,9,1,4);C(1,6,1,4);C(2,5,1,4);C(0,30,0,7);F();
		C(0,11,0,1);B();C(6,12,1,6);C(43,154,1,1);C(5,43,0,3);C(1,20,0,4);C(0,16,1,8);C(0,6,1,7);F();
		C(31,70,0,3);C(8,163,1,2);C(4,38,1,7);C(2,8,0,2);C(0,1,1,6);C(10,44,0,8);C(15,25,0,7);F();
		C(0,4,0,6);C(8,21,1,4);C(23,164,1,6);C(9,221,1,6);C(0,16,1,2);C(15,25,0,5);C(0,8,0,5);F();
		B();C(11,33,1,2);C(0,18,1,6);C(0,34,0,6);C(2,6,0,1);C(0,11,0,2);C(0,44,0,7);C(6,19,1,8);C(11,240,1,6);F();
		B();C(1,21,0,4);C(3,8,1,2);C(15,38,0,5);C(12,18,0,6);C(8,15,0,4);C(0,6,1,4);C(20,108,1,3);C(0,17,0,3);C(0,3,0,2);C(3,246,0,4);C(0,26,0,4);C(7,15,1,4);F();
		C(0,28,0,8);C(0,2,2,3);C(0,40,1,4);C(0,16,1,6);C(25,46,2,5);C(22,31,0,8);C(7,172,1,3);C(31,55,2,2);C(0,255,1,3);F();
		C(9,15,1,3);C(14,258,0,1);F();
		C(0,2,1,6);C(46,147,1,1);C(72,260,1,3);C(6,19,1,7);C(0,12,1,7);F();
		B();B();B();C(39,146,0,1);C(62,64,0,5);C(26,34,1,8);C(0,262,0,5);C(0,262,0,5);F();
		C(21,102,0,1);C(29,30,0,8);C(7,14,0,1);C(27,51,0,6);C(53,61,0,4);C(0,17,0,1);C(14,21,0,6);C(10,28,0,5);C(0,5,1,4);C(82,264,1,3);F();
		B();B();C(0,17,1,1);C(5,9,1,6);C(13,58,0,4);C(8,12,0,4);C(2,13,1,8);C(77,264,1,3);C(38,57,0,7);F();
		C(0,43,1,7);C(15,53,0,4);C(6,26,0,6);B();C(14,21,0,8);C(82,434,1,5);C(0,59,0,1);F();
		C(16,18,1,1);C(0,32,0,4);C(18,19,1,8);C(53,61,0,4);C(82,434,1,5);C(0,45,0,8);C(56,86,1,5);C(2,4,1,2);C(22,63,0,1);F();
		C(0,8,0,6);C(39,85,0,2);C(3,39,1,8);C(111,144,1,6);C(0,12,1,7);C(2,9,1,4);C(3,39,1,8);C(8,20,0,2);C(7,10,0,3);C(64,495,1,5);C(0,4,0,3);C(9,221,1,6);C(76,417,1,5);C(39,85,0,2);C(1,264,1,8);F();
		C(2,20,1,1);C(2,9,1,7);C(6,27,0,6);C(101,141,0,2);C(0,17,1,5);C(0,3,1,7);C(104,652,1,2);C(8,39,0,6);F();
		return 1.016436;
	}

	double compute_indices_30 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,6,2,7);C(0,5,1,1);F();
		C(0,10,0,2);C(0,13,1,1);C(2,4,2,1);F();
		C(0,19,0,4);C(0,19,0,4);F();
		B();C(1,24,0,5);B();C(8,12,0,2);C(0,21,1,8);F();
		C(0,36,0,1);C(0,4,1,4);F();
		C(9,13,0,1);C(0,38,1,6);F();
		C(19,41,1,4);F();
		C(0,42,0,7);B();C(2,5,0,4);C(0,6,0,3);F();
		C(10,22,0,3);C(36,52,0,1);F();
		C(6,52,1,8);C(0,18,1,5);C(41,53,1,1);B();C(0,2,0,2);C(17,27,1,8);B();C(8,12,1,1);C(6,11,1,1);F();
		C(0,7,0,8);C(12,18,0,7);C(7,8,1,7);C(0,17,1,6);C(0,60,0,3);B();C(13,23,1,2);C(4,5,1,8);C(4,36,1,3);F();
		C(0,74,1,2);C(32,52,1,2);C(10,74,0,3);B();C(0,8,0,5);C(5,43,0,2);F();
		C(14,105,0,6);C(0,24,1,2);F();
		C(20,108,1,5);F();
		C(0,15,0,2);C(8,112,0,6);C(0,91,0,8);F();
		C(2,17,1,5);B();C(0,8,1,4);C(99,142,0,2);C(4,6,0,4);C(4,91,0,7);C(0,17,1,8);F();
		C(111,147,0,3);F();
		C(4,144,1,7);C(43,154,1,1);C(31,75,1,8);C(9,34,0,7);F();
		C(4,34,1,1);C(56,86,1,5);C(8,22,1,2);C(22,63,0,6);C(14,258,0,1);C(0,38,0,8);F();
		C(0,260,0,1);C(0,14,0,7);F();
		C(0,17,0,7);B();C(6,28,1,4);C(0,3,1,8);C(21,159,0,7);C(0,262,0,5);F();
		C(1,264,1,7);F();
		C(17,26,1,3);C(0,3,0,1);C(38,54,0,3);C(0,255,0,7);C(82,264,1,3);C(2,33,1,2);C(61,78,0,8);C(0,20,1,5);F();
		C(0,91,0,8);F();
		C(45,355,1,2);F();
		C(0,4,0,7);B();C(76,417,1,5);C(9,91,0,4);C(0,9,0,2);C(1,6,1,4);C(0,59,0,1);C(36,48,0,7);C(21,102,0,1);F();
		C(0,3,0,7);C(0,54,1,2);C(111,144,1,6);C(82,434,1,2);F();
		C(105,665,0,2);F();
		C(112,130,0,2);C(0,3,1,1);C(8,718,1,4);B();C(0,8,1,1);F();
		C(23,31,1,6);C(10,11,1,4);C(0,13,0,3);C(0,6,1,1);C(0,20,0,1);C(0,116,1,1);C(0,116,0,1);C(5,13,1,4);C(6,27,0,4);C(7,19,0,2);C(119,806,0,7);F();
		C(0,14,1,5);C(1,7,0,5);C(97,844,0,2);C(0,42,0,5);F();
		B();C(23,31,0,1);C(29,53,1,3);C(0,46,0,8);C(0,10,1,1);C(0,259,1,3);C(97,844,1,1);F();
		return 4.014323;
	}

	double compute_indices_31 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(3,8,0,3);C(0,2,0,4);C(0,1,1,1);B();C(2,5,2,3);F();
		C(3,54,0,8);C(32,56,1,1);C(7,19,0,2);F();
		C(0,60,1,1);C(8,13,1,4);C(0,2,1,1);C(0,40,0,7);C(0,4,0,2);C(0,6,1,6);C(0,11,1,1);F();
		C(53,61,0,8);F();
		C(7,31,0,4);C(3,8,0,3);C(0,13,0,3);F();
		C(0,29,0,6);C(56,346,1,6);C(10,12,0,6);C(3,214,1,2);C(58,702,1,2);F();
		C(0,8,1,7);C(0,3,1,6);C(5,46,0,4);F();
		C(6,39,0,1);C(0,155,1,5);F();
		C(23,31,0,1);C(0,9,1,2);C(22,28,1,1);F();
		C(1,11,0,7);C(20,28,0,4);F();
		C(16,25,0,6);C(0,3,0,4);B();F();
		C(0,1,1,3);C(16,18,1,6);C(0,58,1,3);C(0,23,1,3);C(34,42,1,2);C(0,68,1,3);F();
		C(0,14,0,5);C(0,31,1,2);C(11,25,1,5);C(0,21,1,5);C(2,6,1,2);F();
		C(5,36,0,2);F();
		C(0,10,0,8);B();C(2,36,1,2);F();
		C(0,7,1,1);B();C(0,31,0,4);C(20,26,0,7);C(0,7,0,7);F();
		C(18,24,0,7);C(0,3,1,3);C(0,6,0,1);C(0,6,0,1);F();
		C(0,1,0,3);C(17,26,0,3);B();C(0,4,1,8);F();
		B();C(0,14,0,5);C(7,19,0,2);C(0,6,0,4);C(6,11,1,4);C(2,8,0,4);C(6,29,0,2);C(5,33,1,3);C(5,33,1,3);F();
		C(2,5,0,4);B();C(0,8,0,3);F();
		C(16,24,0,6);C(24,31,0,6);C(0,8,0,1);C(0,2,0,1);C(1,3,1,2);B();F();
		C(3,15,1,1);C(0,32,0,7);C(0,1,1,4);C(0,36,0,7);F();
		C(7,25,1,3);C(9,13,0,3);C(34,42,1,2);F();
		C(27,71,1,8);C(0,9,0,3);C(8,11,1,4);C(25,46,1,8);C(0,17,1,6);B();B();C(0,8,0,3);C(0,116,0,3);F();
		C(20,30,0,4);C(0,5,1,2);C(0,4,0,5);C(30,164,0,6);C(48,69,0,5);F();
		C(8,18,0,2);C(56,76,1,5);F();
		C(21,47,0,5);C(15,25,0,4);F();
		C(13,18,0,8);C(6,40,0,5);C(17,26,1,3);F();
		C(6,9,0,8);C(11,30,0,4);C(10,19,0,4);C(0,68,0,4);C(0,1,2,5);C(31,70,0,6);C(21,27,0,4);C(0,4,2,3);F();
		C(2,9,0,3);F();
		C(0,9,1,2);C(0,32,1,7);F();
		C(20,22,1,2);F();
		return 2.160476;
	}

	double compute_indices_32 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(5,30,0,2);C(10,22,1,4);B();F();
		C(0,5,0,3);F();
		C(6,11,0,4);C(11,30,0,4);C(0,39,1,6);C(11,19,1,5);C(0,39,1,6);C(4,5,1,4);C(9,16,1,3);C(1,12,1,3);C(16,40,0,8);F();
		B();B();C(18,24,0,4);C(4,6,1,7);C(0,8,0,3);C(34,42,1,2);F();
		C(0,16,0,5);C(0,42,0,8);C(5,43,0,3);F();
		C(44,55,2,1);C(0,13,0,7);C(0,2,1,4);F();
		C(49,51,0,1);C(14,21,0,2);C(24,31,1,6);B();C(7,222,0,6);F();
		C(11,246,0,8);C(46,147,1,4);F();
		C(6,21,1,1);C(9,18,1,2);C(9,18,1,4);C(0,90,0,7);C(2,55,0,1);C(2,249,1,5);C(0,1,0,3);C(24,31,1,5);F();
		C(31,41,0,8);F();
		C(0,8,0,6);C(39,85,0,2);C(1,264,1,8);F();
		C(0,36,1,7);C(15,16,1,4);C(0,11,0,3);C(42,62,1,2);C(88,411,1,5);C(2,6,0,2);C(51,174,1,7);C(0,11,1,4);B();B();B();F();
		C(11,17,1,1);C(0,2,1,8);C(37,57,0,6);C(88,411,1,5);F();
		C(5,13,1,4);C(6,27,0,4);C(7,19,0,2);C(82,434,1,5);C(52,63,1,7);B();C(82,434,1,5);C(52,63,1,7);C(2,8,0,2);C(0,257,1,3);F();
		C(5,13,0,4);C(17,33,1,6);C(12,18,2,4);C(123,646,1,5);C(9,266,1,8);C(45,351,1,2);C(82,434,1,4);C(2,20,0,1);C(0,60,0,3);C(0,16,1,8);C(2,7,0,8);F();
		C(0,3,0,3);C(22,174,1,8);C(6,21,0,5);C(0,26,1,2);B();C(9,17,0,1);C(1,12,1,4);C(114,657,1,7);C(8,15,0,7);F();
		C(7,11,0,4);C(76,417,1,5);C(0,7,0,4);B();C(2,9,0,1);B();C(0,255,1,2);C(9,221,1,6);C(56,346,1,1);C(114,650,1,2);B();C(20,108,1,3);F();
		C(3,214,1,2);C(115,640,1,8);C(0,2,0,1);C(68,425,1,1);C(7,18,1,8);C(107,128,0,7);C(54,704,1,2);B();F();
		C(47,715,1,6);C(141,639,1,5);C(62,64,0,3);C(60,63,2,6);F();
		C(0,8,0,8);C(0,4,1,5);C(8,26,1,7);C(114,650,1,2);C(6,712,1,5);C(82,434,1,5);C(0,153,0,8);C(0,5,1,7);C(48,63,2,5);F();
		C(13,14,1,5);C(0,231,0,5);C(0,18,1,3);C(18,22,0,5);C(8,718,1,4);C(10,11,1,2);C(46,147,1,1);C(0,32,1,7);C(8,12,0,5);C(82,434,1,5);C(0,1,0,4);C(111,144,1,6);C(0,208,1,6);C(0,15,1,4);C(8,163,1,3);C(76,417,1,5);C(0,11,1,1);F();
		C(15,23,0,5);C(55,725,1,2);C(27,78,1,4);C(84,409,1,5);C(2,5,1,4);C(14,25,0,4);C(56,86,0,2);C(56,86,1,7);F();
		C(8,718,1,4);B();C(23,31,0,3);C(17,751,1,7);C(76,417,1,1);C(57,503,1,2);F();
		C(0,719,1,4);C(18,25,0,4);C(27,78,1,4);C(75,407,0,8);C(75,407,0,8);C(107,827,1,1);C(2,5,1,4);F();
		C(76,435,1,8);C(80,284,1,7);C(8,10,1,8);C(84,830,1,4);C(0,4,1,7);C(16,19,0,2);C(0,4,1,7);C(0,14,1,4);C(3,7,1,8);C(0,8,0,8);C(1,6,1,2);C(6,9,1,4);C(11,22,0,3);F();
		C(7,17,0,7);C(8,163,1,3);B();C(95,837,1,6);C(19,30,0,3);C(0,39,0,6);C(9,18,0,4);C(5,20,0,4);C(19,42,1,2);C(19,65,1,8);F();
		C(10,28,1,5);C(120,838,1,6);F();
		C(76,417,1,1);C(0,4,1,5);C(7,10,0,4);C(1,264,1,7);C(4,86,1,3);C(68,808,1,6);C(37,502,1,5);C(27,38,1,1);C(11,729,1,8);C(1,844,1,3);C(102,653,1,7);C(0,6,1,3);C(0,1,1,1);C(7,10,1,1);C(22,47,0,3);F();
		C(115,660,1,6);C(9,221,1,6);C(40,42,1,5);C(105,851,1,3);C(57,503,1,2);C(6,19,0,8);C(11,240,1,6);C(0,4,1,7);C(0,12,1,1);C(22,35,0,2);F();
		C(5,43,0,3);C(52,72,1,5);C(0,10,1,7);C(20,72,1,8);C(105,851,1,3);C(2,6,0,7);C(0,44,1,7);C(8,163,1,8);C(0,21,1,8);C(25,46,1,2);C(0,40,0,4);C(0,40,0,4);F();
		C(5,13,0,3);C(45,351,1,2);C(103,852,1,2);C(76,417,1,5);C(25,46,0,2);F();
		C(37,57,1,1);C(106,872,1,1);C(0,24,0,5);C(0,3,0,3);C(0,2,1,6);C(0,4,0,4);C(23,31,0,1);B();C(46,147,0,1);F();
		return 1.580490;
	}

	double compute_indices_33 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,6,1,3);C(0,3,0,8);F();
		C(8,31,1,1);F();
		C(0,14,0,3);C(0,11,0,6);F();
		C(14,19,2,4);C(1,13,1,7);B();F();
		C(0,20,2,3);C(0,8,1,7);C(0,6,0,8);F();
		C(0,21,1,1);C(0,21,0,1);F();
		C(2,9,0,4);C(0,21,1,3);F();
		C(12,24,0,6);C(0,7,0,4);C(0,3,2,2);C(6,25,0,5);F();
		C(30,164,0,2);C(9,24,0,4);F();
		C(0,6,0,2);B();C(26,29,1,4);C(4,38,0,8);C(20,24,1,1);C(18,24,1,4);F();
		C(0,2,2,3);C(0,11,0,5);C(25,44,0,3);F();
		C(48,183,1,7);C(37,49,1,5);F();
		C(0,60,0,1);C(0,18,0,8);F();
		C(0,60,0,3);C(1,11,1,8);C(21,159,0,8);F();
		C(64,70,1,2);C(57,82,0,1);F();
		C(39,63,1,1);C(13,25,1,1);C(31,38,0,3);C(0,30,0,7);C(6,27,0,5);F();
		C(10,71,1,2);C(0,23,1,3);B();F();
		C(6,21,1,1);C(0,40,0,4);C(0,40,0,4);C(31,75,0,1);C(7,12,1,3);C(0,2,1,3);C(0,5,0,3);F();
		C(61,78,0,1);C(17,26,1,6);C(1,8,0,2);B();C(0,15,0,5);C(49,82,0,6);F();
		B();C(11,20,1,1);C(3,89,1,6);F();
		C(22,63,0,6);C(7,21,0,1);C(0,91,0,8);F();
		C(111,144,1,2);F();
		C(91,125,0,6);C(0,70,1,8);F();
		C(34,42,0,2);C(51,159,0,4);C(0,5,0,2);C(8,20,0,7);C(39,85,1,2);B();F();
		C(6,13,0,1);C(0,7,0,5);C(8,12,0,7);C(3,8,0,6);C(0,23,0,4);C(0,18,1,1);C(8,163,1,6);C(0,7,1,8);C(22,63,1,1);F();
		C(0,6,1,2);C(0,38,0,6);C(10,12,0,6);C(0,18,0,5);C(9,42,1,4);C(3,214,1,2);F();
		C(9,221,1,5);C(9,14,1,5);C(0,2,0,4);B();B();F();
		C(83,225,1,2);F();
		C(13,27,0,1);C(0,4,1,4);C(7,9,0,5);C(0,253,1,4);C(111,144,1,6);C(7,13,0,6);F();
		C(0,255,1,8);C(26,35,1,7);F();
		B();C(6,28,1,4);C(0,262,1,5);F();
		C(72,144,0,1);F();
		return 2.505239;
	}

	double compute_indices_34 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,16,0,8);C(29,33,1,4);C(1,12,0,1);B();F();
		C(0,7,0,1);C(0,21,1,4);C(0,2,2,5);C(61,78,0,7);C(0,47,0,1);F();
		C(0,2,0,1);C(8,16,0,5);C(0,275,0,5);C(0,245,1,3);C(0,16,1,4);F();
		C(0,2,0,4);B();C(0,18,0,4);C(0,4,1,4);C(10,28,1,2);F();
		B();C(0,32,1,7);C(22,47,1,3);F();
		C(27,71,1,8);C(25,46,1,8);C(0,17,1,6);B();B();C(0,8,0,3);C(0,116,0,3);F();
		C(0,10,0,3);F();
		C(108,141,0,6);B();F();
		C(72,260,0,3);F();
		C(0,8,1,1);C(10,11,1,5);C(14,105,0,6);C(22,47,1,3);F();
		C(0,255,0,1);B();C(29,33,1,7);C(6,244,1,2);C(9,10,0,8);C(51,70,1,2);C(0,7,0,2);C(0,2,0,8);C(5,25,0,2);F();
		B();C(9,221,0,6);F();
		C(15,16,1,4);F();
		C(8,12,1,4);C(5,69,0,1);C(0,13,1,6);C(0,37,1,1);F();
		C(12,24,1,2);F();
		C(0,40,1,6);C(5,24,0,2);B();C(0,32,1,7);C(22,47,1,3);F();
		C(0,10,1,1);C(8,718,1,4);F();
		C(4,10,0,5);C(4,6,0,4);C(0,25,1,4);C(61,78,0,3);C(0,47,0,1);F();
		C(24,30,1,2);F();
		C(15,23,1,1);C(3,15,1,1);C(0,60,1,4);C(0,65,1,6);C(57,63,2,3);F();
		C(15,70,0,7);B();C(5,28,0,6);C(15,70,0,7);F();
		C(17,29,0,3);C(0,15,1,2);C(0,1,1,2);F();
		C(9,22,1,2);C(11,20,1,1);C(3,89,0,8);F();
		C(25,46,1,2);C(0,40,0,4);F();
		B();C(0,255,1,2);C(76,417,1,5);C(56,346,1,1);C(114,650,1,2);B();C(1,12,1,4);F();
		C(0,58,0,2);C(31,37,0,2);C(3,8,1,3);C(0,10,0,8);F();
		C(39,85,0,2);B();C(15,25,0,4);F();
		C(12,24,1,5);C(0,10,1,4);C(10,22,0,3);F();
		C(1,3,1,2);C(0,116,0,3);C(3,6,0,3);C(11,30,0,4);F();
		C(61,78,0,3);C(0,31,0,5);C(0,208,1,5);C(7,8,1,4);C(1,6,1,2);C(0,4,1,7);F();
		B();C(77,264,0,3);C(0,12,1,7);C(46,147,1,1);F();
		C(0,36,0,7);F();
		return 2.359873;
	}

	double compute_indices_35 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,6,2,7);C(0,5,1,1);F();
		C(3,5,0,1);B();C(0,23,1,2);C(47,63,2,8);C(2,8,1,8);F();
		B();C(14,21,0,2);C(0,18,1,8);B();C(3,164,0,3);C(11,17,0,8);F();
		C(18,31,0,8);F();
		C(4,8,0,5);C(5,36,0,2);C(5,36,0,2);C(2,55,0,2);F();
		C(4,45,0,4);C(0,12,1,7);C(0,2,1,1);F();
		C(0,12,0,5);C(6,47,1,1);C(43,59,1,3);F();
		C(3,61,1,7);C(0,41,0,3);F();
		C(22,63,0,1);C(0,18,1,3);C(10,26,1,1);F();
		C(51,70,0,2);F();
		C(0,17,0,8);C(39,85,0,2);C(2,9,1,4);C(0,14,1,6);C(0,15,0,1);C(34,42,1,6);F();
		C(0,3,1,8);C(4,11,1,5);C(0,2,1,5);C(21,102,0,1);B();F();
		C(0,107,0,6);B();C(0,8,0,3);C(0,7,0,4);B();B();F();
		C(0,14,1,3);C(99,142,0,4);C(0,14,1,3);C(2,7,1,6);F();
		C(0,14,1,7);F();
		C(36,100,1,1);B();C(82,241,0,3);C(62,64,0,5);F();
		C(0,6,0,8);C(20,246,1,6);C(11,22,1,7);C(22,31,0,4);C(0,208,1,6);C(5,23,1,2);C(0,8,0,2);F();
		C(0,249,1,2);C(40,47,1,5);F();
		C(0,5,1,4);C(0,3,1,3);C(9,16,0,3);B();C(0,6,0,4);C(0,15,1,6);C(0,255,1,1);C(17,38,0,2);F();
		C(0,5,2,4);C(0,259,1,3);C(4,6,1,7);F();
		B();C(2,13,0,6);C(58,80,0,4);C(82,264,0,4);C(0,4,0,6);C(99,142,0,2);C(0,40,0,2);C(22,63,0,1);C(9,266,1,6);C(3,7,1,2);C(0,58,1,3);C(9,266,1,6);C(51,121,1,4);F();
		C(0,4,1,4);B();C(0,6,1,7);C(19,30,1,4);F();
		C(82,434,1,1);C(8,12,1,4);F();
		C(18,24,0,4);C(0,18,1,5);C(61,78,0,8);C(42,62,0,2);C(41,53,1,1);C(0,67,1,3);C(82,434,1,5);C(24,37,1,1);C(46,147,0,1);F();
		C(0,13,0,6);C(10,15,0,6);C(0,6,1,5);C(12,25,1,4);C(20,246,1,6);C(0,7,1,4);C(64,495,1,5);F();
		C(0,1,1,1);B();C(17,22,1,2);C(15,164,1,8);C(8,15,1,7);B();C(9,17,0,1);C(1,12,1,4);C(114,657,1,7);C(6,21,0,5);C(2,9,1,4);F();
		B();C(56,86,1,5);C(0,8,0,1);C(0,7,1,7);C(0,698,0,1);C(0,17,0,2);C(8,33,0,1);C(4,86,1,3);F();
		C(0,16,1,4);C(53,61,1,7);C(2,701,0,5);F();
		C(21,102,1,8);C(0,153,1,6);C(2,9,0,2);C(114,650,1,2);C(82,434,1,5);C(6,712,1,5);C(2,6,1,5);F();
		C(0,13,0,6);C(105,671,1,7);F();
		C(9,16,0,3);C(7,19,0,2);C(8,718,1,4);B();F();
		C(0,11,2,8);C(76,417,1,1);C(97,844,1,2);C(7,9,0,7);C(2,4,2,1);C(76,417,1,1);C(97,844,1,2);F();
		return 3.426419;
	}

	double compute_indices_36 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(111,144,1,2);C(0,17,0,5);C(2,25,1,7);C(0,13,1,4);C(12,18,1,2);C(3,6,1,1);C(4,5,0,8);C(8,39,1,1);F();
		C(8,20,0,4);C(3,7,0,2);C(3,8,0,5);C(0,6,0,8);C(0,60,0,4);C(0,255,0,1);F();
		C(2,9,1,4);C(2,9,1,4);C(0,8,0,1);C(14,21,0,2);C(0,1,1,6);C(0,8,0,4);C(0,24,1,3);C(0,1,0,8);C(2,3,0,5);F();
		C(103,135,1,6);C(2,7,0,6);C(0,2,0,1);C(2,55,0,1);C(9,16,0,3);F();
		C(2,5,2,3);C(0,18,1,3);C(0,1,0,4);B();C(0,17,0,8);B();C(25,46,0,2);C(1,11,0,5);F();
		C(0,7,0,4);C(0,6,0,5);C(0,1,0,1);C(0,18,1,1);C(8,163,1,6);C(0,7,1,8);C(22,63,0,1);F();
		C(0,5,1,8);C(0,4,0,6);C(0,12,1,5);C(2,20,0,1);C(0,65,1,2);C(0,16,1,1);C(0,60,0,1);C(2,20,0,1);F();
		B();C(39,85,0,2);F();
		C(43,135,0,5);B();C(46,147,1,1);F();
		C(2,55,0,1);F();
		C(0,12,1,2);C(11,30,1,8);B();C(8,26,1,1);C(77,264,1,3);F();
		C(0,14,0,2);F();
		C(0,16,1,1);C(4,7,1,4);F();
		C(70,84,0,3);F();
		C(9,18,0,4);C(5,20,0,4);C(99,142,0,2);F();
		C(0,7,0,1);C(0,21,1,4);C(0,2,2,5);C(61,78,0,1);C(0,47,0,1);B();F();
		C(63,74,1,6);F();
		C(5,13,1,3);C(71,90,0,7);C(2,8,1,8);C(69,81,1,2);C(0,15,1,8);F();
		C(48,66,0,7);F();
		C(23,31,0,1);B();C(0,47,0,1);C(0,3,1,3);C(0,11,1,4);F();
		B();C(2,9,1,8);C(2,9,1,4);C(0,38,1,1);C(0,7,1,8);C(15,16,1,4);C(8,12,1,4);C(9,16,0,3);C(5,69,0,1);C(0,13,1,6);C(0,256,0,4);C(12,24,1,2);F();
		C(24,31,1,6);C(57,63,2,1);B();C(99,107,1,1);C(4,6,0,4);F();
		C(0,2,1,3);C(0,7,1,4);C(11,30,1,1);C(2,55,0,1);C(7,19,0,2);C(0,12,1,2);B();C(0,16,0,4);C(8,26,1,1);C(56,346,1,3);F();
		C(7,19,1,2);C(36,48,1,6);C(0,94,0,7);C(0,14,0,6);F();
		C(53,61,0,4);C(0,255,0,1);C(5,9,1,5);C(7,18,0,2);F();
		C(62,91,1,3);C(31,64,1,3);C(0,64,0,4);F();
		C(14,36,0,2);C(9,12,0,4);C(1,7,0,5);F();
		C(0,8,0,1);B();C(14,30,0,1);C(9,14,0,1);B();C(27,30,1,6);F();
		C(129,648,1,8);C(44,103,1,8);F();
		C(6,17,0,8);C(0,37,1,2);C(0,2,0,4);C(0,91,0,8);C(0,275,1,2);F();
		C(27,41,1,1);C(0,4,0,6);C(1,13,0,2);C(11,29,0,3);B();C(59,711,1,2);F();
		C(1,24,0,2);C(34,42,0,2);C(5,13,1,4);C(18,19,1,8);B();F();
		return 4.444104;
	}

	double compute_indices_37 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,7,0,4);C(0,10,1,4);C(0,15,0,6);C(0,3,1,7);C(0,14,1,2);C(0,4,0,1);C(0,4,1,7);B();B();C(0,3,0,3);C(0,7,0,8);F();
		C(22,31,0,7);B();C(0,6,0,8);F();
		C(31,37,0,7);C(0,11,0,2);C(8,20,0,7);B();C(0,9,0,4);C(7,16,0,8);F();
		C(0,23,0,7);C(6,8,1,1);C(32,55,0,2);F();
		C(0,40,0,4);C(4,19,1,3);C(25,46,1,2);C(0,6,1,7);C(0,40,1,4);C(0,7,0,2);B();C(2,55,0,7);C(3,10,0,3);C(37,57,0,6);C(48,49,0,7);C(53,61,0,2);C(0,232,1,1);F();
		C(0,60,0,3);C(0,16,1,6);C(3,22,0,8);C(2,20,0,6);C(0,65,0,2);F();
		C(61,78,0,4);C(22,28,0,7);C(34,42,1,3);C(0,257,0,4);F();
		C(56,86,0,5);C(7,14,1,1);C(4,86,1,1);F();
		C(0,11,1,6);C(2,9,0,2);C(0,1,2,5);C(5,16,1,7);C(0,89,0,3);C(0,50,0,8);C(0,68,0,4);C(36,78,1,6);F();
		C(17,26,0,3);C(1,6,0,7);C(0,102,0,3);C(0,18,0,7);B();B();C(2,55,1,1);C(7,14,1,6);F();
		C(0,24,1,1);C(0,25,0,2);B();C(4,26,1,4);C(112,130,0,2);C(15,38,1,6);C(0,16,0,8);C(3,8,1,2);C(2,20,0,1);C(8,15,0,4);B();C(9,16,0,3);C(15,16,1,4);C(8,12,1,8);C(78,86,1,8);F();
		C(1,11,0,7);C(5,46,0,4);C(66,139,0,3);C(18,24,1,6);F();
		C(0,10,1,1);C(25,46,0,8);C(99,142,0,3);C(4,6,1,4);C(1,12,1,3);F();
		C(36,48,0,1);C(46,147,0,6);C(24,31,1,6);C(0,2,0,2);C(0,17,1,6);B();C(0,8,0,3);C(27,71,1,8);C(0,116,0,3);C(11,41,1,5);C(40,54,0,3);F();
		C(3,13,0,1);C(0,230,0,4);C(8,20,1,5);C(0,6,0,5);F();
		C(0,2,0,1);C(0,2,1,1);C(0,1,1,1);C(3,89,0,8);C(16,20,1,1);C(0,2,0,4);C(0,233,0,2);C(0,20,1,5);C(48,76,1,5);C(27,51,1,4);F();
		C(105,217,0,2);C(0,43,1,2);C(0,240,0,4);C(21,22,1,6);F();
		C(0,16,1,3);C(1,9,1,4);C(0,1,0,2);C(0,109,0,3);C(0,255,0,3);C(83,225,1,2);C(10,12,1,4);F();
		C(0,1,0,5);C(0,111,0,3);C(1,36,1,1);C(2,55,1,6);C(0,259,1,3);F();
		C(2,12,1,6);C(0,33,1,8);C(61,78,0,8);C(53,61,1,8);C(0,42,0,5);C(4,38,1,1);C(42,62,0,2);C(41,53,1,1);C(0,1,0,3);C(0,9,0,5);C(0,27,1,2);C(14,266,0,6);C(0,6,1,2);C(5,69,0,1);C(0,1,0,4);C(0,9,0,7);C(0,40,0,6);C(6,52,1,8);F();
		C(0,5,1,5);C(2,55,0,1);C(35,81,1,2);C(5,13,0,4);C(6,19,1,4);C(0,4,0,2);C(14,266,1,5);F();
		B();C(2,4,1,2);C(0,4,1,3);C(0,37,1,2);C(22,63,0,1);C(0,281,1,6);C(8,163,0,2);F();
		C(25,26,1,7);C(0,23,1,3);C(18,24,0,4);C(5,46,0,7);C(0,18,0,1);C(8,96,0,4);C(2,287,0,3);C(0,10,1,4);C(11,246,1,6);F();
		C(46,147,1,1);C(29,33,1,4);C(12,24,1,7);C(45,351,1,2);C(20,24,0,2);C(19,65,1,3);F();
		C(76,417,1,1);C(0,10,0,7);C(0,2,1,1);F();
		C(0,3,1,1);C(0,5,1,7);C(82,434,1,1);C(15,18,0,2);C(0,4,1,1);B();F();
		C(75,407,0,8);C(1,24,1,8);F();
		B();C(48,69,0,1);C(11,44,2,7);C(114,650,1,2);C(0,5,1,2);B();C(0,10,0,7);F();
		C(0,10,1,1);C(6,27,0,4);C(0,13,1,4);C(8,39,0,6);C(2,54,1,1);C(114,650,1,2);F();
		C(59,711,1,2);C(1,12,1,4);C(0,14,0,7);C(0,11,0,8);B();C(12,18,1,2);C(2,26,1,1);C(0,14,1,5);C(0,26,1,1);C(8,15,1,7);C(0,1,0,4);C(0,11,1,2);C(1,12,1,4);C(0,6,1,1);C(1,11,0,5);F();
		C(27,49,0,3);C(2,9,1,4);C(1,6,1,4);C(48,63,1,5);C(0,40,0,4);C(66,139,0,3);C(10,54,1,2);C(4,6,0,4);C(0,14,0,8);C(1,11,1,7);C(6,712,1,5);C(0,153,1,6);C(8,718,1,4);F();
		C(5,13,0,3);C(103,852,1,2);C(45,351,1,2);C(76,417,1,5);F();
		return 2.061970;
	}

	double compute_indices_38 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(0,54,0,3);C(0,3,0,3);C(6,712,1,5);F();
		C(23,31,0,4);F();
		C(0,4,1,6);C(0,3,0,3);F();
		C(0,2,0,4);C(0,1,1,6);F();
		C(1,10,0,8);C(113,640,0,6);F();
		C(14,258,0,1);C(6,8,1,1);C(14,258,0,1);C(6,14,1,1);F();
		C(8,718,0,2);C(0,8,1,2);C(16,40,0,8);C(0,15,0,4);F();
		C(5,58,0,7);B();C(0,6,0,8);F();
		C(21,102,0,1);F();
		C(0,17,0,1);B();C(71,884,1,2);F();
		C(0,15,1,6);C(11,240,1,8);C(8,302,0,6);C(23,31,0,3);F();
		C(0,6,0,1);C(45,351,0,2);F();
		C(34,42,0,5);C(39,85,0,2);C(12,13,1,2);C(6,192,0,5);F();
		C(14,258,0,1);F();
		B();C(100,162,0,3);C(52,95,0,5);B();C(6,14,0,8);C(41,53,1,1);F();
		C(59,719,0,2);C(2,6,0,7);C(0,18,1,3);F();
		C(0,208,1,6);C(0,73,1,8);F();
		C(0,125,0,3);C(3,246,1,8);F();
		C(0,34,0,4);C(0,68,0,4);C(0,19,0,8);C(7,41,0,1);F();
		C(0,3,1,8);C(1,5,0,6);C(2,55,1,3);C(0,11,1,3);C(5,11,1,4);C(0,76,1,4);C(0,7,0,8);C(0,1,1,6);C(1,22,1,1);F();
		C(0,16,1,8);C(0,36,0,7);C(3,89,0,8);F();
		C(34,44,1,2);C(0,255,0,3);C(0,255,0,1);F();
		B();C(25,42,0,6);C(2,8,0,4);C(8,14,0,2);C(0,14,0,8);C(25,75,0,1);F();
		C(0,6,0,1);F();
		C(0,2,0,7);C(72,144,0,1);C(0,14,0,4);C(0,43,1,7);C(0,6,0,3);C(0,9,1,5);C(15,53,0,4);C(2,23,1,2);C(15,53,0,4);C(2,23,1,2);F();
		C(22,63,1,1);B();C(9,266,1,6);C(0,40,0,2);C(25,75,0,1);F();
		C(82,264,0,5);C(22,31,0,4);C(0,38,1,6);C(12,24,1,2);C(5,43,0,1);B();F();
		C(18,106,0,3);C(99,142,1,2);C(0,3,1,6);C(0,275,0,5);C(0,11,1,6);C(0,4,1,4);F();
		C(8,718,1,4);F();
		C(2,6,1,1);C(3,69,0,5);F();
		C(46,147,0,1);C(62,64,0,5);F();
		C(0,11,0,1);C(0,5,0,1);C(2,9,0,2);C(2,9,1,4);C(1,23,1,2);F();
		return 2.695175;
	}

	double compute_indices_39 (unsigned int pc, sampler_info *u) {
		unsigned int j = 0, t = 0;
		C(3,17,0,2);C(1,6,1,4);C(0,6,1,6);C(0,60,0,3);F();
		C(0,6,0,7);C(9,12,1,8);C(0,16,1,8);C(5,22,0,4);F();
		C(0,14,1,5);C(29,33,2,4);C(0,3,0,1);C(0,11,1,6);C(3,5,1,5);F();
		C(0,40,1,6);C(0,13,1,8);C(0,4,1,1);B();C(0,4,1,4);C(17,26,1,3);B();F();
		C(0,42,1,1);F();
		C(0,42,1,1);C(2,8,0,3);C(19,24,1,5);C(0,7,1,4);C(22,28,1,7);C(0,3,1,4);F();
		C(0,50,0,3);C(42,62,1,2);C(61,78,0,8);C(41,53,1,1);C(0,9,1,5);C(9,18,1,7);C(3,10,0,3);C(46,61,0,4);C(0,18,1,5);C(12,268,1,2);F();
		C(0,4,1,4);C(39,85,0,2);C(27,41,1,2);C(57,503,1,2);F();
		C(5,10,1,6);C(0,17,1,3);C(0,42,0,5);B();C(0,7,1,1);C(11,20,1,1);C(4,98,0,8);C(7,13,0,6);C(0,60,0,4);F();
		C(0,30,1,4);C(0,12,2,5);C(0,6,1,1);C(0,94,0,7);C(0,4,0,1);C(6,28,1,4);C(5,33,1,3);F();
		C(2,55,0,1);C(18,106,0,3);F();
		C(0,14,1,4);C(0,7,1,8);C(31,70,0,3);C(0,34,0,5);B();C(0,116,0,3);C(0,15,1,7);C(2,25,1,3);F();
		C(0,15,0,5);C(0,2,1,8);C(5,19,2,4);C(99,142,0,2);C(4,26,1,4);C(5,9,1,6);C(70,338,0,3);F();
		C(10,24,0,2);C(42,51,1,8);C(9,25,1,8);C(0,2,1,3);B();B();C(99,142,0,2);C(0,17,1,8);C(0,42,0,8);C(3,9,1,3);C(0,6,1,4);F();
		C(9,181,0,3);C(0,17,0,2);C(1,11,1,3);F();
		C(1,6,0,2);C(0,20,1,4);C(0,14,1,4);C(0,208,1,5);C(39,85,0,2);C(0,36,0,1);C(2,3,1,5);F();
		C(0,28,0,1);C(0,208,0,5);C(0,4,0,1);F();
		C(0,1,0,3);C(5,226,0,3);C(12,18,1,2);C(2,26,1,1);C(0,14,1,5);C(0,4,1,1);C(8,37,0,1);F();
		C(9,12,1,5);C(0,4,0,2);C(2,252,1,2);C(0,43,0,8);C(0,232,1,1);C(2,252,1,1);C(0,9,0,4);C(18,36,1,6);C(6,98,0,3);B();B();C(0,5,0,8);F();
		C(8,20,1,2);C(37,57,1,2);C(3,69,1,3);C(0,9,1,5);C(26,67,0,2);C(0,18,1,5);C(0,260,0,1);F();
		B();C(52,79,1,2);C(22,63,0,8);C(2,4,1,2);C(0,12,2,5);C(0,14,1,6);B();C(103,266,0,3);C(5,9,1,4);C(52,79,1,2);F();
		C(0,5,0,5);B();C(2,13,0,6);C(8,163,1,2);C(0,281,0,6);C(0,275,1,5);C(43,154,1,1);C(31,75,1,8);F();
		C(36,48,1,6);C(59,719,0,1);F();
		C(56,346,1,1);B();C(56,63,2,8);F();
		C(7,41,0,3);C(56,346,1,3);C(14,258,0,1);F();
		C(37,72,1,4);C(20,64,0,6);C(60,365,1,3);C(48,78,0,5);F();
		C(34,42,1,2);C(2,20,1,1);C(23,31,0,1);C(0,10,1,4);C(87,424,0,1);C(0,2,1,2);C(2,5,0,7);C(1,6,0,1);C(37,57,1,5);F();
		C(0,10,0,6);C(0,21,0,3);C(0,6,1,4);C(0,4,0,1);C(0,8,0,7);C(0,4,1,5);C(0,2,1,8);C(11,36,1,7);C(0,40,0,4);C(25,46,0,2);C(82,434,1,5);C(13,20,1,6);F();
		C(5,43,0,1);C(53,61,1,1);C(113,640,0,6);F();
		C(0,2,0,4);C(21,102,0,1);C(0,17,0,3);C(102,653,1,7);F();
		C(17,34,1,4);C(50,155,1,5);C(8,718,1,1);C(19,30,1,4);F();
		C(5,9,1,1);C(99,142,0,2);C(2,25,0,3);C(12,24,0,3);C(0,2,0,4);C(0,18,1,3);C(0,18,1,7);C(14,19,1,8);B();C(2,55,1,6);C(3,8,0,5);C(16,52,0,8);C(0,11,1,5);C(90,826,1,3);F();
		return 2.716582;
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
	}

	void compute_sums (unsigned int pc, sampler_info *u) {
		// initialize indices

		for (int i=0; i<num_tables; i++) u->indices[i] = 0;

		// for each sample, accumulate a partial index

		histories[0] = global_history;
		histories[1] = path_history;
		histories[2] = &callstack_history;
		double coeff = compute_indices (pc, u);
		// now we have all the indices. use them to compute
		// the weighted and unweighted sums

		// add in the local prediction

		if (local_table_size) {
			u->local_index = get_local_history (pc);
			int x = local_pht[u->local_index];
			u->sum += (int) (coeff * x);
			u->weighted_sum += (int) (coeff_local * x);
		}
	}

	// look up a prediction in the predictor

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

			if (filter[0][idx]) {
				u->prediction (false);
				u->weighted_sum = min_weight;
				u->sum = min_weight;
			} else if (filter[1][idx]) {
				u->prediction (true);
				u->weighted_sum = max_weight;
				u->sum = max_weight;
			}
		} else {
			// it's in both filters; have to predict it with
			// perceptron predictor
			compute_sums (pc, u);
		}
		if (new_branch) {
			// if this is a new branch, predict it statically

			u->prediction (static_prediction);
		} else {
			// give the prediction with the kind of sum that seems do
			// be doing well
			bool prediction;

			if (psel >= 0) 
				prediction = u->weighted_sum >= 0.0;
			else
				prediction = u->sum >= 0.0;
			u->prediction (prediction);
		}
		return u;
	}

	// update an executed branch

	void update (branch_info *hu, bool taken) {
		sampler_info *u = (sampler_info *) hu;
		bool 
			never_taken = false, 
			never_untaken = false;

		// an index into the filters

		unsigned int idx = 0;

		// we don't know that we need to update yet

		bool need_to_update = false;
		if (filter_size) {

			// get the index into the filters

			idx = hash (u->pc, 0) % filter_size;

			// the first time we encounter a branch, we will update
			// no matter what to prime the predictor

			need_to_update = !(filter[0][idx] || filter[1][idx]);

			// we have seen this branch with this sense at least once now

			filter[!!taken][idx] = true;

			// see if the branch has never been taken or never been not taken

			never_untaken = !filter[0][idx];
			never_taken = !filter[1][idx];
		}

		// we need to update if there is no filter or if the filter
		// has the branch with both senses

		need_to_update |= !filter_size || (filter[0][idx] && filter[1][idx]);
		if (need_to_update) {
			// was the prediction from the weighted sum correct?

			bool weighted_correct = (u->weighted_sum >= 0) == taken;

			// was the prediction from the non-weighted sum correct?

			bool correct = (u->sum >= 0) == taken;

			// keep track of which one of those techniques is performing better

			if (weighted_correct && !correct) {
				if (psel < 511) psel++;
			} else if (!weighted_correct && correct) {
				if (psel > -512) psel--;
			}

			// update the coefficients

			for (int i=0; i<num_tables; i++) {

				// see what the sum would have been without this table

				int sum = u->sum - table[i][u->indices[i]];

				// would the prediction have been correct?

				bool this_correct = (sum >= 0) == taken;
				if (correct) {
					// this table helped; increase its coefficient

					if (!this_correct) if (coeffs[i] < (max_exp-1)) coeffs[i]++;
				} else {
					// this table hurt; decrease its coefficient

					if (this_correct) if (coeffs[i] > -max_exp) coeffs[i]--;
				}
			}

			// get the magnitude of the sum times a fudge factor

			int a = abs ((int) (u->sum * coeff_train));

			// get a random number between 0 and 1

			double p = (rand_r (&my_seed) % 1000000) / 1000000.0;

			// if the branch was predicted incorrectly according
			// to the unweighted sum, if if the magnitude of the
			// some does not exceed some random value near theta,
			// then we must update the predictor

			bool do_train = !correct || (a - (p/2 * theta_fuzz)) < theta;
			if (do_train) {
				// train the global tables

				for (int i=0; i<num_tables; i++)
					table[i][u->indices[i]] = satincdec (table[i][u->indices[i]], taken, max_weight, min_weight);

				// train the local table

				if (local_table_size) {
					int x = local_pht[u->local_index];
					x = satincdec (x, taken, max_weight, min_weight);
					local_pht[u->local_index] = x;
				}
			}

			// adjust theta

			threshold_setting (u, correct, a);
		}

		// update global, path, and local histories

		shift_history (global_history, taken ^ !!(u->pc & 4));
		shift_history (path_history, (u->pc >> path_bit) & 1);
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
		200000000, 		// number of bytes allocated to predictor
		6, 		// bit width
		32, 		// number of tables
		24, 		// folding parameter, i.e. hash function width
		1<<21,		 // filter size
		24, 		// initial theta
		6, 		// shift shift
		7, 		// adaptive threshold learning speed
		2.25, 		// local predictor fudge factor
		1<<21, 		// local predictor table size
		16, 		// local predictor history length
		1<<20, 		// local predictor number of histories
		8.72, 		// theta taper
		1.004, 		// fudge factor 2
		1.000025800, 		// fudge factor 3
		131072, 	// maximum exponent for coefficient learning
		true,
		0x40000018,
		false, 
		5);
	}

	branch_info *u;

	bool GetPrediction(UINT32 PC) {
		u = pred->lookup(PC & 0x0fffffff, false, false);
		return u->prediction();
	}

	void UpdatePredictor(UINT32 PC, bool resolveDir, bool predDir, UINT32 branchTarget) {
		pred->update (u, resolveDir);
	}

	void TrackOtherInst(UINT32 PC, OpType opType, UINT32 branchTarget) {
		switch (opType) {
			case OPTYPE_CALL_DIRECT:
			case OPTYPE_RET:
			case OPTYPE_BRANCH_UNCOND:
			case OPTYPE_INDIRECT_BR_CALL:
			pred->info (PC & 0x0fffffff, opType); break;
			default: ;
		}
	}

  // Contestants can define their own functions below

};


/***********************************************************/
#endif

