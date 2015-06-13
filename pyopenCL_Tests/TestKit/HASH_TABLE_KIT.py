'''
This utility:

a) generates a HASH_TABLE string for C files, using (PRIME, NUM_HASHES) as the input, 
     whereas PRIME is the initial (largest, twin) prime determining the size of the HASH_TABLE,
     and NUM_HASHES is the number of times the system must be hashed.

b) it tests how many collisions occur in the set [0, 2^20 -1] for this HASH_TABLE, in python,
   using the logic of the opencl code.


    hash_index = (mem_pos % HASH_TABLE_SIZE);          // Hashing 7 times, see (cormen et al) "introduction to algorihms" section 12.4, on "open addressing". Performance doesn't degrade in the macbook pro.
    if (hash_table_gpu[hash_index]>0) {              // 7 reaches diminishing returns; in parallel the system can read 0s simultaneously, and may have a collision..
      hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE2)) % HASH_TABLE_SIZE;
      if (hash_table_gpu[hash_index]>0) {
        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE3)) % HASH_TABLE_SIZE;
        if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE; 
            if (hash_table_gpu[hash_index]>0) {
              hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE; 
              }
            }
          }
        }
      }  
    }

''' 




import numpy
import random

def test_collisions (PRIME, NUM_HASHES):
    HASH_TABLE_SIZE = PRIME
    HASH_TABLE_SIZE2 = PRIME -1
    HASH_TABLE_SIZE3 = PRIME -2
    HASH_TABLE_SIZE4 = PRIME -3
    HASH_TABLE_SIZE5 = PRIME -4
    HASH_TABLE_SIZE6 = PRIME -5
    HASH_TABLE_SIZE7 = PRIME -6
    hash_table_gpu = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    collisions = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    count = 1

    #print hash_table_gpu
    #print numpy.sum(hash_table_gpu)
    for mem_pos in range (0,2**20):
    	if (random.randint(0, 1000)==253):  #mem_pos % 1000 == 523):
	        hash_index = (mem_pos % HASH_TABLE_SIZE)
	        #print count, mem_pos, hash_index
	        count +=1
	         
	        if (hash_table_gpu[hash_index]>0):  
	            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE2)) % HASH_TABLE_SIZE
	        
	            if (hash_table_gpu[hash_index]>0):
	                hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE3)) % HASH_TABLE_SIZE
	        '''
	                if (hash_table_gpu[hash_index]>0):
	                    hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE 
	                    if (hash_table_gpu[hash_index]>0):
	                        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE 
	                        if (hash_table_gpu[hash_index]>0):
	                            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE
	                            if (hash_table_gpu[hash_index]>0):
	                                hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE
	        ''' 
	        if (hash_table_gpu[hash_index]>0):
	        	print 'colision of', mem_pos, 'in hash_index', hash_index
	        	collisions[hash_index]+=1
	        hash_table_gpu [hash_index] = mem_pos

	numpy.set_printoptions(threshold= 400) #numpy.nan)
    #print hash_table_gpu
    #print 'hash_table_gpu sum', numpy.sum(hash_table_gpu)
    numpy.set_printoptions(threshold= numpy.nan)
    #print collisions
    #print 'collisions sum', numpy.max(collisions)
    return numpy.max(collisions)
       


# test_collisions results ---
#100 runs of p=75707, 3-hashed through this weird sum, yields 0 runs with a single collision, 131 seconds
#100 runs of p=46273, 3-hashed through this weird sum, yields 0 runs with a single collision, 119 seconds
#100 runs of p=36931, 3-hashed through this weird sum, yields 1 runs with a single collision, 131 seconds


def test_collisions_shiftleft (PRIME, NUM_HASHES):
    HASH_TABLE_SIZE = PRIME
    HASH_TABLE_SIZE2 = PRIME -1
    HASH_TABLE_SIZE3 = PRIME -2
    HASH_TABLE_SIZE4 = PRIME -3
    HASH_TABLE_SIZE5 = PRIME -4
    HASH_TABLE_SIZE6 = PRIME -5
    HASH_TABLE_SIZE7 = PRIME -6
    hash_table_gpu = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    collisions = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    count = 1

    #print hash_table_gpu
    #print numpy.sum(hash_table_gpu)
    for mem_pos in range (0,2**20):
    	if (random.randint(0, 1000)==253):  #mem_pos % 1000 == 523):
	        hash_index = (mem_pos % HASH_TABLE_SIZE)
	        #print count, mem_pos, hash_index
	        count +=1
	         
	        if (hash_table_gpu[hash_index]>0):  
	            hash_index= ( (hash_index << 3) +1) % HASH_TABLE_SIZE
	        
	            if (hash_table_gpu[hash_index]>0):
	                hash_index= ((hash_index << 3) +7) % HASH_TABLE_SIZE
	        '''
	                if (hash_table_gpu[hash_index]>0):
	                    hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE 
	                    if (hash_table_gpu[hash_index]>0):
	                        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE 
	                        if (hash_table_gpu[hash_index]>0):
	                            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE
	                            if (hash_table_gpu[hash_index]>0):
	                                hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE
	        ''' 
	        if (hash_table_gpu[hash_index]>0):
	        	print 'colision of', mem_pos, 'in hash_index', hash_index
	        	collisions[hash_index]+=1
	        hash_table_gpu [hash_index] = mem_pos

	numpy.set_printoptions(threshold= 400) #numpy.nan)
    #print hash_table_gpu
    #print 'hash_table_gpu sum', numpy.sum(hash_table_gpu)
    numpy.set_printoptions(threshold= numpy.nan)
    #print collisions
    #print 'collisions sum', numpy.max(collisions)
    return numpy.max(collisions)


# test_collisions results ---
# 1 shift left
#100 runs of p=36931, 2-hashed, yields 42 collisions, 120 seconds
#100 runs of p=36931, 3-hashed, yields 4 collisions, 131 seconds
# 2 shift lefts
#100 runs of p=36931, 2-hashed, yields 36 collisions, 120 seconds
#100 runs of p=36931, 3-hashed, yields 1 collision, 131 seconds
# 4 shift lefts
#100 runs of p=36931, 2-hashed, yields 46 collisions, 131 seconds
# 3 shift lefts, then add 1
#100 runs of p=36931, 2-hashed, yields 25 collisions, 131 seconds
#100 runs of p=36931, 2-hashed, yields 36 collisions, 132 seconds
# 2 shift lefts, then add 1
#100 runs of p=36931, 2-hashed, yields 46 collisions, 131 seconds
# 3 shift left +1, then 3 shift left +7
#100 runs of p=36931, 3-hashed, yields 0 collisions, 132.4 seconds



def test_collisions_no_if_with_xor (PRIME, NUM_HASHES):
    HASH_TABLE_SIZE = PRIME
    HASH_TABLE_SIZE2 = PRIME -1
    HASH_TABLE_SIZE3 = PRIME -2
    HASH_TABLE_SIZE4 = PRIME -3
    HASH_TABLE_SIZE5 = PRIME -4
    HASH_TABLE_SIZE6 = PRIME -5
    HASH_TABLE_SIZE7 = PRIME -6
    hash_table_gpu = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    collisions = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    count = 1

    #print hash_table_gpu
    #print numpy.sum(hash_table_gpu)
    for mem_pos in range (0,2**20):
    	if (random.randint(0, 1000)==253):  #mem_pos % 1000 == 523):
	        hash_index = ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE

	        if (hash_table_gpu[hash_index]!=0):
	        	print 'colision of', mem_pos, 'in hash_index', hash_index, 'mem_pos', mem_pos, 'value', hash_table_gpu[hash_index], 'xor', mem_pos ^ hash_table_gpu[mem_pos %HASH_TABLE_SIZE], 'alt-pos', ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE
	        	collisions[hash_index]+=1

	        hash_table_gpu[ hash_index ] = mem_pos # note: instead of mod 2**N, using faster & (2**N-1) here
	        

	numpy.set_printoptions(threshold= 400) #numpy.nan)
    #print hash_table_gpu
    #print 'hash_table_gpu sum', numpy.sum(hash_table_gpu)
    numpy.set_printoptions(threshold= numpy.nan)
    #print collisions
    #print 'collisions sum', numpy.max(collisions)
    return numpy.max(collisions)



def test_collisions_no_if (PRIME, NUM_HASHES):
    HASH_TABLE_SIZE = PRIME
    HASH_TABLE_SIZE2 = PRIME -1
    HASH_TABLE_SIZE3 = PRIME -2
    HASH_TABLE_SIZE4 = PRIME -3
    HASH_TABLE_SIZE5 = PRIME -4
    HASH_TABLE_SIZE6 = PRIME -5
    HASH_TABLE_SIZE7 = PRIME -6
    hash_table_gpu = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    collisions = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.uint32)
    count = 1

    #print hash_table_gpu
    #print numpy.sum(hash_table_gpu)
    for mem_pos in range (0,2**20):
    	if (random.randint(0, 1000)==253):  #mem_pos % 1000 == 523):
	        hash_index = ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE
	        #hash_index = ( (not hash_table_gpu[(mem_pos)%HASH_TABLE_SIZE] * | hash_table_gpu[ (not mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE 

	        if (hash_table_gpu[hash_index]!=0):
	        	print 'colision of', mem_pos, 'in hash_index', hash_index, 'mem_pos', mem_pos, 'value', hash_table_gpu[hash_index], 'xor', mem_pos ^ hash_table_gpu[mem_pos %HASH_TABLE_SIZE], 'alt-pos', ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE
	        	collisions[hash_index]+=1

	        hash_table_gpu[ hash_index ] = mem_pos # note: instead of mod 2**N, using faster & (2**N-1) here
	        

	numpy.set_printoptions(threshold= 400) #numpy.nan)
    #print hash_table_gpu
    #print 'hash_table_gpu sum', numpy.sum(hash_table_gpu)
    numpy.set_printoptions(threshold= numpy.nan)
    #print collisions
    #print 'collisions sum', numpy.max(collisions)
    return numpy.max(collisions)

# test_collisions results ---


p = 149911 #46273 #75707 #65535 #36931 #75707 #55933 #46273 #36931 #25033
iterations = 7

import time

numtimes = 100
start = time.time()
collisions_max = numpy.zeros(numtimes).astype(numpy.uint32)
for t in range (numtimes):
    collisions_max [t]= test_collisions_no_if_with_xor (p, iterations)  
exec_time = time.time() - start
print exec_time, 'seconds for' , numtimes, 'runs' 
print collisions_max



import matplotlib.pyplot as plt
plt.hist(collisions_max, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], range = (0, numpy.max(collisions_max)), histtype='stepfilled') #bins = numpy.max(collisions_max)+1,
plt.show()
plt.savefig('hist.svg')