import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
DIMENSIONS = 256
BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS = 1300  #Compute analytically; prove it's safe... 

HASH_TABLE_SIZE =  25033 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 25033 \n\
#define HASH_TABLE_SIZE2 25032 \n\
#define HASH_TABLE_SIZE3 25031 \n\
#define HASH_TABLE_SIZE4 25030 \n\
#define HASH_TABLE_SIZE5 25029 \n\
#define HASH_TABLE_SIZE6 25028 \n\
#define HASH_TABLE_SIZE7 25027 \n\
"


# HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but way less collisions.  It should also be "far" from a power of 2.


print "HASH_TABLE_SIZE=", HASH_TABLE_SIZE  #WHAT IS THE OPTIMUM HASH_TABLE_SIZE??

ACCESS_RADIUS_THRESHOLD = 104 #COMPUTE EXPECTED NUMBER OF num_ACTIVE_locations_found HARD LOCATIONS



numpy.random.seed(seed=12345678)  

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mem_flags = cl.mem_flags


def Get_Hash_Table():
    hash_table_active_index = numpy.zeros(HASH_TABLE_SIZE).astype(numpy.int32) 
    return hash_table_active_index

def Get_Hash_Table_GPU_Buffer(ctx):
    hash_table_active_index = Get_Hash_Table()
    hash_table_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=hash_table_active_index)
    return hash_table_gpu

def Get_Hamming_Distances():
    hamming_distances = numpy.zeros(HARD_LOCATIONS).astype(numpy.uint32) #32 BITS??????
    return hamming_distances

def Get_Distances_GPU_Buffer(ctx):
    Distances = Get_Hamming_Distances() 
    hamming_distances_gpu = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=Distances) 
    return hamming_distances_gpu

def Get_Random_Bitstring():
    bitstrings = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
    return bitstrings


def Get_Bitstring_GPU_Buffer(ctx):
    bitstring = Get_Random_Bitstring()
    bitstring_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstring)
    return bitstring_gpu


'''
def Get_num_times_Random_Bitstrings():
    for x in range(num_times):
        bitstrings[x] = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
    return bitstrings
'''

'''
def Get_num_times_Bitstrings_GPU_Buffer(ctx):
    for x in range(num_times):
        bitstrings = numpy.random.random_integers(0,2**32,size=8*num_times).astype(numpy.uint32)
        #bitstrings.shape = (8,num_times)
        bitstring_gpu = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstrings)
    return bitstring_gpu
'''


def Create_Memory_Addresses():
    print 'creating memory memory_addresses...'
    memory_addresses = numpy.random.random_integers(0,2**32,size=(HARD_LOCATIONS,8)).astype(numpy.uint32)
    return memory_addresses


def Get_Text_code(filename):
    with open (filename, "r") as myfile:
        data = myfile.read()
        return HASH_TABLE_SIZE_FILE + data 

def create_sdm_values():
    return numpy.zeros((HASH_TABLE_SIZE, DIMENSIONS), dtype = numpy.int8)


'''

def write_x_at_x(active_hard_locations , bitstring):
    for pos in active_hard_locations:
'''


def read_address_sum(active_hard_locations):
    sum = numpy.zeros((DIMENSIONS,), dtype = int32)
    for pos in active_hard_locations:
        bitstring = SDM_addresses[pos,]
        for checkbit in range (256):
            if bool(numpy.bitwise_and((2**checkbit)%maximum, bitstring[  (  (checkbit // maximum) + checkbit % maximum ) // 32  ] ) ): 
                #increase something
                sum[pos,checkbit]+=1
            else:
                #decrease something
                sum[pos,checkbit]-=1
    return sum







def Get_Active_Locations(bitstring, ctx):
    err = prg.clear_hash_table_gpu(queue, (HASH_TABLE_SIZE,), None, hash_table_gpu.data).wait()
 
    err = prg.get_active_hard_locations_no_dist_buffer(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu, bitstring_gpu, hash_table_gpu.data).wait()  
    if err: print 'Error --> ',err
    err = cl.enqueue_read_buffer(queue, hash_table_gpu.data , hash_table_active_index).wait()
    if err: print 'Error in retrieving hash_table_active_index? --> ',err
    active_hard_locations = hash_table_active_index[hash_table_active_index!=0]  ## THIS HAS TO BE DONE ON THE GPU, MORON!  If it's sorted, you can get the num of items and copy only those...
    '''
    #ALSO, when GID=0, you are counting the hard locations there.  HL[0] IS ALWAYS ACTIVE.
    '''

    return active_hard_locations



#from pyopencl.algorithm import copy_if  
import my_pyopencl_algorithm
from my_pyopencl_algorithm import copy_if  

def Get_Active_Locations2(ctx):
    prg.compute_distances(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu.data).wait()  
    
    final_gpu, evt = my_pyopencl_algorithm.copy_if_2(distances_gpu, "ary[i] < 104", queue = queue)

    return final_gpu




def Get_Active_Locations3(ctx):
    hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
    prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu.data, hash_table_gpu.data ).wait()
    #if err: print 'Error --> ',err

    active_hard_locations_gpu, event = my_pyopencl_algorithm.sparse_copy_if(hash_table_gpu, "ary[i] > 0", queue = queue)  

    active = active_hard_locations_gpu.get()
    Num_HLs = active.size

    prg.get_HL_distances_from_gpu5(queue, (Num_HLs,), None, active_hard_locations_gpu.data, distances_gpu.data, final_distances_gpu.data)

    return Num_HLs, final_locations_gpu.get(), final_distances_gpu.get() 




def Get_Active_Locations4(bitstring, ctx):
    hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
    prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu, hash_table_gpu.data ).wait()

    active_hard_locations_gpu, count_active_gpu, event = copy_if(distances_gpu, "ary[i] < 104")  

    Num_HLs = int(count_active_gpu.get())

    final_distances_gpu = cl_array.zeros(queue, (Num_HLs,), dtype=numpy.int32)

    prg.get_HL_distances_from_gpu(queue, (Num_HLs,), None, active_hard_locations_gpu.data, hash_table_gpu.data, distances_gpu)

    prg.copy_final_results(queue, (Num_HLs,), None, final_locations_gpu.data, active_hard_locations_gpu.data, final_distances_gpu.data, hash_table_gpu.data)
    # err = 
    return Num_HLs, final_locations_gpu.get(), final_distances_gpu.get()





def Get_Active_Locations5( ctx):
    hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)
 
    prg.get_active_hard_locations_32bit(queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu.data, hash_table_gpu.data ).wait()
    
    active_hard_locations_gpu, event = my_pyopencl_algorithm.sparse_copy_if(hash_table_gpu, "ary[i] > 0", queue = queue)  
    #active_hard_locations_gpu, final_distances_gpu, event = my_pyopencl_algorithm.sparse_copy_if_with_distances(hash_table_gpu, "ary[i] > 0", extra_args = [distances_gpu], queue = queue)  

    active = active_hard_locations_gpu.get()
    count = active.size

    final_distances_gpu = cl_array.Array(queue, (count,), dtype=numpy.int32)

    prg.get_HL_distances_from_gpu(queue, (count,), None, active_hard_locations_gpu.data, distances_gpu.data, final_distances_gpu.data)

    return count, active_hard_locations_gpu.get(), final_distances_gpu.get()



criteria = 104

OpenCL_code = Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
import pyopencl.array as cl_array

#final_locations_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)
#final_distances_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

SDM_addresses = Create_Memory_Addresses()

print 'sending memory_addresses from host to compute device...'
memory_addresses_gpu = cl_array.to_device(queue, SDM_addresses) 


distances_host = Get_Hamming_Distances

distances_gpu = Get_Distances_GPU_Buffer(ctx)


prg = cl.Program(ctx, OpenCL_code).build()

hash_table_active_index = Get_Hash_Table()
hamming_distances = Get_Hamming_Distances()


print "\n"


num_times = 2000
Results_and_Statistics = numpy.zeros(num_times+1).astype(numpy.uint32) 
usual_result = 2238155  # for 2000 runs of 2^20 hard locations

'''
using copy_if_2() yields 2238159!!!
''' 


print '\n\n===================================================='

distances_gpu = pyopencl.array.arange(queue, HARD_LOCATIONS, dtype=numpy.int32)

hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)

start = time.time()
for x in range(num_times):
    
    bitstring_gpu = Get_Bitstring_GPU_Buffer(ctx)  #Optimize THIS!
    
    count, active_hard_locations, distances = Get_Active_Locations5(ctx) 
    
    #distances = Get_Active_Locations2(ctx) 

    Results_and_Statistics[x] = count
    #Results_and_Statistics[x] = distances.size
    
time_elapsed = (time.time()-start)


print Results_and_Statistics[Results_and_Statistics !=0].min(), " the minimum of HLs found should be 1001"
print Results_and_Statistics[Results_and_Statistics !=0].mean(), "the mean of HLs found should be 1119.077"
print Results_and_Statistics[Results_and_Statistics !=0].max(), "the max of HLs found should be 1249"

print '\n Seconds to Scan 2^20 Hard Locations', num_times,'times:', time_elapsed

print distances

print active_hard_locations
print active_hard_locations.size

sum = numpy.sum(Results_and_Statistics)

print 'creating SDM values'
start = time.time()
sdm_values = numpy.zeros((HASH_TABLE_SIZE, DIMENSIONS), dtype = numpy.int8)  #.astype(numpy.int8) 
print (time.time()-start), 'Seconds'


bitstring = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
maximum = (2**32)-1
print maximum
for checkbit in range (256):
    yes2 = bool(numpy.bitwise_and((2**checkbit)%maximum, bitstring[  (  (checkbit // maximum) + checkbit % maximum ) // 32  ] ) )

    #bool(n&(1<<b))
    
    n = bitstring [(  (checkbit // maximum) + checkbit % maximum ) // 32]
    bit = checkbit%32
    yes1 = bool((n&(1<< checkbit%32  ) ))
    #yes2 = bool(bin(numpy.bitwise_and((2**checkbit), bin(bitstring) )))
    print 'bit', checkbit, 'set to', yes1, yes2, 'in uint32 #', (  (checkbit // maximum) + checkbit % maximum ) // 32



