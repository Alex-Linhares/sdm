import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time



HARD_LOCATIONS = 2**20
DIMENSIONS = 256
BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS = 1300  #Compute analytically; prove it's safe... 
maximum = (2**32)-1


HASH_TABLE_SIZE =  46273 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 46273 \n\
#define HASH_TABLE_SIZE2 46272 \n\
#define HASH_TABLE_SIZE3 46271 \n\
#define HASH_TABLE_SIZE4 46270 \n\
#define HASH_TABLE_SIZE5 46269 \n\
#define HASH_TABLE_SIZE6 46268 \n\
#define HASH_TABLE_SIZE7 46267 \n\
"

HASH_TABLE_SIZE =  75707 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 75707 \n\
#define HASH_TABLE_SIZE2 75706 \n\
#define HASH_TABLE_SIZE3 75705 \n\
#define HASH_TABLE_SIZE4 75704 \n\
#define HASH_TABLE_SIZE5 75703 \n\
#define HASH_TABLE_SIZE6 75702 \n\
#define HASH_TABLE_SIZE7 75701 \n\
"

HASH_TABLE_SIZE =  149911 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 149911 \n\
#define HASH_TABLE_SIZE2 149910 \n\
#define HASH_TABLE_SIZE3 149909 \n\
#define HASH_TABLE_SIZE4 149908 \n\
#define HASH_TABLE_SIZE5 149907 \n\
#define HASH_TABLE_SIZE6 149906 \n\
#define HASH_TABLE_SIZE7 149905 \n\
"

HASH_TABLE_SIZE =  65449 
HASH_TABLE_SIZE_FILE = \
"#define HASH_TABLE_SIZE 65449 \n\
#define HASH_TABLE_SIZE2 65448 \n\
#define HASH_TABLE_SIZE3 65447 \n\
#define HASH_TABLE_SIZE4 65446 \n\
#define HASH_TABLE_SIZE5 65445 \n\
#define HASH_TABLE_SIZE6 65444 \n\
#define HASH_TABLE_SIZE7 65443 \n\
"

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

'''
def random_int64(size):
    a0 = numpy.random.random_integers(0, 0xFFFF, size=size).astype(numpy.uint64)
    a1 = numpy.random.random_integers(0, 0xFFFF, size=size).astype(numpy.uint64)
    a2 = numpy.random.random_integers(0, 0xFFFF, size=size).astype(numpy.uint64)
    a3 = numpy.random.random_integers(0, 0xFFFF, size=size).astype(numpy.uint64)
    a = a0 + (a1<<16) + (a2 << 32) + (a3 << 48)
    return a.view(dtype=numpy.uint64)
'''


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
    bitstring = numpy.random.random_integers(0,2**32-1,size=8).astype(numpy.uint32) #TRYING THIS OUT
    #import address_space_through_sha256_SDM
    #import random
    #bitstring = address_space_through_sha256_SDM.get_bitstring(str(random.randrange(2**32-1)))
    return bitstring


def Get_Bitstring_GPU_Buffer(ctx, bitstring):
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
    memory_addresses = numpy.random.random_integers(0,maximum,size=(HARD_LOCATIONS,8)).astype(numpy.uint32) #numpy.random.random_integers(0,(2**32)-1,size=(HARD_LOCATIONS,8)).astype(numpy.uint32)
    return memory_addresses

    #The mistake here is that numpy is working with ints, and we're working with uints, so we have to have the **int** range, then cast to uint---not the uint range.  Yes, we are morons.  
    #memory_addresses = numpy.random.random_integers(-2**15+1,2**15-1,size=(HARD_LOCATIONS,8)).astype(numpy.uint32)

def load_address_space():
    import cPickle
    address_space_file = open('hard_locations.sha256.sdm.pickle', 'rb')
    space = cPickle.load(address_space_file)
    return space

def Get_Text_code(filename):
    with open (filename, "r") as myfile:
        data = myfile.read()
        return HASH_TABLE_SIZE_FILE + data 

def create_sdm_values():
    return numpy.zeros((HARD_LOCATIONS, DIMENSIONS), dtype = numpy.int8) 


def write_x_at_x_kanerva(active_hard_locations, bitstring):
    maximum=255 #isn't it 255?
    for pos in numpy.nditer(active_hard_locations):
        bitstring = SDM_addresses[pos,0:8] #WHAT THE FUCK IS THIS? JUST RECEIVED AS PARAMETER!
        
        for dimension in range (256):
            uint_to_check_bit = ((dimension // maximum) + dimension % maximum ) // 32
            #print dimension
            add = bool((bitstring [uint_to_check_bit]  &  (1<< dimension%32  ) ))
            #print add
            if add: 
                sdm_values[pos,dimension] +=1
            elif not(add): 
                sdm_values[pos,dimension] -=1
        print 'location', pos,'has been updated to',sdm_values[pos,]


'''
def read_address_sum_kanerva(active_hard_locations):
    #maximum=255
    sum = numpy.zeros((DIMENSIONS,), dtype = int32)
    for pos in active_hard_locations:
        bitstring = SDM_addresses[pos,]
        for checkbit in range (256):
            if bool(bitstring [(  (checkbit // maximum) + checkbit % maximum ) // 32]  &  (1<< checkbit%32  )): 
                #increase something
                sum[pos,checkbit]+=1
            else:
                #decrease something
                sum[pos,checkbit]-=1
    return sum
'''




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
    
    final_gpu, evt = my_pyopencl_algorithm.sparse_copy_if(distances_gpu, "ary[i] < 104", queue = queue)

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
    hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.uint32)
    #prg.clear_hash_table_gpu(queue, hash_table_gpu.data, distances_gpu.data)

    prg.get_active_hard_locations_32bit (queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu.data, hash_table_gpu.data ).wait()

    
    active_hard_locations_gpu, event = my_pyopencl_algorithm.sparse_copy_if(hash_table_gpu, "ary[i] > 0", queue = queue)  
    #active_hard_locations_gpu, final_distances_gpu, event = my_pyopencl_algorithm.sparse_copy_if_with_distances(hash_table_gpu, "ary[i] > 0", extra_args = [distances_gpu], queue = queue)  

    active = active_hard_locations_gpu.get()
    count = active.size

    final_distances_gpu = cl_array.Array(queue, (count,), dtype=numpy.uint32)

    prg.get_HL_distances_from_gpu(queue, (count,), None, active_hard_locations_gpu.data, distances_gpu.data, final_distances_gpu.data)


    return count, active_hard_locations_gpu.get(), final_distances_gpu.get()



def Get_Active_Locations5_2_dist_computations( ctx):
    hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.uint32)
    #prg.clear_hash_table_gpu(queue, hash_table_gpu.data, distances_gpu.data)


    prg.get_active_hard_locations_32bit_no_if_2_dist_computes (queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, hash_table_gpu.data ).wait()    
    
    active_hard_locations_gpu, event = my_pyopencl_algorithm.sparse_copy_if(hash_table_gpu, "ary[i] > 0", queue = queue)  
    #active_hard_locations_gpu, final_distances_gpu, event = my_pyopencl_algorithm.sparse_copy_if_with_distances(hash_table_gpu, "ary[i] > 0", extra_args = [distances_gpu], queue = queue)  

    active = active_hard_locations_gpu.get()
    count = active.size

    final_distances_gpu = cl_array.Array(queue, (count,), dtype=numpy.uint32)

    prg.get_HL_distances_from_gpu_2nd_compute(queue, (count,), None, bitstring_gpu, memory_addresses_gpu.data, active_hard_locations_gpu.data, final_distances_gpu.data)

    return count, active_hard_locations_gpu.get(), final_distances_gpu.get()



criteria = 104

OpenCL_code = Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
import pyopencl.array as cl_array

#final_locations_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)
#final_distances_gpu = cl_array.zeros(queue, (BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS,), dtype=numpy.int32)


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#SDM_addresses = Create_Memory_Addresses()
SDM_addresses = load_address_space()
print SDM_addresses[0,0]



print 'sending memory_addresses from host to compute device...'
memory_addresses_gpu = cl_array.to_device(queue, SDM_addresses) 


'''
print 'creating SDM values'
start = time.time()
sdm_values = numpy.zeros((HARD_LOCATIONS, 256), dtype = numpy.int8)
cl_array.to_device(queue, sdm_values) 
print (time.time()-start), 'Seconds'
'''

distances_host = Get_Hamming_Distances

distances_gpu = Get_Distances_GPU_Buffer(ctx)


prg = cl.Program(ctx, OpenCL_code).build()

hash_table_active_index = Get_Hash_Table()
hamming_distances = Get_Hamming_Distances()


print "\n"


num_times = 20000
Results_and_Statistics = numpy.zeros(num_times+1).astype(numpy.uint32) 
usual_result = 2238155  # for 2000 runs of 2^20 hard locations

'''
using copy_if_2() yields 2238159!!!  4 more!!
''' 


for platform in cl.get_platforms():
    for device in platform.get_devices():
        print("===============================================================")
        print("Platform name:", platform.name)
        print("Platform profile:", platform.profile)
        print("Platform vendor:", platform.vendor)
        print("Platform version:", platform.version)
        print("---------------------------------------------------------------")
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024//1024, 'GB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)

        Platform_name = platform.name
        Device_name = device.name + ' on platform ' + platform.name
        print Device_name
 




print '\n\n===================================================='

distances_gpu = pyopencl.array.arange(queue, HARD_LOCATIONS, dtype=numpy.int32)

hash_table_gpu = cl_array.zeros(queue, (HASH_TABLE_SIZE,), dtype=numpy.int32)

start = time.time()
for x in range(num_times):
    
    bitstring = Get_Random_Bitstring()
    bitstring_gpu = Get_Bitstring_GPU_Buffer(ctx, bitstring)  #Optimize THIS!
    
    count, active_hard_locations, distances = Get_Active_Locations5 (ctx) 
    #count, active_hard_locations, distances = Get_Active_Locations5_2_dist_computations (ctx) 
    #active_hard_locations = Get_Active_Locations2(ctx)
    
    #write_x_at_x_kanerva(active_hard_locations,bitstring)
    
    #distances = Get_Active_Locations2(ctx) 

    Results_and_Statistics[x] = active_hard_locations.size
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

'''
bitstring = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
maximum = (2**32)-1
print maximum
for checkbit in range (256):
    yes2 = bool(numpy.bitwise_and((2**checkbit)%maximum, bitstring[  (  (checkbit // maximum) + checkbit % maximum ) // 32  ] ) )

    #bool(n&(1<<b))
    
    #n = bitstring [(  (checkbit // maximum) + checkbit % maximum ) // 32]
    #bit = checkbit%32
    yes1 = bool((bitstring [(  (checkbit // maximum) + checkbit % maximum ) // 32]&(1<< checkbit%32  ) ))
    #yes2 = bool(bin(numpy.bitwise_and((2**checkbit), bin(bitstring) )))
    print 'bit', checkbit, 'set to', yes1, yes2, 'in uint32 #', (  (checkbit // maximum) + checkbit % maximum ) // 32

'''

''' 
Profiling... first time.  20,000 GPU run on W9000-liquid-cool

python -m cProfile -s cumtime SDM_OpenCL_get_active_locations.py 

Looks like sparse_copy_if is taking too long... 
also, what is array.py get() being called from?  sparse_copy_if?  

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    1.183    1.183   38.805   38.805 SDM_OpenCL_get_active_locations.py:1(<module>)
    20000    6.638    0.000   35.206    0.002 SDM_OpenCL_get_active_locations.py:272(Get_Active_Locations5)
    20000    0.567    0.000   18.141    0.001 my_pyopencl_algorithm.py:113(sparse_copy_if)
    80000    0.502    0.000   11.965    0.000 array.py:671(get)
    80001    9.517    0.000    9.752    0.000 __init__.py:930(enqueue_copy)
    20007    0.216    0.000    6.956    0.000 __init__.py:178(build)
    20007    0.012    0.000    6.607    0.000 __init__.py:219(_build_and_catch_errors)
    20007    0.017    0.000    6.595    0.000 __init__.py:210(<lambda>)
    20007    0.105    0.000    6.577    0.000 cache.py:469(create_built_program_from_source_cached)
    20007    0.831    0.000    5.438    0.000 cache.py:320(_create_built_program_from_source_cached)
    20007    0.617    0.000    3.378    0.000 cache.py:247(retrieve_from_cache)
   140003    2.136    0.000    2.932    0.000 array.py:447(__init__)
   140002    1.393    0.000    2.771    0.000 __init__.py:499(kernel_call)
    20000    0.564    0.000    2.609    0.000 scan.py:1292(__call__)
    20001    0.066    0.000    1.997    0.000 array.py:1686(zeros)
        1    0.005    0.005    1.709    1.709 __init__.py:767(create_some_context)
        2    0.000    0.000    1.704    0.852 __init__.py:794(get_input)
        2    1.704    0.852    1.704    0.852 {raw_input}
    20001    0.039    0.000    1.386    0.000 array.py:1043(fill)
   140002    1.111    0.000    1.339    0.000 __init__.py:530(kernel_set_args)
    20002    0.203    0.000    1.321    0.000 array.py:197(kernel_runner)
    80001    0.646    0.000    1.046    0.000 stride_tricks.py:22(as_strided)
    20007    0.831    0.000    1.034    0.000 __init__.py:415(program_build)
   820157    0.962    0.000    0.962    0.000 __init__.py:317(result)
    60006    0.152    0.000    0.862    0.000 __init__.py:164(__getattr__)
    40000    0.092    0.000    0.754    0.000 array.py:590(_new_with_changes)
   120012    0.320    0.000    0.745    0.000 __init__.py:460(wrapper)
    20007    0.057    0.000    0.722    0.000 cache.py:80(__init__)
    60006    0.534    0.000    0.649    0.000 __init__.py:492(kernel_init)
    20007    0.610    0.000    0.610    0.000 {posix.open}
    20008    0.488    0.000    0.606    0.000 {cPickle.load}
''' 