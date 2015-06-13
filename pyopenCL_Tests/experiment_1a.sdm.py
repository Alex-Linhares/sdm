import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time


HARD_LOCATIONS = 2**20
DIMENSIONS = 256
#BUFFER_SIZE_EXPECTED_ACTIVE_HARD_LOCATIONS = 1300  #Compute analytically; prove it's safe...
maximum = (2**32)-1
HASH_TABLE_SIZE =  25033
print "HASH_TABLE_SIZE=", HASH_TABLE_SIZE  #WHAT IS THE OPTIMUM HASH_TABLE_SIZE??

ACCESS_RADIUS_THRESHOLD = 104 #COMPUTE EXPECTED NUMBER OF num_ACTIVE_locations_found HARD LOCATIONS


import scan_toolkit as ocl_scan

import os
import pyopencl.array as cl_array

SDM_addresses = ocl_scan.load_address_space()
print SDM_addresses[0,0]

numpy.random.seed(seed=12345678)


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mem_flags = cl.mem_flags

OpenCL_code = ocl_scan.Get_Text_code ('GPU_Code_OpenCLv1_2.cl')

import os
import pyopencl.array as cl_array

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#SDM_addresses = Create_Memory_Addresses()



print 'sending memory_addresses from host to compute device...'
memory_addresses_gpu = cl_array.to_device(queue, SDM_addresses)

distances_host = ocl_scan.Get_Hamming_Distances()

distances_gpu = ocl_scan.Get_Distances_GPU_Buffer(ctx)


prg = cl.Program(ctx, OpenCL_code).build()




num_times = 200

start = time.time()
for x in range(num_times):
    
    bitstring = ocl_scan.Get_Random_Bitstring()
    bitstring_gpu = ocl_scan.Get_Bitstring_GPU_Buffer(ctx, bitstring)  #Optimize THIS!
    
    count, active_hard_locations, distances = ocl_scan.Get_Active_Locations5(ctx, bitstring_gpu)
    #active_hard_locations = Get_Active_Locations2(ctx)
    
    #write_x_at_x_kanerva(active_hard_locations,bitstring)
    
    #distances = Get_Active_Locations2(ctx)
    
    Results_and_Statistics[x] = active_hard_locations.size

time_elapsed = (time.time()-start)


print Results_and_Statistics[Results_and_Statistics !=0].min(), " the minimum of HLs found should be 1001"
print Results_and_Statistics[Results_and_Statistics !=0].mean(), "the mean of HLs found should be 1119.077"
print Results_and_Statistics[Results_and_Statistics !=0].max(), "the max of HLs found should be 1249"

print '\n Seconds to Scan 2^20 Hard Locations', num_times,'times:', time_elapsed

print distances

print active_hard_locations
print active_hard_locations.size

sum = numpy.sum(Results_and_Statistics)
