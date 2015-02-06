import pyopencl as cl
import numpy as np
import my_pyopencl_algorithm
import time

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

from pyopencl.clrandom import rand as clrand

random  = np.arange(1, 2**20, dtype = np.int32)
print '\n\n==============\nDATASET\nsorted', random, '\n==========\n'

random_gpu = cl.array.to_device(queue, random) 

start = time.time()
final_gpu, count_gpu, evt = my_pyopencl_algorithm.copy_if(random_gpu, "ary[i] < 200000", queue = queue)
final = final_gpu.get()
count = int(count_gpu.get())
#count = final.size
print '\ncopy_if():\nresults=',final[:count], '\nfound=', count, '\ntime=', (time.time()-start), '\n========\n'


start = time.time()
final_gpu, evt = my_pyopencl_algorithm.sparse_copy_if(random_gpu, "ary[i] < 200000", queue = queue)
final = final_gpu.get()
#count = int(count_gpu.get())
count = final.size
print '\nsparse_copy_if():\nresults=',final, '\nfound=', count, '\ntime=', (time.time()-start)





random  = np.random.permutation(random)
print '\n\n==============\nDATASET\npermutated', random, '\n==========\n'

random_gpu = cl.array.to_device(queue, random) 

start = time.time()
final_gpu, count_gpu, evt = my_pyopencl_algorithm.copy_if(random_gpu, "ary[i] < 200000", queue = queue)
final = final_gpu.get()
count = int(count_gpu.get())
#count = final.size
print '\ncopy_if() on permuted array:\nresults=',final[:count], '\nfound=', count, '\ntime=', (time.time()-start), '\n========\n'


start = time.time()
final_gpu, evt = my_pyopencl_algorithm.sparse_copy_if(random_gpu, "ary[i] < 200000", queue = queue)
final = final_gpu.get()
#count = int(count_gpu.get())
count = final.size
print '\nsparse_copy_if() on permuted array:\nresults=',final, '\nfound=', count, '\ntime=', (time.time()-start)


