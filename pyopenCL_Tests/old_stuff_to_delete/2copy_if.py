import pyopencl as cl
import numpy as np
from pyopencl import array
from pyopencl.algorithm import copy_if
from pyopencl.algorithm import algorithm

'''
context = cl.create_some_context()
queue = cl.CommandQueue(context)
mf = cl.mem_flags

rand = cl_array.to_device (ctx, queue, numpy.random.randn(4,4).astype(np.int32))

#rand = np.random.random_integers(0,2**10,size=(2**10)*8).astype(np.int32) 

print rand


a = array.to_device(queue, rand, allocator=None, async=False)


scan_kernel = copy_if(a, predicate = "(a[i]<104) ? 1:0")

#a = cl.array.arange(queue, 10000, dtype=np.int32)


out, count, event = scan_kernel(a, queue=queue)

print a 
'''

ctx = cl.create_some_context()

knl = GenericScanKernel(
        ctx, np.int32,
        arguments="__global int *ary, __global int *out",
        input_expr="(ary[i] > 300) ? 1 : 0",
        scan_expr="a+b", neutral="0",
        output_statement="""
            if (prev_item != item) out[item-1] = ary[i];
            """)

out = a.copy()
knl(a, out)

a_host = a.get()
out_host = a_host[a_host > 300]

assert (out_host == out.get()[:len(out_host)]).all()