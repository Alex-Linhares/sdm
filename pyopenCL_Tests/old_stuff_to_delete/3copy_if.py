#https://github.com/pyopencl/pyopencl/blob/master/test/test_algorithm.py

from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la
import sys
from pytools import memoize
#from test_array import general_clrand

import pytest

import pyopencl as cl
import pyopencl.array as cl_array  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from pyopencl.characterize import has_double_support
from pyopencl.scan import InclusiveScanKernel, ExclusiveScanKernel

PYOPENCL_COMPILER_OUTPUT=1

scan_test_counts = [2 ** 12 + 5]

def test_copy_if(ctx_factory):
    from pytest import importorskip
    importorskip("mako")

    from pyopencl.clrandom import rand as clrand

    
    for n in scan_test_counts:
        a_dev = clrand(queue, (n,), dtype=np.int32, a=0, b=1000)
        #a = a_dev.get()

        from pyopencl.algorithm import copy_if

        crit = a_dev.dtype.type(6)
        #selected = a[a < crit]
        selected_dev, count_dev, evt = copy_if(a_dev, "ary[i] < myval", [("myval", crit)])

        print count_dev
        print selected_dev[selected_dev>0]
        
        #assert (selected_dev.get()[:count_dev.get()] == selected).all()
        #from gc import collect
        #collect()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mem_flags = cl.mem_flags

test_copy_if(ctx)