import hashlib
import numpy

MEM_SIZE = 200

'''
loop through the hard locations
'''

#a = numpy.zeros((MEM_SIZE,), dtype = numpy.uint32)
a = []

for x in range (MEM_SIZE): 
	a.append( hashlib.sha256(str(x)).digest()) 
	print x, a[x]



'''
Ok, running into a problem, 256bits from sha256 won't fit...  come back to this later on...

have to convert the 256 bits into different indexes of a[x]


finally, save sha256.SDM.addresses.pickle

'''


