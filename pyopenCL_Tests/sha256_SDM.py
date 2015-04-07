import hashlib
import numpy

hash_string = hashlib.sha256(str('foo')).hexdigest()
bitstring = numpy.zeros(8,dtype = numpy.uint32)
for x in range (8): 
    hash_substring = '0x'+hash_string[x*8:x*8+8]

    #struct.unpack('f',"ED6F3C01".decode('hex'))
    bitstring [x] = int(hash_substring, 16) & 0xFFFFFFFF
    print hash_substring, bitstring[x], hex(bitstring[x])



'''
Ok, running into a problem, 256bits from sha256 won't fit...  come back to this later on...

have to convert the 256 bits into different indexes of a[x]


finally, save sha256.SDM.addresses.pickle

'''


