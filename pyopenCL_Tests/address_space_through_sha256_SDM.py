import hashlib
import numpy


def get_bitstring(bitstring_name):
	hash_string = hashlib.sha256(str(bitstring_name)).hexdigest()
	bitstring = numpy.zeros(8,dtype = numpy.uint32)
	for x in range (8): 
	    hash_substring = '0x'+hash_string[x*8:x*8+8]
	    bitstring [x] = int(hash_substring, 16) & 0xFFFFFFFF
	    # if verbose print hash_substring, bitstring[x], hex(bitstring[x])
	return bitstring


def get_address_space(HARD_LOCATIONS):
	HL_Address = numpy.zeros((HARD_LOCATIONS,8), dtype = numpy.uint32)
	for x in range(HARD_LOCATIONS):
		address = get_bitstring(str(x))
		for y in range(8):
			HL_Address[x,y]= address[y]
		#print address
	return HL_Address


def create_address_space_pickle(HARD_LOCATIONS):
    address_space = get_address_space(HARD_LOCATIONS)
    import cPickle
    out = open('hard_locations.sha256.sdm.pickle', 'wb')
    cPickle.dump (address_space, out, cPickle.HIGHEST_PROTOCOL)
    out.close()



create_address_space_pickle(2**20)



'''
Ok, running into a problem, 256bits from sha256 won't fit...  come back to this later on...

have to convert the 256 bits into different indexes of a[x]


finally, save sha256.SDM.addresses.pickle

'''


