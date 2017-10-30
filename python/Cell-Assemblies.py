#from bitstring import *
import numpy
#import slipnode
#import slipnet
#from slipnet import *
#from slipnode import *

from pygraphviz import *


# 1. Inherit Class, add methods (start with N=64, fixed)
# 1a. Create KBitArray;

# 2. Class KNetworks --> leads with s & Generates Graphs
# 2a. generates nodes, edges, and has an observer module that generates the graphs.

'''

N = 256
mu = 128
sigma = 8.0
sigma_threshold = 4
max_sigma = -16.0


def Get_Bitstring():
    bitstring256 = numpy.random.random_integers(0,2**32,size=8).astype(numpy.uint32)
    rnd = BitArray(uint=bitstring256[0], length=32)+BitArray(uint=bitstring256[1], length=32)\
    +BitArray(uint=bitstring256[2], length=32)+BitArray(uint=bitstring256[3], length=32)\
    +BitArray(uint=bitstring256[4], length=32)+BitArray(uint=bitstring256[5], length=32)\
    +BitArray(uint=bitstring256[6], length=32)+BitArray(uint=bitstring256[7], length=32)
    return rnd

def Hamming (s1, s2):
    xor = s1 ^ s2
    return xor.count(1)


def Num_Sigma (h):
    x_sigma = (h-mu)/sigma
    return x_sigma


def Sigma_dist_from_origin (origin, dest):
    x= Num_Sigma(Hamming(origin, dest))
    S_from_origin = x-max_sigma
    return S_from_origin

def GraphVizDist (origin,dest):
    GV_Dist= Sigma_dist_from_origin(origin, dest)/(- max_sigma)
    Exponent = 6
    GV_Dist = GV_Dist**Exponent
    if GV_Dist>1.0:
      GV_Dist=1.0
    return GV_Dist

  #This function creates a r-link from a to b (to be saved in memory r)
  #### ATTENTION JIMMY SHOULD BE MOVED TO THE NETWORKS CLASS!
def GenLink (r, origin, dest):

    link_part1 = (origin & ~r)
    link_part2 = dest & r

    link = link_part1 | link_part2
    return link


def Get_link_end_point (r, link):

    returnlink_part1 = (~link & ~r)
    returnlink_part2 = link & r

    returnlink = returnlink_part1 | returnlink_part2
    return returnlink




def Get_self_link_end_point (r, link):
    return (link)

'''

#======================================
# slipnet

#N = slipnet.slipnodes
#L = slipnet.sliplinks


Node_Names = []

G = AGraph(strict='false', directed='true', landscape='false', ranksep='2.5', splines='true', overlap='scalexy', nodesep='0.6', elenght=1.0, eweight=0.5)

for node in N:
    Node_Names.append(node.get_name())
    G.add_node(node.get_name())
    for node2 in N:
        if node.linked(node2):
            # e = linked_node.points_at(node, )
            name = node2.get_name()
            G.add_edge(node.get_name(), node2.get_name())

'''for node in N:
    Node_Names.append(node.get_name())
    G.add_node(node.get_name())
    for linked_node in L:
        e = linked_node.points_at(node, )
        name = lnode.get_name()
        G.add_edge(node.get_name(), name)
'''

filename = 'slipnet'
program = 'neato'
G.layout(program, args='-Gepsilon=0.000001; overlap="false"')
file_name = program + "_" + filename
G.write(file_name + "_output.xdot")
G.draw(file_name + ".svg")


'''


'''
