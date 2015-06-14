/* FROM https://devtalk.nvidia.com/default/topic/524601/opencl-linux-header-files-opencl-status/
SEE ALSO: 

http://devgurus.amd.com/thread/160474
http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#axzz3WS8tpXJK
https://github.com/marwan-abdellah/GPU-Computing-SDK-4.2.9/tree/master/OpenCL/src/oclInlinePTX

inline uint nvidia_popcount(const uint i) {
  uint n;
  asm("popc.b32 %0, %1;" : "=r"(n) : "r" (i));
  return n;
}
*/


/*
   * Hacker's Delight 32 bit pop function:
   * http://www.hackersdelight.org/HDcode/newCode/pop_arrayHS.c.txt
   * 
   * int pop(unsigned x) {
   * x = x - ((x >> 1) & 0x55555555);
   * x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
   * x = (x + (x >> 4)) & 0x0F0F0F0F;
   * x = x + (x >> 8);
   * x = x + (x >> 16);
   * return x & 0x0000003F;
   * }
   * *
   */

  /* 64 bit java version of the C function from above
  x = x - (x >>> 1 & 0x5555555555555555L);
  x = (x & 0x3333333333333333L) + (x >>> 2 & 0x3333333333333333L);
  x = x + (x >>> 4) & 0x0F0F0F0F0F0F0F0FL;
  x = x + (x >>> 8);
  x = x + (x >>> 16);
  x = x + (x >>> 32);
  return (int) x & 0x7F;
*/ 




//For N=256 use ulong4, if N=1024 use ulong16.

/*
Speed test:  if we can send/receive only the active HLs, instead of the entire hash table, how much speed do we get?
(presuming, of course, that additional computation is negligible...)
*/

// HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but less collisions.  It should also be "far" from a power of 2. 

/* BROGLIATO's CODE for reading & writing

bs_len = dimensions = 256

inline int is_bit_true_64(bitstring* a, int bit) {
  int i = bit/64, j = bit%64;
  return (a[bs_len-1-i]&((uint64_t)1<<j) ? 1 : 0);
}

inline int is_bit_true_32(bitstring* a, int bit) {
  int i = bit/32, j = bit%32;
  return (a[bs_len-1-i]&((uint32_t)1<<j) ? 1 : 0);
}






void hl_write(hardlocation* hl, bitstring* data) {
  int i, a;
  for(i=0; i<bs_dimension; i++) 


  //This is the code of the write kernel
  {
    a = bs_bitsign(data, i);
    if (a > 0) {
      if (hl->adder[i] < 127) hl->adder[i]++;
      else printf("@@ WARNING WARNING!\n");
    } else if (a < 0) {
      if (hl->adder[i] > -127) hl->adder[i]--;
      else printf("@@ WARNING WARNING!\n");
    }
  }




}

bitstring* hl_read(hardlocation* hl) {
  return bs_init_adder(bs_alloc(), hl->adder);
}

*/ 


inline int bs_bitsign(bitstring* a, int bit) {
  return (bs_bit(a, bit) ? 1 : -1);
}

__kernel void hl_write(__global uint8 *hl, __global uint8 *data, __global char *HL_adder) 
{
  __local int dim, a;
  for(i=0; i<bs_dimension; i++) 
  //This is the code of the write kernel
  {
    a = bs_bitsign(data, i);
    if (a > 0) {
      if (HL_adder[gid,dim] < 127) HL_adder[gid,dim]++;
    } else if (a < 0) {
      if (HL_adder[gid, dim] > -127) HL_adder[gid,dim]--;
  }
}



#define ACCESS_RADIUS_THRESHOLD 104


__kernel void get_active_hard_locations(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances, __global int *hash_table_gpu)
{
  __private int mem_pos;
  __private ulong4 Aux;
  __private int hash_index;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);

  distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3);

  if (distances[mem_pos]<ACCESS_RADIUS_THRESHOLD)   //104 is the one: 128-24: mu-3sigma. With seed = 123456789 (see python code), we get 1153 Active Hard Locations (re-check this)
  {                                                 
    hash_index = (mem_pos % HASH_TABLE_SIZE);          // Hashing 7 times, see (cormen et al) "introduction to algorihms" section 12.4, on "open addressing". Performance doesn't degrade in the macbook pro.
    if (hash_table_gpu[hash_index]>0) {              // 7 reaches diminishing returns; in parallel the system can read 0s simultaneously, and may have a collision..
      hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE2)) % HASH_TABLE_SIZE;
      if (hash_table_gpu[hash_index]>0) {
        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE3)) % HASH_TABLE_SIZE;
        if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE; 
            if (hash_table_gpu[hash_index]>0) {
              hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE; 
              }
            }
          }
        }
      }  
    }
    hash_table_gpu[hash_index]=mem_pos;
    distances[hash_index]=distances[mem_pos];
  }
}









__kernel void get_active_hard_locations_32bit(__global uint8 *HL_address, __constant uint8 *bitstring, __global int *distances, __global int *hash_table_gpu)
{
  __private uint mem_pos;
  __private uint8 Aux;
  __private uint hash_index;
  __local uint distance ;

  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);

  //distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);
  distance = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);

  if (distance<ACCESS_RADIUS_THRESHOLD)   //104 is the one: 128-24: mu-3sigma. With seed = 123456789 (see python code), we get 1153 Active Hard Locations (re-check this)
  {                                                 
    hash_index = (mem_pos % HASH_TABLE_SIZE);          // Hashing 7 times, see (cormen et al) "introduction to algorihms" section 12.4, on "open addressing". Performance doesn't degrade in the macbook pro.
    if (hash_table_gpu[hash_index]>0) {              // 7 reaches diminishing returns; in parallel the system can read 0s simultaneously, and may have a collision..
      hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE2)) % HASH_TABLE_SIZE;
      if (hash_table_gpu[hash_index]>0) {
        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE3)) % HASH_TABLE_SIZE;
        if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
            hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE; 
            if (hash_table_gpu[hash_index]>0) {
              hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE; 
              if (hash_table_gpu[hash_index]>0) {
                hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE; 
              }
            }
          }
        }
      }  
    }
    hash_table_gpu[hash_index]=mem_pos;
    //distances[hash_index]=distances[mem_pos];   ???
    distances[mem_pos]= distance;
  }
}








/*
The next function has no branches, except for the threshold if (can it be taken off?)
Look st what this guy did here, can we do the same for three possible hash positions?
result = (30 &a_bigger) | (35 & b_bigger) | (40 & equal);
OR YOU CAN AS WELL:
b_bigger = a>b;
b_bigger --;
equal = ~(a_bigger |b_bigger); 
from : https://www.khronos.org/message_boards/showthread.php/8562-Do-i-have-divergence-with-(bool)-val1-val2-operator?p=27979&viewfull=1#post27979
*/
__kernel void get_active_hard_locations_32bit_no_if(__global uint8 *HL_address, __constant uint8 *bitstring, __global uint *distances, __global uint *hash_table_gpu)
{
  __private uint mem_pos;
  __local uint8 Aux;
  __private uint hash_index;
  __local uint distance;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);

  //distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);
  distance = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);

  if (distance<ACCESS_RADIUS_THRESHOLD) //(distances[mem_pos]<ACCESS_RADIUS_THRESHOLD)   //104 is the one: 128-24: mu-3sigma. With seed = 123456789 (see python code), we get 1153 Active Hard Locations (re-check this)
  {                                                 
    hash_index = ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE;
    hash_table_gpu[hash_index]=mem_pos;
  }
  distances[mem_pos]= distance;
}

__kernel void get_HL_distances_from_gpu(__global int *active_hard_locations_gpu, __global int *distances, __global int *final_distances_gpu)
{
  __private uint gid;
  gid = get_global_id(0);  
  final_distances_gpu [gid] = distances [ active_hard_locations_gpu [gid] ];
}




//called by prg.get_active_hard_locations_32bit_no_if_2_dist_computes (queue, (HARD_LOCATIONS,), None, memory_addresses_gpu.data, bitstring_gpu, distances_gpu.data, hash_table_gpu.data ).wait()    
__kernel void get_active_hard_locations_32bit_no_if_2_dist_computes (__global uint8 *HL_address, __constant uint8 *bitstring, __global uint *hash_table_gpu)
{
  __private uint mem_pos;
  __local uint8 Aux;
  __private uint hash_index;
  __local uint distance;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);

  //distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);
  distance = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);

  if (distance<ACCESS_RADIUS_THRESHOLD) //(distances[mem_pos]<ACCESS_RADIUS_THRESHOLD)   //104 is the one: 128-24: mu-3sigma. With seed = 123456789 (see python code), we get 1153 Active Hard Locations (re-check this)
  {                                                 
    hash_index = ( (mem_pos) ^ hash_table_gpu[(mem_pos) %HASH_TABLE_SIZE]) % HASH_TABLE_SIZE;
    hash_table_gpu[hash_index]=mem_pos;
  }
  //distance has to be recomputed later, after the hash table is sorted out...  This way, we never send 2**20 uints... 
}


// called by prg.get_HL_distances_from_gpu_2nd_compute(queue, (count,), None, bitstring_gpu, memory_addresses.gpu.data, active_hard_locations_gpu.data, final_distances_gpu.data)
__kernel void get_HL_distances_from_gpu_2nd_compute(__constant uint8 *bitstring, __global uint8 *HL_address, __global uint *active_hard_locations_gpu, __global uint *final_distances_gpu)
{
  __private uint gid;
  __local uint8 Aux;
        
  gid = get_global_id(0);

  Aux = HL_address[active_hard_locations_gpu[gid]] ^ bitstring[0];
  Aux = popcount(Aux);

  final_distances_gpu[gid] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);
}










__kernel void copy_final_results(__global int *final_locations_gpu, __global int *active_hard_locations_gpu, __global int *final_distances_gpu, __global int *hash_table_gpu)
{
  __private uint gid;
  gid = get_global_id(0);
  final_locations_gpu [gid] = active_hard_locations_gpu [gid];
  final_distances_gpu [gid] = hash_table_gpu [gid];
}




__kernel void compute_distances(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances)
{
  __private uint mem_pos;
  __private ulong4 Aux;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3);
}

__kernel void compute_distances64(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances)
{
  __private uint mem_pos;
  __private ulong4 Aux;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distances [mem_pos] = (uint) (Aux.s0);
}

__kernel void compute_distances128(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances)
{
  __private int mem_pos;
  __private ulong4 Aux;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distances [mem_pos] = (uint) (Aux.s0+Aux.s1);
}


__kernel void compute_distances192(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances)
{
  __private int mem_pos;
  __private ulong4 Aux;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2);
}



//COULD this be faster with no distances buffer?
__kernel void get_active_hard_locations_no_dist_buffer(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *hash_table_gpu)
{
  __private int mem_pos;
  __private ulong4 Aux;
  __private uint hash_index;
  __private uint distance;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distance = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3);

  if (distance<ACCESS_RADIUS_THRESHOLD)   //104 is the one: 128-24: mu-3sigma. With seed = 123456789 (see python code), we get 1153 Active Hard Locations (re-check this)
  {                                                 
    hash_index = (mem_pos % HASH_TABLE_SIZE);          // Hashing 7 times, see (cormen et al) "introduction to algorihms" section 12.4, on "open addressing". Performance doesn't degrade in the macbook pro.
    if (hash_table_gpu[hash_index]>0) {              // 7 reaches diminishing returns; in parallel the system can read 0s simultaneously, and may have a collision..
      hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE2)) % HASH_TABLE_SIZE;
      if (hash_table_gpu[hash_index]>0) {
        hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE3)) % HASH_TABLE_SIZE;
        if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE4)) % HASH_TABLE_SIZE; 
          if (hash_table_gpu[hash_index]>0) {
          hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE5)) % HASH_TABLE_SIZE; 
            if (hash_table_gpu[hash_index]>0) {
              hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE6)) % HASH_TABLE_SIZE; 
              if (hash_table_gpu[hash_index]>0) {
                hash_index= (hash_index + 1 + (mem_pos % HASH_TABLE_SIZE7)) % HASH_TABLE_SIZE; 
              }
            }
          }
        }
      }  
    }
    hash_table_gpu[hash_index]=mem_pos;
  }
}


__kernel void clear_hash_table_gpu(__global uint *hash_table_gpu, __global uint *distances_gpu) 
{
  __private uint gid; 
  gid = get_global_id(0);
  distances_gpu[hash_table_gpu[gid]] = 0;
  hash_table_gpu[gid]=0; 

}

