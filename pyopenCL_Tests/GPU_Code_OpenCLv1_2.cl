// FROM https://devtalk.nvidia.com/default/topic/524601/opencl-linux-header-files-opencl-status/
inline uint nvidia_popcount(const uint i) {
  uint n;
  asm("popc.b32 %0, %1;" : "=r"(n) : "r" (i));
  return n;
}




//For N=256 use ulong4, if N=1024 use ulong16.

/*
Speed test:  if we can send/receive only the active HLs, instead of the entire hash table, how much speed do we get?
(presuming, of course, that additional computation is negligible...)
*/

/*
#define HASH_TABLE_SIZE 48781
#define HASH_TABLE_SIZE2 48780
#define HASH_TABLE_SIZE3 48779
#define HASH_TABLE_SIZE4 48778
#define HASH_TABLE_SIZE5 48777
#define HASH_TABLE_SIZE6 48776
#define HASH_TABLE_SIZE7 48775
*/



/*
#define HASH_TABLE_SIZE 12043
#define HASH_TABLE_SIZE2 12042
#define HASH_TABLE_SIZE3 12041
#define HASH_TABLE_SIZE4 12040
#define HASH_TABLE_SIZE5 12039
#define HASH_TABLE_SIZE6 12038
#define HASH_TABLE_SIZE7 12037
*/


// HASH_TABLE_SIZE must be prime.  The higher it is, the more bandwidth, but less collisions.  It should also be "far" from a power of 2. 

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

__kernel void get_active_hard_locations_32bit(__global uint8 *HL_address, __global uint8 *bitstring, __global int *distances, __global int *hash_table_gpu)
{
  __private int mem_pos;
  __private uint8 Aux;
  __private int hash_index;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);

  distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3+Aux.s4+Aux.s5+Aux.s6+Aux.s7);

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

__kernel void get_HL_distances_from_gpu(__global int *active_hard_locations_gpu, __global int *distances, __global int *final_distances_gpu)
{
  __private int gid;
  gid = get_global_id(0);  
  final_distances_gpu [gid] = distances [ active_hard_locations_gpu [gid] ];
}





__kernel void copy_final_results(__global int *final_locations_gpu, __global int *active_hard_locations_gpu, __global int *final_distances_gpu, __global int *hash_table_gpu)
{
  __private int gid;
  gid = get_global_id(0);
  final_locations_gpu [gid] = active_hard_locations_gpu [gid];
  final_distances_gpu [gid] = hash_table_gpu [gid];
}




__kernel void compute_distances(__global ulong4 *HL_address, __global ulong4 *bitstring, __global int *distances)
{
  __private int mem_pos;
  __private ulong4 Aux;
        
  mem_pos = get_global_id(0);

  Aux = HL_address[mem_pos] ^ bitstring[0];
  Aux = popcount(Aux);
  distances [mem_pos] = (uint) (Aux.s0+Aux.s1+Aux.s2+Aux.s3);
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


__kernel void clear_hash_table_gpu(__global int *hash_table_gpu) 
{
  hash_table_gpu[get_global_id(0)]=0; 
}

