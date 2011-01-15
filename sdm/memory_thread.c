#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

#include "memory_thread.h"
#include "hardlocation.h"
#include "bitstring.h"

#define min(a, b) ((a)<(b) ? (a) : (b))

extern unsigned int bs_dimension;

extern unsigned int sdm_sample;
extern unsigned int sdm_radius;
extern hardlocation** sdm_memory;

unsigned int sdm_thread_count = 4;

/*
void* sdm_thread_initialize_task(void* ptr) {
	for(i=0; i<sdm_sample; i++) {
		sdm_memory[i] = hl_init_random(hl_alloc());
	}
}

int sdm_thread_initialize() {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	unsigned int i;

	bs_initialize();
	hl_initialize();

	sdm_memory = (hardlocation**) malloc(sizeof(hardlocation*)*sdm_sample);
	assert(sdm_memory != NULL);

	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		pthread_create(&thread[i], NULL, sdm_thread_distance_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}

	return 0;
}

void sdm_free() {
	unsigned int i;

	for(i=0; i<sdm_sample; i++) {
		hl_free(sdm_memory[i]);
	}

	free(sdm_memory);
}
*/

inline void sdm_thread_offset(sdm_thread_params* params, unsigned int* id, unsigned int* offset, unsigned int* len) {
	int qty, extra;
	qty = sdm_sample/sdm_thread_count;
	extra = sdm_sample%sdm_thread_count;
	*id = params->id;
	*len = qty + (*id < extra ? 1 : 0);
	*offset = (*id)*qty + min(*id, extra);
	//printf("@@ (%d) %d [%d, %d]\n", myid, len, offset, offset+len-1);
}

void* sdm_thread_read_task(void* ptr) {
	sdm_thread_params* params = (sdm_thread_params*) ptr;
	unsigned int myid, offset, len;
	unsigned int i, j, k, counter = 0;
	unsigned int dist;

	sdm_thread_offset(params, &myid, &offset, &len);

	for(i=0; i<bs_dimension; i++) params->adder[i] = 0;
	for(i=0; i<len; i++) {
		dist = bs_distance(sdm_memory[offset+i]->address, params->address);
		if (dist <= sdm_radius) {
			for(j=0; j<bs_dimension; j++) {
				if (sdm_memory[offset+i]->adder[j] > 0) k = 1;
				else if (sdm_memory[offset+i]->adder[j] < 0) k = -1;
				else k = (rand()%2 == 0 ? 1 : -1);
				params->adder[j] += k;
			}
			counter++;
		}
	}

	params->counter = counter;
	return NULL;
}

bitstring* sdm_thread_read(bitstring* address) {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	int32_t adder[sdm_thread_count][bs_dimension];
	adder_t adder2[bs_dimension];
	unsigned int i, j;
	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		params[i].address = address;
		params[i].adder = adder[i];
		pthread_create(&thread[i], NULL, sdm_thread_read_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}
	// we can't add all adders in an adder_t type because
	// it will probably overflow.
	for(i=0; i<sdm_thread_count; i++) {
		for(j=0; j<bs_dimension; j++) {
			if (adder[j] > 0) adder2[j] = 1;
			else if (adder[j] < 0) adder2[j] = -1;
			else adder2[j] = 0;
		}
	}
	//printf("Hardlocation inside radius %d = %d\n", sdm_radius, counter);
	return bs_init_adder(bs_alloc(), adder2);
}

void* sdm_thread_write_task(void* ptr) {
	sdm_thread_params* params = (sdm_thread_params*) ptr;
	unsigned int myid, offset, len;
	unsigned int i, counter = 0;
	unsigned int dist;

	sdm_thread_offset(params, &myid, &offset, &len);

	for(i=0; i<len; i++) {
		dist = bs_distance(sdm_memory[offset+i]->address, params->address);
		if (dist <= sdm_radius) {
			hl_write(sdm_memory[offset+i], params->data);
			counter++;
		}
	}

	params->counter = counter;
	return NULL;
}

unsigned int sdm_thread_write(bitstring* address, bitstring* data) {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	unsigned int i, counter;
	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		params[i].address = address;
		params[i].data = data;
		pthread_create(&thread[i], NULL, sdm_thread_write_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}
	counter = 0;
	for(i=0; i<sdm_thread_count; i++) {
		counter += params[i].counter;
	}
	return counter;
}

void* sdm_thread_radius_count_intersect_task(void* ptr) {
	sdm_thread_params* params = (sdm_thread_params*) ptr;
	unsigned int myid, offset, len;
	unsigned int i, counter = 0;
	unsigned int d1, d2;

	sdm_thread_offset(params, &myid, &offset, &len);

	for(i=0; i<len; i++) {
		d1 = bs_distance(sdm_memory[offset+i]->address, params->addr1);
		d2 = bs_distance(sdm_memory[offset+i]->address, params->addr2);
		if (d1 <= params->radius && d2 <= params->radius) {
			counter++;
		}
	}

	params->counter = counter;
	return NULL;
}

unsigned int sdm_thread_radius_count_intersect(bitstring* addr1, bitstring* addr2, unsigned int radius) {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	unsigned int i, counter;
	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		params[i].addr1 = addr1;
		params[i].addr2 = addr2;
		params[i].radius = radius;
		pthread_create(&thread[i], NULL, sdm_thread_radius_count_intersect_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}
	counter = 0;
	for(i=0; i<sdm_thread_count; i++) {
		counter += params[i].counter;
	}
	return counter;
}

void* sdm_thread_radius_count_task(void* ptr) {
	sdm_thread_params* params = (sdm_thread_params*) ptr;
	unsigned int myid, offset, len;
	unsigned int i, counter = 0;
	unsigned int dist;

	sdm_thread_offset(params, &myid, &offset, &len);

	for(i=0; i<len; i++) {
		dist = bs_distance(sdm_memory[offset+i]->address, params->address);
		if (dist <= params->radius) {
			counter++;
		}
	}

	params->counter = counter;
	return NULL;
}

unsigned int sdm_thread_radius_count(bitstring* address, unsigned int radius) {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	unsigned int i, counter;
	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		params[i].address = address;
		params[i].radius = radius;
		pthread_create(&thread[i], NULL, sdm_thread_radius_count_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}
	counter = 0;
	for(i=0; i<sdm_thread_count; i++) {
		counter += params[i].counter;
	}
	return counter;
}

void* sdm_thread_distance_task(void* ptr) {
	sdm_thread_params* params = (sdm_thread_params*) ptr;
	unsigned int myid, offset, len;
	unsigned int i;

	sdm_thread_offset(params, &myid, &offset, &len);

	for(i=0; i<len; i++) {
		params->res[offset+i] = bs_distance(sdm_memory[offset+i]->address, params->address);
	}

	return NULL;
}

void sdm_thread_distance(bitstring* address, unsigned int *res) {
	pthread_t thread[sdm_thread_count];
	sdm_thread_params params[sdm_thread_count];
	unsigned int i;
	for(i=0; i<sdm_thread_count; i++) {
		params[i].id = i;
		params[i].address = address;
		params[i].res = res;
		pthread_create(&thread[i], NULL, sdm_thread_distance_task, (void*) &params[i]);
	}
	for(i=0; i<sdm_thread_count; i++) {
		pthread_join(thread[i], NULL);
	}
}

