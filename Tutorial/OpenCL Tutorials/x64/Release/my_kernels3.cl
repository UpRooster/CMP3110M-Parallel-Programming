// Kernels

//======================================================================================//
//======================================Reduction=======================================//
__kernel void reduction_sum(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

__kernel void reduction_gmin(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] > scratch[lid + i])
			scratch[lid] = scratch[lid + i];}

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	
	atomic_min(&B[1],scratch[lid]);
	
}

__kernel void reduction_gmax(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] < scratch[lid + i])
			scratch[lid] = scratch[lid + i];}

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	
	atomic_max(&B[2],scratch[lid]);
	
}

//======================================================================================//
//========================================Atomic========================================//
__kernel void atomic_sum(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[3],scratch[lid]);
	}
}

__kernel void atomic_gmin(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	
	atomic_min(&B[4],scratch[lid]);
	
}

__kernel void atomic_gmax(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	
	atomic_max(&B[5],scratch[lid]);	// Deter race conditions (Atomic functions limit variable accessiblity)
	
}

//======================================================================================//
//=====================================St Deviation=====================================//
__kernel void std_variance_meansqr(__global const int* A, __global int* C, int mean, int dataSize, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	if (id < dataSize)	// Only operate to original data size (before padding)
		scratch[lid] = (A[id] - mean);	// Subtract mean from all values
	
	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	C[id] = (scratch[lid] * scratch[lid]);	// Square the result and output to C vector
}



__kernel void std_sum(__global const int* C, __global int* D, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = C[id];	// Copy global values to local memory

	barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads

	for (int i = 1; i < N; i *= 2) {	// Add all values in kernel
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];	// Incremental Add

		barrier(CLK_LOCAL_MEM_FENCE);	// Sync threads
	}
	scratch[lid] = scratch[lid] / 100.0f;	// / 100.f to ensure data is converted back to float

	if (!lid) {
		atomic_add(&D[0],scratch[lid]);	// Deter race conditions (Atomic functions limit variable accessiblity)
	}
}
//======================================================================================//
//======================================Functions=======================================//
/*void ascend (int A, int C){
	if (A > C){
		int temp = C;
		A = C;
		C = temp;
	}
}

void descend (int A, int B){
	if (A < B){
		int temp = A;
		A = B;
		B = temp;
	}
}*/ // Unused functions for sorting with Merge/Bitonic

//======================================================================================//
//=======================================Sorting========================================//
__kernel void sort_bubble(__global int* A, __global int* C, __local int* scratch){
	int id = get_global_id(0);	// Global Data ID
	int lid = get_local_id(0);	// Kernel Data ID
	int N = get_local_size(0);	// Used for local sorting
	int M = get_global_size(0); // Used for global sorting (NON-FUNCTIONAL | WILL CRASH PC)

	scratch[lid] = A[id];		// Global to Local Memory

	for (int i=0; i < N-1; i++){	// For the length of the workgroup array
		for (int j=0; j < N-i; j++){	// ^^ ^^ ^^
			if (scratch[j] > scratch[j+1]){	// If the current value is greater then the next value, ascend current
				int temp = scratch[j];
				scratch[j] = scratch[j+1];
				scratch[j+1] = temp;
			}
		}
		if (scratch[i] > scratch[i+1]){	// As above, acts as a "Capture" for stray values
			int temp = scratch[i];
			scratch[i] = scratch[i+1];
			scratch[i+1] = temp;
		}
	}
	C[id] = scratch[lid];	// Copy values from local to global (C Vector)
}// Defunct Sort -- Only sorts workgroups (But it's super fast! 132000[ns])