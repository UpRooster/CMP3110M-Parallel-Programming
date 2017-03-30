#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
// End of Include

// Global Definitions
typedef int mytype;
std::vector<mytype> A;
// Methods
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

float set_time() {
	float msP = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
	return msP;
}

void get_time(float msP) {
	float msN = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count();
	float msD = msN - msP;
	std::cout << "Time Elapsed [Seconds]: " << msD / 1000 << endl;
	return;
}

vector<mytype> readFile(std::ifstream &ifs) {
	std::string line;
	while (std::getline(ifs, line))
	{
		// Get line from input string stream
		std::istringstream iss(line);
		std::size_t found = line.find_last_of(" ");
		float val = std::stof(line.substr(found + 1));
		// cout << std::setprecision(2) << std::fixed << val << " ";
		A.push_back(val * 10);
	}
	return A;
}

// MAIN CLASS
int main(int argc, char **argv) {
	// Get operating device & system statistics
	int platform_id = 0;
	int device_id = 0;
	float msP;
	// Exception detection
	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
	
	try {
		cl::Context context = GetContext(platform_id, device_id);																	// Select compute device (GPU/APU)
		std::cout << "Running on: " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;	// Print compute device (GPU/APU)
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);																	// Create command queue, enable profiling
		cl::Event bubble_event, redSum_event, redMin_event, redMax_event, atoSum_event, atoMin_event, atoMax_event, stdVariance_event, stdSum_event;						// Init profiling events
		cl::Program::Sources sources;																								// Enable sourcing
		AddSources(sources, "my_kernels.cl");																						// Kernel Source File/s
		cl::Program program(context, sources);																						// Define program platform & kernels
		try {	// Attempt to build kernel file/s
			program.build();
		} // Kernel Error Logging
		catch (const cl::Error& err) {
			std::cout << "Build Status: " <<	program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;	// Print status
			std::cout << "Build Options:\t" <<	program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;	// Print options
			std::cout << "Build Log:\t " <<		program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;		// Print error logging
			throw err;	// Output error as defined above
		}

		// Input
		char* filePath = "C:/Users/Computing/Desktop/CMP3110M-Parallel-Programming/Tutorial/BER14475946_CMP3110M/x64/temp_lincolnshire_datasets/temp_lincolnshire.txt";		// Set Filepath
		std::cout << "*Streaming File From Directory*" << std::endl;	// Console Out
		std::ifstream ifs(filePath, std::ifstream::in);					// Begin file streaming
		std::cout << "Acquiring Temp Data.." << std::endl;				// Console Out
		msP = set_time();	// Set time
		readFile(ifs);		// Call readFile method
		get_time(msP);		// Get time taken
		// Padding
		int dataSize = A.size();	// Get size of input data
		size_t local_size = 32;		// Set workgroup size (How many items are placed into a workgroup)
		size_t padding_size = A.size() % local_size;		// Calculate padding size to ensure all workergroups recieve a full dataset, using modulo if there is a match padding will not be added
		std::vector<int> A_ext(local_size-padding_size, 0); // Create an extra vector with 0 values (Padding array)
		A.insert(A.end(), A_ext.begin(), A_ext.end());		// Append A_ext to A to apply padding
		size_t input_elements = A.size();					// Size of input data + padding
		size_t input_size = A.size()*sizeof(mytype);		// Size in bytes
		size_t nr_groups = input_elements / local_size;		// Get the number of workgroups

		// Buffers for Kernel Operations
		std::vector<mytype> B(6);				// Create B vector for holding individual outputs
		std::vector<mytype> C(input_elements);	// Create C vector for holding the modified input for standard deviation
		std::vector<unsigned int> D(1);			// Create D vector for holding the standard deviation sum in unsigned int (Used due to the excessive size of the sum)
		size_t output_size = B.size()*sizeof(mytype);	// Size in bytes
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);		// Creates buffer to pass context (Device Info), flags and data
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);				// ^^ ^^ ^^
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, input_size);				// ^^ ^^ ^^
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));	// ^^ ^^ ^^

		// Device operations
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);	// Copy A to buffer_A
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);				// Create blank buffer for writing/reading
		queue.enqueueFillBuffer(buffer_C, 0, 0, input_size);				// ^^ ^^ ^^
		queue.enqueueFillBuffer(buffer_D, 0, 0, 1 * sizeof(unsigned int));	// ^^ ^^ ^^

		// Kernel Operation ================================
		// SORT
		cl::Kernel sort_bubble = cl::Kernel(program, "sort_bubble");	// Define kernel, call kernel from .cl file
		sort_bubble.setArg(0, buffer_A);	// Pass arguement/buffer to kernel
		sort_bubble.setArg(1, buffer_C);	// ^^ ^^ ^^
		sort_bubble.setArg(2, cl::Local(local_size * sizeof(mytype)));	// Local memory instantiated to local_size(bytes)
		queue.enqueueNDRangeKernel(sort_bubble, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &bubble_event);	// Call kernel in sequence
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, input_size, &C[0]); // Recall data from buffer_C to C

		// REDUCTION
		// Sum kernel
		cl::Kernel reduction_sumK = cl::Kernel(program, "reduction_sum");	// Define kernel, call kernel from .cl file
		reduction_sumK.setArg(0, buffer_A);	// Pass arguement/buffer to kernel
		reduction_sumK.setArg(1, buffer_B);	// ^^ ^^ ^^
		reduction_sumK.setArg(2, cl::Local(local_size*sizeof(mytype)));	// Local memory instantiated to local_size(bytes)

		// Min kernel
		cl::Kernel reduction_gminK = cl::Kernel(program, "reduction_gmin"); // ^^ ^^ ^^
		reduction_gminK.setArg(0, buffer_A);	// ^^ ^^ ^^
		reduction_gminK.setArg(1, buffer_B);	// ^^ ^^ ^^
		reduction_gminK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// ^^ ^^ ^^
		
		// Max kernel
		cl::Kernel reduction_gmaxK = cl::Kernel(program, "reduction_gmax");	// ^^ ^^ ^^
		reduction_gmaxK.setArg(0, buffer_A);	// ^^ ^^ ^^
		reduction_gmaxK.setArg(1, buffer_B);	// ^^ ^^ ^^
		reduction_gmaxK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// ^^ ^^ ^^

		// ATOMIC
		// Sum kernel
		cl::Kernel atomic_sumK = cl::Kernel(program, "atomic_sum");	// ^^ ^^ ^^
		atomic_sumK.setArg(0, buffer_A);	// ^^ ^^ ^^
		atomic_sumK.setArg(1, buffer_B);	// ^^ ^^ ^^
		atomic_sumK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// ^^ ^^ ^^
		// Min kernel
		cl::Kernel atomic_gminK = cl::Kernel(program, "atomic_gmin");	// ^^ ^^ ^^
		atomic_gminK.setArg(0, buffer_A);	// ^^ ^^ ^^
		atomic_gminK.setArg(1, buffer_B);	// ^^ ^^ ^^
		atomic_gminK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// ^^ ^^ ^^

		// Max kernel
		cl::Kernel atomic_gmaxK = cl::Kernel(program, "atomic_gmax");	// ^^ ^^ ^^
		atomic_gmaxK.setArg(0, buffer_A);	// ^^ ^^ ^^
		atomic_gmaxK.setArg(1, buffer_B);	// ^^ ^^ ^^
		atomic_gmaxK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// ^^ ^^ ^^

		
		queue.enqueueNDRangeKernel(reduction_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redSum_event);	// Call kernel in sequence
		queue.enqueueNDRangeKernel(reduction_gminK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redMin_event);	// ^^ ^^ ^^
		queue.enqueueNDRangeKernel(reduction_gmaxK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redMax_event);	// ^^ ^^ ^^
		queue.enqueueNDRangeKernel(atomic_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoSum_event);		// ^^ ^^ ^^
		queue.enqueueNDRangeKernel(atomic_gminK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoMin_event);		// ^^ ^^ ^^
		queue.enqueueNDRangeKernel(atomic_gmaxK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoMax_event);		// ^^ ^^ ^^
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); // Recall data from buffer_B to B
		
		int mean = B[0] / A.size();

		// Standard Deviation
		// Variance kernel
		cl::Kernel std_varianceK = cl::Kernel(program, "std_variance_meansqr");	// Define kernel, call kernel from .cl file
		std_varianceK.setArg(0, buffer_A);	// Pass arguement/buffer to kernel
		std_varianceK.setArg(1, buffer_C);	// ^^ ^^ ^^
		std_varianceK.setArg(2, mean);		// ^^ ^^ ^^
		std_varianceK.setArg(3, dataSize);	// ^^ ^^ ^^
		std_varianceK.setArg(4, cl::Local(local_size * sizeof(mytype)));		// ^^ ^^ ^^
		queue.enqueueNDRangeKernel(std_varianceK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &stdVariance_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, input_size, &C[0]); // Recall data from buffer_C to C

		// Sum kernel
		cl::Kernel std_sumK = cl::Kernel(program, "std_sum");	// Define kernel, call kernel from .cl file
		std_sumK.setArg(0, buffer_C);	// Pass arguement/buffer to kernel
		std_sumK.setArg(1, buffer_D);	// ^^ ^^ ^^
		std_sumK.setArg(2, cl::Local(local_size * sizeof(mytype)));	// Local memory instantiated to local_size(bytes)
		queue.enqueueNDRangeKernel(std_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &stdSum_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, 1 * sizeof(unsigned int), &D[0]); // Recall data from buffer_D to D

		float std_mean = (float)D[0] / C.size();
		float std_sqrt = sqrt(std_mean);

		// Console Output
		std::cout << "========[   Temp Data   ]========" << std::endl;
		std::cout << "Temp Data Values [Unpadded]:	" << dataSize << std::endl;
		std::cout << "Temp Data Values [Padded]:	" << A.size() << std::endl;
		std::cout << "========[    Results    ]========" << std::endl;
		std::cout << "[Reduction]	Sum: " << B[0] << "	Min: " << B[1] / 10.0f << "	Max: " << B[2] / 10.0f << std::endl;

		std::cout << "[Atomic]	Sum: " << B[3] << "	Min: " << B[4] / 10.0f << "	Max: " << B[5] / 10.0f << std::endl;

		std::cout << "Mean:	" << mean / 10.0f << std::endl;
		std::cout << "========[   Deviation   ]========" << std::endl;
		std::cout << "Sum: " << D[0] << "	Mean: " << std_mean << "	Deviation: " << std_sqrt << std::endl;
		//std::cout << A;
		std::cout << "========[   Reduction   ]========" << std::endl;
		std::cout << "Reduction Sum execution time[ns]:	" << redSum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - redSum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Reduction Min execution time[ns]:	" << redMin_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - redMin_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Reduction Max execution time[ns]:	" << redMax_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - redMax_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "========[    Atomic     ]========" << std::endl;
		std::cout << "Atomic Sum execution time[ns]:		" << atoSum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - atoSum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Atomic Min execution time[ns]:		" << atoMin_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - atoMin_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Atomic Max execution time[ns]:		" << atoMax_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - atoMax_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "========[   Deviation   ]========" << std::endl;
		std::cout << "StD Mean/Square execution time[ns]:	" << stdVariance_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - stdVariance_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "StD Sum execution time[ns]:	" << stdSum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - stdSum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "========[    Sorting    ]========" << std::endl;
		std::cout << "Partial Bubble Sort execution time(Workgroup Localized)[ns]:	" << bubble_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - bubble_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		system("pause");
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		system("pause"); // Pause to see error
	}

	return 0;
}