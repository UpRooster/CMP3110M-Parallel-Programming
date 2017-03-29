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

void basic_stats(cl::CommandQueue queue, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::Buffer buffer_C, cl::Buffer buffer_D){

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
		cl::Event redSum_event, redMin_event, redMax_event, atoSum_event, atoMin_event, atoMax_event, stdVariance_event, stdSum_event;						// Init profiling events
		cl::Program::Sources sources;																								// Enable sourcing
		AddSources(sources, "my_kernels3.cl");																						// Kernel Source File/s
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

		// Input File
		char* filePath = "C:/Users/Computing/Desktop/CMP3110M-Parallel-Programming/Tutorial/OpenCL Tutorials/x64/Release/temp_lincolnshire.txt";		// Set Filepath
		std::cout << "*Streaming File From Directory*" << std::endl;	// Console Out
		std::ifstream ifs(filePath, std::ifstream::in);					// Begin file streaming
		std::cout << "Acquiring Temp Data.." << std::endl;				// Console Out
		msP = set_time();	// Set time
		readFile(ifs);		// Call readFile method
		get_time(msP);		// Get time taken
		int dataSize = A.size();
		// Apply padding to increase efficiency
		size_t local_size = 32;
		size_t padding_size = A.size() % local_size;
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		// Buffers for Kernel Operations
		std::vector<mytype> B(6);
		std::vector<mytype> C(input_elements);
		std::vector<unsigned int> D(1);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, 1 * sizeof(unsigned int));

		// Device operations

		// Copy buffer_a from A and setup other buffers for kernel operations
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, input_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, 1 * sizeof(unsigned int));//zero B buffer on device memory

		// Kernel Operation ================================
		// REDUCTION
		// Sum kernel
		cl::Kernel reduction_sumK = cl::Kernel(program, "reduction_sum");
		reduction_sumK.setArg(0, buffer_A);
		reduction_sumK.setArg(1, buffer_B);
		reduction_sumK.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		// Min kernel
		cl::Kernel reduction_gminK = cl::Kernel(program, "reduction_gmin");
		reduction_gminK.setArg(0, buffer_A);
		reduction_gminK.setArg(1, buffer_B);
		reduction_gminK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory siz
		
		// Max kernel
		cl::Kernel reduction_gmaxK = cl::Kernel(program, "reduction_gmax");
		reduction_gmaxK.setArg(0, buffer_A);
		reduction_gmaxK.setArg(1, buffer_B);
		reduction_gmaxK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		// ATOMIC
		// Sum kernel
		cl::Kernel atomic_sumK = cl::Kernel(program, "atomic_sum");
		atomic_sumK.setArg(0, buffer_A);
		atomic_sumK.setArg(1, buffer_B);
		atomic_sumK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		// Min kernel
		cl::Kernel atomic_gminK = cl::Kernel(program, "atomic_gmin");
		atomic_gminK.setArg(0, buffer_A);
		atomic_gminK.setArg(1, buffer_B);
		atomic_gminK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory siz

		// Max kernel
		cl::Kernel atomic_gmaxK = cl::Kernel(program, "atomic_gmax");
		atomic_gmaxK.setArg(0, buffer_A);
		atomic_gmaxK.setArg(1, buffer_B);
		atomic_gmaxK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		queue.enqueueNDRangeKernel(reduction_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redSum_event); // Call kernel in sequence
		queue.enqueueNDRangeKernel(reduction_gminK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redMin_event); // Call kernel in sequence
		queue.enqueueNDRangeKernel(reduction_gmaxK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &redMax_event); // Call kernel in sequence
		queue.enqueueNDRangeKernel(atomic_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoSum_event); // Call kernel in sequence
		queue.enqueueNDRangeKernel(atomic_gminK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoMin_event); // Call kernel in sequence
		queue.enqueueNDRangeKernel(atomic_gmaxK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &atoMax_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); // Add kernel to queue
		
		int mean = B[0] / A.size();

		// Standard Deviation
		// Variance kernel
		cl::Kernel std_varianceK = cl::Kernel(program, "std_variance_meansqr");
		std_varianceK.setArg(0, buffer_A);
		std_varianceK.setArg(1, buffer_C);
		std_varianceK.setArg(2, mean);
		std_varianceK.setArg(3, dataSize);
		std_varianceK.setArg(4, cl::Local(local_size * sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(std_varianceK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &stdVariance_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, input_size, &C[0]); // Add kernel to queue

		// cout << C;

		// Sum kernel
		cl::Kernel std_sumK = cl::Kernel(program, "std_sum");
		std_sumK.setArg(0, buffer_C);
		std_sumK.setArg(1, buffer_D);
		std_sumK.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(std_sumK, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &stdSum_event); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, 1 * sizeof(unsigned int), &D[0]); // Add kernel to queue

		std::cout << "========[   Temp Data   ]========" << std::endl;
		std::cout << "Temp Data Values [Unpadded]:	" << dataSize << std::endl;
		std::cout << "Temp Data Values [Padded]:	" << A.size() << std::endl;
		std::cout << "========[    Results    ]========" << std::endl;
		std::cout << "[Reduction]	Sum: " << B[0] << "	Min: " << B[1] / 10.0f << "	Max: " << B[2] / 10.0f << std::endl;

		std::cout << "[Atomic]	Sum: " << B[3] << "	Min: " << B[4] / 10.0f << "	Max: " << B[5] / 10.0f << std::endl;

		std::cout << "Mean:	" << mean / 10.0f << std::endl;
		float std_mean = (float)D[0] / C.size();
		float std_sqrt = sqrt(std_mean);
		std::cout << "========[   Deviation   ]========" << std::endl;
		std::cout << "Sum: " << D[0] << "	Mean: " << std_mean << "	Deviation: " << std_sqrt << std::endl;

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

		

		system("pause");
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		system("pause"); // Pause to see error
	}

	return 0;
}