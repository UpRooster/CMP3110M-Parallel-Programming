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
typedef int mytype;
std::vector<mytype> A;
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
	std::cout << "Time Elapsed[ms]: " << msD << endl;
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
		// Select compute device
		cl::Context context = GetContext(platform_id, device_id);
		// Show compute device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		// Create command queue
		cl::CommandQueue queue(context);
		// Load kernels
		cl::Program::Sources sources;
		AddSources(sources, "my_kernels3.cl");
		cl::Program program(context, sources);
		// Attempt to build kernels
		try {
			program.build();
		} // Error logging
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		// Variable memory allocation

		// Main Operation ================================
		// Input File
		std::cout << "Loading Input" << std::endl;
		std::ifstream ifs("C:/Users/Computing/Desktop/CMP3110M-Parallel-Programming/Tutorial/OpenCL Tutorials/x64/Release/temp_lincolnshire.txt", std::ifstream::in);

		// Strip Values
		std::cout << "Acquiring Values" << std::endl;
		msP = set_time();	// Set time
		readFile(ifs);
		get_time(msP);		// Get time taken
		std::cout << "Input Size:	" << A.size() << std::endl;
		std::cout << "Padding!" << std::endl;


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
		std::cout << "With Padding:	" << A.size() << std::endl;
		std::cout << "Creating Buffers" << std::endl;

		// Buffers for Kernel Operations
		std::vector<mytype> B(4);
		std::vector<mytype> C(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);

		// Device operations

		// Copy buffer_a from A and setup other buffers for kernel operations
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero B buffer on device memory

		// Kernel Operation ================================
		// Sum kernel
		cl::Kernel kernel_1 = cl::Kernel(program, "sum");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		
		// Min kernel
		cl::Kernel kernel_2 = cl::Kernel(program, "gmin");
		kernel_2.setArg(0, buffer_A);
		kernel_2.setArg(1, buffer_B);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory siz
		
		// Max kernel
		cl::Kernel kernel_3 = cl::Kernel(program, "gmax");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_B);
		kernel_3.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // Call kernel in sequence
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // Call kernel in sequence
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); // Add kernel to queue
		
		int mean = B[0] / A.size();
		
		std::cout << "Sum:	" << B[0] << std::endl;
		std::cout << "Min:	" << B[1]/10.0f << std::endl;
		std::cout << "Max:	" << B[2]/10.0f << std::endl;
		std::cout << "Mean:	" << mean/10.0f << std::endl;

		// StD_variance kernel
		cl::Kernel kernel_4 = cl::Kernel(program, "std_variance");
		kernel_4.setArg(0, buffer_A);
		kernel_4.setArg(1, buffer_C);
		kernel_4.setArg(2, mean);
		kernel_4.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory siz
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &C[0]); // Add kernel to queue

		// StD_sum kernel
		cl::Kernel kernel_5 = cl::Kernel(program, "sum");
		kernel_5.setArg(0, buffer_A);
		kernel_5.setArg(1, buffer_B);
		kernel_5.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(kernel_5, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); // Call kernel in sequence
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[3]); // Add kernel to queue

		int std_mean = C[0] / A.size();
		std::cout << "StD Variance:	" << std_mean << std::endl;
		system("pause");
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		system("pause"); // Pause to see error
	}

	return 0;
}
