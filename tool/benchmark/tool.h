#include <stdexcept>
#include <iostream>
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <memory>

void checkCudaErrors(cudaError_t err)
{
	if (err != cudaSuccess)
		throw std::runtime_error(cudaGetErrorName(err));
}

constexpr int divUp(int a, int b) { return (a + b - 1) / b; }

// RAII helpers to automatically manage memory resource and TensorRT objects.
template <typename T>
struct TrtDeleter
{
	void operator()(T *p) noexcept { p->destroy(); }
};

template <typename T>
struct CuMemDeleter
{
	void operator()(T *p) noexcept { checkCudaErrors(cudaFree(p)); }
};

template <typename T, template <typename> typename DeleterType = TrtDeleter>
using UniqPtr = std::unique_ptr<T, DeleterType<T>>;

template <typename T>
UniqPtr<T, CuMemDeleter> mallocCudaMem(size_t nbElems)
{
	T *ptr = nullptr;
	checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * nbElems));
	return UniqPtr<T, CuMemDeleter>{ptr};
}

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
			: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char *msg) noexcept override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "UNKNOWN: ";
			break;
		}
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity;
};

extern Logger gLogger;

struct StreamDeleter
{
	void operator()(CUstream_st *stream) noexcept
	{
		checkCudaErrors(cudaStreamDestroy(stream));
	}
};

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags = cudaStreamDefault)
{
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, flags));
	return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

struct EventDeleter
{
	void operator()(CUevent_st *event) noexcept
	{
		checkCudaErrors(cudaEventDestroy(event));
	}
};

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags = cudaEventDefault)
{
	cudaEvent_t event;
	checkCudaErrors(cudaEventCreateWithFlags(&event, flags));
	return std::unique_ptr<CUevent_st, EventDeleter>{event};
}
