///
/// ref nv impl
///
/// lix19937
///

#include "tools.h"
#include <unistd.h>
#include <fstream>
#include <vector>
#include <cassert>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <numeric>
#include <future>
#include <csignal>
#include <cstdlib>
#include <chrono>
#include <string>
using namespace nvinfer1;

class Task
{
public:
	Task(const char *filename, int dlaCore, int streamCreateFlag = cudaStreamDefault, int eventCreateFlag = cudaEventDefault);
	void exec_once(int batchSize = 1)
	{
		//
		const bool success = context->enqueue(batchSize, bindingArray.data(), stream.get(), nullptr);
		assert(success);
	}

	enum class SyncType
	{
		Stream,
		Event
	};
	std::thread repeatWithSync(SyncType syncType, std::function<void(Task &)> threadInit = nullptr);

	static void signalHandler(int s)
	{
		gStop = true;
	}

	static bool shouldStop()
	{
		return gStop;
	}

	// called from master thread when repeatWithSync is running
	float reportFPS()
	{
		const size_t frames = counter->load();
		counter->fetch_sub(frames); // reset counter safely
		auto timeEnd = std::chrono::steady_clock::now();
		const float fps = frames / std::chrono::duration_cast<std::chrono::duration<float>>(timeEnd - timeBeg).count();
		timeBeg = timeEnd;
		return fps;
	}

private:
	static Logger gLogger;
	static std::atomic_bool gStop;

	UniqPtr<IRuntime> runtime = UniqPtr<IRuntime>{createInferRuntime(gLogger)};
	std::unique_ptr<std::atomic_size_t> counter = std::unique_ptr<std::atomic_size_t>(new std::atomic_size_t(0));
	UniqPtr<ICudaEngine> engine = nullptr;
	UniqPtr<IExecutionContext> context = nullptr;
	std::vector<UniqPtr<char, CuMemDeleter>> bindings;
	std::vector<void *> bindingArray; // same content as bindings
	std::unique_ptr<CUstream_st, StreamDeleter> stream;
	std::unique_ptr<CUevent_st, EventDeleter> event;

	// for use by host thread
	std::chrono::time_point<std::chrono::steady_clock> timeBeg;
};

Logger Task::gLogger;
std::atomic_bool Task::gStop;

Task::Task(const char *filename, int dlaCore, int streamCreateFlag, int eventCreateFlag)
		: stream{makeCudaStream(streamCreateFlag)}, event{makeCudaEvent(eventCreateFlag)}
{
	if (dlaCore >= 0)
		runtime->setDLACore(dlaCore);
	std::cout << "Load engine from :" << filename << std::endl;
	std::ifstream fin(filename, std::ios::binary);
	std::vector<char> inBuffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	engine.reset(runtime->deserializeCudaEngine(inBuffer.data(), inBuffer.size(), nullptr));
	assert(engine);
	context.reset(engine->createExecutionContext());

	const int nbBindings = engine->getNbBindings();
	for (int i = 0; i < nbBindings; i++)
	{
		const auto dataType = engine->getBindingDataType(i);
		const int elemSize = [&]() -> int
		{
			switch (dataType)
			{
			case DataType::kFLOAT:
				return 4;
			case DataType::kHALF:
				return 2;
			default:
				throw std::runtime_error("invalid data type");
			}
		}();
		const auto dims = engine->getBindingDimensions(i);
		const int bindingSize = elemSize * std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});
		bindings.emplace_back(mallocCudaMem<char>(bindingSize));
		bindingArray.emplace_back(bindings.back().get());
	}
}

std::thread Task::repeatWithSync(Task::SyncType syncType, std::function<void(Task &)> threadInit)
{
	return std::thread([this, syncType, threadInit]()
										 {
		if (threadInit)
			threadInit(*this);
		counter->store(0);
		timeBeg = std::chrono::steady_clock::now();
		while(!gStop){
			exec_once();
			switch (syncType) {
			case SyncType::Stream: checkCudaErrors(cudaStreamSynchronize(stream.get())); break;
			case SyncType::Event:
                                checkCudaErrors(cudaEventRecord(event.get(), stream.get()));
                                checkCudaErrors(cudaEventSynchronize(event.get()));
                                break;
			default: throw std::runtime_error("invalid sync type");
			}
			counter->fetch_add(1);
		} });
}

int main(int argc, char *argv[])
{
	std::cout << "Main ..." << std::endl;
	if (argc == 1)
	{
		std::cout << "Usage: ./test [engine]" << std::endl;
		return 0;
	}

	// Configurations
	const int streamCreateFlags = cudaStreamDefault;
	const int eventCreateFlags = cudaEventBlockingSync;
	auto threadInit = [](Task &task) -> void
	{
		// If you want to do something at the begining of each worker threads, put it here.
		// ...
		//
		//
	};
	const Task::SyncType syncType = Task::SyncType::Event;

	// Configuration is done. Now we start.
	signal(SIGINT, Task::signalHandler);

	std::vector<Task> tasks;
	for (int i = 0; i < argc - 1; i++)
	{
		const int dlaCore = std::stoi(argv[i + 1]);
		const char *filename = dlaCore < 0 ? "gpu.engine" : "dla.engine";
		tasks.emplace_back(filename, dlaCore, streamCreateFlags, eventCreateFlags);
	}

	std::vector<std::thread> workers;
	for (Task &task : tasks)
		workers.emplace_back(task.repeatWithSync(syncType, threadInit));

	while (!Task::shouldStop())
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::cout << "FPS:\t";
		for (auto &task : tasks)
			std::cout << task.reportFPS() << "\t";
		std::cout << std::endl;
	}

	for (auto &thrd : workers)
		thrd.join();
	workers.clear();

	std::cout << "All done" << std::endl;

	return 0;
}
