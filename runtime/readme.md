**相对于 build or compile 期间**         
runtime 属性     

https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/sampleOptions.cpp#L2215    
```
    os << "=== Inference Options ==="                                                                                                << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "              << defaultBatch << ")"  << std::endl <<
          "                              This option should not be used when the engine is built from an ONNX model or when dynamic" << std::endl <<
          "                              shapes are provided when the engine is built."                                              << std::endl <<
          "  --shapes=spec               Set input shapes for dynamic shapes inference inputs."                                      << std::endl <<
          R"(                              Note: Input names can be wrapped with escaped single quotes (ex: 'Input:0').)"            << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128"                          << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"           << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."         << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                     << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                 << std::endl <<
          "  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be "
                                                                                       "wrapped with single quotes (ex: 'Input:0')"  << std::endl <<
          R"(                            Input values spec ::= Ival[","spec])"                                                       << std::endl <<
          R"(                                         Ival ::= name":"file)"                                                         << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "               << defaultIterations << ")"  << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                            << defaultWarmUp << ")"  << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                                          << defaultDuration << ")"  << std::endl <<
          "                              If -1 is specified, inference will keep running unless stopped manually"                    << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                               "(default = " << defaultSleep << ")"  << std::endl <<
          "  --idleTime=N                Sleep N milliseconds between two continuous iterations"
                                                                                               "(default = " << defaultIdle << ")"   << std::endl <<
          "  --infStreams=N              Instantiate N engines to run inference concurrently (default = "  << defaultStreams << ")"  << std::endl <<
          "  --exposeDMA                 Serialize DMA transfers to and from device (default = disabled)."                           << std::endl <<
          "  --noDataTransfers           Disable DMA transfers to and from device (default = enabled)."                              << std::endl <<
          "  --useManagedMemory          Use managed memory instead of separate host and device allocations (default = disabled)."   << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                             "increase CPU usage and power (default = disabled)"     << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads"
                                                                                " or speed up refitting (default = disabled) "       << std::endl <<
          "  --useCudaGraph              Use CUDA graph to capture engine execution and then launch inference (default = disabled)." << std::endl <<
          "                              This flag may be ignored if the graph capture fails."                                       << std::endl <<
          "  --timeDeserialize           Time the amount of time it takes to deserialize the network and exit."                      << std::endl <<
          "  --timeRefit                 Time the amount of time it takes to refit the engine before inference."                     << std::endl <<
          "  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second "
                                                                                "profile run will be executed (default = disabled)"  << std::endl <<
          "  --skipInference             Exit after the engine has been built and skip inference perf measurement "
                                                                                                             "(default = disabled)"  << std::endl <<
          "  --persistentCacheRatio      Set the persistentCacheLimit in ratio, 0.5 represent half of max persistent L2 size "
              
```

| 名称  |  说明   |  备注|     
|---    | ---    | ---- |   
|setAuxStreams   |   | https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a9126b0a6f7fdff69546d374cfc9c2aaa |    
|setBindingDimensions   |   |  |    
|setDebugSync   |   |  |    
|setDeviceMemory   |   |  |    
|setEnqueueEmitsProfile   |   |  |    
|setErrorRecorder   |   |  |    
|setInputConsumedEvent   |   |  |    
|setInputShape   |   |  |    
|setInputShapeBinding   |   |  |    
|setInputTensorAddress   |   |  |    
|setName   |   |  |    
|setNvtxVerbosity   |   |  |    
|setOptimizationProfile   |   |  |    
|setOptimizationProfileAsync   |   |  |    
|setOutputAllocator   |   |  |    
|setPersistentCacheLimit   |   |  |    
|setProfiler   |   |  |    
|setTemporaryStorageAllocator   |   |  |    
|setTensorAddress   |   |  |      
