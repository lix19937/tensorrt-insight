# Add custom task
- this guide will help you to add custom tasks to this app

- Users can ref class `CudaKernelTask` as an example
### 1. Directory structure
create a new dir in src/customtasks
- if you have cuda kernel function in your task, create a cu file in the dir, If no cuda-kernel files, you do not need it.
- take cudaKernelTask as an example:
    ```bash
    src/customtasks/
    ├── cudaKernelTask
    │   ├── cudaKernelTask.cpp
    │   ├── cudaKernelTask.cu
    │   └── cudaKernelTask.h
    ├── customTask.cpp
    ├── customTask.h
    ```
### 2. implements of the class

####  1) the class must derive from the `CustomTask`.

####  2) use Macro `DECLARE_CLASS()` inside the class

```c++
class CudaKernelTask : public CustomTask {
    DECLARE_CLASS()
public:
    ...
```
####  3) the constructor of the class 
the constructor should calls `TaskInit` first.
And then, Do the initialization of your Task in the constructor, eg. malloc buffers, setup inputs.

```c++
CudaKernelTask() = delete;
CudaKernelTask(std::string name, SyncType syncType) {
    // the same to all the task
    this->TaskInit(syncType);
    this->mTaskName = std::string("Cuda kernel task");
    ReportArgs();
    // the privite operation for the CudaKernel Task
    initMatrix(&hptrA, &hptrB, &hptrC, &dptrA, &dptrB, &dptrC, MATRIXDIM);
}
```

####  4) implements the `Run` function
the trt_bench-app will start a thread and excute `Run` function cycling forever. And print the fps to the screen. And do not forget call `Sync()` and `mInference_count++` and `return true`.
```c++
virtual bool Run() {
    // run cuda kernel here
    runMultiplyMatrix(dptrA, dptrB, dptrC, this->mStream.get(), MATRIXDIM, LOOPCOUNT);
    // the same to all the sub-class
    this->Sync();
    mInference_count++;
    return true;
}
```
####  5) distruct the sources inside the distructor
```c++
~CudaKernelTask() { FreeMatrix(hptrA, hptrB, hptrC, dptrA,dptrB, dptrC); }
```
####  6) use marco `IMPLEMENT_CLASS` in the cpp file
the first paramenter will be the task-name in the app's args
```c++
IMPLEMENT_CLASS("CudaKernelTask", CudaKernelTask);
```

