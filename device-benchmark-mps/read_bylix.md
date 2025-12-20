
+ Starting MPS control daemon
As $UID, run the commands

```bash   
export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that’s accessible to the given $UID

export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that’s accessible to the given $UID

nvidia-cuda-mps-control -d # Start the daemon.

This will start the MPS control daemon that will spawn a new MPS Server instance for that $UID starting an application and associate it with GPU visible to the control daemon.

```


+ Starting MPS client application
Set the following variables in the client process’s environment. Note that CUDA_VISIBLE_DEVICES should not be set in the client’s environment.
```bash   
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Set to the same location as the MPS control daemon

export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Set to the same location as the MPS control daemon

```

+ Shutting Down MPS
To shut down the daemon, as $UID, run
```bash   
echo quit | nvidia-cuda-mps-control
```

https://docs.nvidia.com/deploy/topics/topic_6_1_2_2.html
https://docs.deep-hybrid-datacloud.eu/en/latest/technical/others/gpu-sharing-with-mps.html
