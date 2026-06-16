
https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/DRIVE_AGX_SoC/Graphics_Programming/GPU_Scheduling_Improvements/RunningtheGPUSchedulingSampleApplication56.html



# Setting the Timeslice
Use the following guidelines when setting the timeslice, depending on the priority of the application.       
根据应用程序的优先级，按照以下准则设置时间片。

## High-Priority Applications
For high priority applications, set the timeslice large enough so that all work can be completed within one timeslice.     
对于高优先级应用程序，需将**时间片设置得足够大**，确保所有`任务可在单个时间片内完成`。

The recommended upper bound for timeslice in single, high-priority applications is:
```
16.6 ms − lpt − cst – crt = 11.6 ms
```

单款高优先级应用程序的时间片建议上限计算公式如下：
```
16.6 毫秒 − 低优先级时间片 − 上下文切换超时时间 − 通道重置时间 = 11.6 毫秒
```

Where:
其中：

lpt is the low-priority timeslice, set to 1.5 ms     
lpt：低优先级时间片，取值为1.5毫秒   

cst is the context-switch timeout, equal to 2.0 ms   
cst：上下文切换超时时间，取值为2.0毫秒   

crt is the channel reset time, equal to 1.5 ms     
crt：通道重置时间，取值为1.5毫秒   

For multiple, high-priority applications, use the timeslice for each high-priority application to determine a reasonable bound. The recommended combined workload of all high-priority applications must not exceed 50% of a display refresh cycle.      
对于多款高优先级应用程序，需结合每款应用的时间片设定合理上限。所有高优先级应用的总负载，建议不得超过一个显示刷新周期的50%。

High-priority applications must avoid flushing work prematurely, whether by calling glFlush or glFinish or by other means. This ensures all rendering for a frame completes without any context switches.     
高优先级应用程序禁止提前刷新任务，包括调用`glFlush`、`glFinish`接口或使用其他方式。以此保证单帧的所有渲染工作可在不发生上下文切换的情况下完成。

## Medium-Priority Applications

For medium-priority applications, set the timeslice both:     
Large enough that an application can make progress, but Not so large that it affects the scheduling latency of high-priority applications.         
对于中优先级应用程序，时间片的设置需同时满足以下两点：
时间片大小需足以支撑应用程序正常运行；
同时不能过大，避免影响高优先级应用的调度延迟。

The recommended upper bound for timeslice in medium-priority applications is 2 ms.         
中优先级应用程序的时间片建议上限为2毫秒。

## Low-Priority Applications

For low-priority applications, set the timeslice both:
Large enough that an application can make progress, but Not so large that it affects the scheduling latency of high- or medium-priority applications.      
对于低优先级应用程序，时间片的设置需同时满足以下两点：
时间片大小需足以支撑应用程序正常运行；
同时不能过大，避免影响高、中优先级应用的调度延迟。

The recommended upper bound for timeslice in low-priority applications is 1.5 milliseconds (ms).        
低优先级应用程序的时间片建议上限为1.5毫秒。

## Reserve Time for Lower-Priority Applications

To ensure lower-priority applications make reasonable progress, you must ensure that high- and medium-priority applications do not use 100% of the GPU by:
Lowering your application frame rate targets and/or Reducing complexity of rendered frames.    
为保障低优先级应用程序正常运行，需限制高、中优先级应用不占用100%的GPU资源，可通过以下方式实现：    
降低应用的目标帧率；    
以及/或者 降低渲染帧的复杂度。   

The proportion of time to reserve for low-priority applications depends on the number and nature of the applications.      
需为低优先级应用预留的时间占比，取决于应用的数量与业务特性。


