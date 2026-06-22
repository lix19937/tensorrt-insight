
### has timing.cache, no out/*yaml   

```
2022-01-14 20:30:37,242 - [modelopt][onnx] - INFO - Trtexec benchmark initialized
2022-01-14 20:30:37,243 - [modelopt][onnx] - INFO - Loading model: ResNet50.onnx
2022-01-14 20:30:37,312 - [modelopt][onnx] - INFO - Initializing autotuner (quant_type=int8, default_dq_dtype=float32)
2022-01-14 20:30:37,323 - [modelopt][onnx] - INFO - Initializing autotuner
2022-01-14 20:30:37,323 - [modelopt][onnx] - INFO - Discovering optimization regions
2022-01-14 20:30:37,324 - [modelopt][onnx] - INFO - Phase 1: Bottom-up partitioning
2022-01-14 20:30:37,326 - [modelopt][onnx] - INFO - Partitioning graph (176 nodes)
2022-01-14 20:30:37,327 - [modelopt][onnx] - INFO - Partitioning complete: 21 regions, 176/176 nodes (100.0%)
2022-01-14 20:30:37,327 - [modelopt][onnx] - INFO - Phase 1 complete: 21 regions, 176/176 nodes (100.0%)
2022-01-14 20:30:37,327 - [modelopt][onnx] - INFO - Phase 2: Top-down refinement
2022-01-14 20:30:37,339 - [modelopt][onnx] - INFO - Phase 2 complete: refined 16/21 regions

2022-01-14 20:30:37,339 - [modelopt][onnx] - INFO - Discovery complete: 53 regions (37 LEAF, 16 COMPOSITE, 0 ROOT)

2022-01-14 20:30:37,339 - [modelopt][onnx] - INFO - Starting new autotuning session

2022-01-14 20:30:37,339 - [modelopt][onnx] - INFO - Ready to profile 53 regions
2022-01-14 20:30:37,339 - [modelopt][onnx] - INFO - Measuring baseline (no Q/DQ)
2022-01-14 20:30:37,706 - [modelopt][onnx] - INFO - Exported baseline model with 0 Q/DQ pairs  → out_0622/baseline.onnx
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.18 ms
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - Baseline latency: 1.179 ms
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - Baseline: 1.18 ms
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - Starting region profiling (30 schemes per region)
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - Region 1/53 (ID=0, level=0)
2022-01-14 20:30:46,745 - [modelopt][onnx] - INFO - Profiling region 0 [level 0, size4, starting fresh]
2022-01-14 20:30:46,746 - [modelopt][onnx] - INFO - Scheme #1: generated new scheme (0 Q/DQ points)
2022-01-14 20:30:46,923 - [modelopt][onnx] - INFO - Exported INT8 model with 0 Q/DQ pairs 
2022-01-14 20:30:55,974 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.17 ms
2022-01-14 20:30:55,974 - [modelopt][onnx] - INFO - Scheme #1: 1.175 ms (1.00x speedup)

2022-01-14 20:30:55,975 - [modelopt][onnx] - INFO - Scheme #2: generated new scheme (2 Q/DQ points)
2022-01-14 20:30:56,241 - [modelopt][onnx] - INFO - Exported INT8 model with 4 Q/DQ pairs 
2022-01-14 20:31:05,536 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.15 ms
2022-01-14 20:31:05,536 - [modelopt][onnx] - INFO - Scheme #2: 1.151 ms (1.02x speedup)
2022-01-14 20:31:05,536 - [modelopt][onnx] - INFO -   ★ New best: 1.151 ms (1.02x speedup)

2022-01-14 20:31:05,537 - [modelopt][onnx] - INFO - Scheme #3: generated new scheme (2 Q/DQ points)
2022-01-14 20:31:05,711 - [modelopt][onnx] - INFO - Exported INT8 model with 3 Q/DQ pairs 
2022-01-14 20:31:14,879 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.15 ms
2022-01-14 20:31:14,880 - [modelopt][onnx] - INFO - Scheme #3: 1.148 ms (1.03x speedup)
2022-01-14 20:31:14,880 - [modelopt][onnx] - INFO -   ★ New best: 1.148 ms (1.03x speedup)

2022-01-14 20:31:14,880 - [modelopt][onnx] - INFO - Scheme #4: generated new scheme (1 Q/DQ points)
2022-01-14 20:31:15,055 - [modelopt][onnx] - INFO - Exported INT8 model with 2 Q/DQ pairs 
2022-01-14 20:31:24,136 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.17 ms
2022-01-14 20:31:24,136 - [modelopt][onnx] - INFO - Scheme #4: 1.173 ms (1.00x speedup)

2022-01-14 20:31:24,137 - [modelopt][onnx] - INFO - Scheme #5: generated new scheme (1 Q/DQ points)
2022-01-14 20:31:24,313 - [modelopt][onnx] - INFO - Exported INT8 model with 2 Q/DQ pairs 
2022-01-14 20:31:33,409 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.27 ms
2022-01-14 20:31:33,409 - [modelopt][onnx] - INFO - Scheme #5: 1.268 ms (0.93x speedup)

2022-01-14 20:31:33,411 - [modelopt][onnx] - INFO - Scheme #6: generated new scheme (2 Q/DQ points)
2022-01-14 20:31:33,584 - [modelopt][onnx] - INFO - Exported INT8 model with 3 Q/DQ pairs 
2022-01-14 20:31:42,586 - [modelopt][onnx] - INFO - TrtExec benchmark (median): 1.17 ms
2022-01-14 20:31:42,587 - [modelopt][onnx] - INFO - Scheme #6: 1.169 ms (1.01x speedup)

2022-01-14 20:31:42,587 - [modelopt][onnx] - INFO - Scheme #7: generated new scheme (1 Q/DQ points)
2022-01-14 20:31:42,760 - [modelopt][onnx] - INFO - Exported INT8 model with 1 Q/DQ pairs 
```
