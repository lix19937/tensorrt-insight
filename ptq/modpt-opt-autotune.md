# Autotune (ONNX)

## Overview
The `modelopt.onnx.quantization.autotune` module automates Q/DQ (Quantize/Dequantize) placement optimization in ONNX models. It explores placement strategies and uses TensorRT latency measurements to choose a configuration that minimizes inference time.      

`modelopt.onnx.quantization.autotune` 模块能够自动完成 ONNX 模型中量化/反量化（Q/DQ）节点的位置优化。该工具会遍历多种节点放置方案，并结合 TensorRT 实测推理时延，选出推理耗时最短的配置。

### 软件组成   

#### regions   
```
Phase 1: Bottom-up partitioning
Partitioning graph (8692 nodes)
Partitioning complete: 1783 regions, 8492/8692 nodes (97.7%)
Phase 1 complete: 1783 regions, 8492/8692 nodes (97.7%)

Phase 2: Top-down refinement
Phase 2 complete: refined 400/1783 regions
```
#### scheme  

### Key Features:
- Automatic Region Discovery: Intelligently partitions the model into optimization regions
- Pattern-Based Optimization: Groups structurally similar regions and optimizes them together
- TensorRT Performance Measurement: Uses actual inference latency (not theoretical estimates)
- Crash Recovery: Checkpoint/resume capability for long-running optimizations
- Warm-Start Support: Reuses learned patterns from previous runs
- Multiple Quantization Types: Supports INT8 and FP8 quantization

### 核心功能：
- 自动区域识别：智能将模型划分为多个**待优化区域**
- 基于模式优化：将**结构相似的区域分组并统一优化**
- TensorRT 性能实测：采用真实推理时延评估，而非理论估算
- 故障恢复：长时间优化任务支持断点续跑
- 热启动支持：复用过往运行中得到的优化模式
- 多量化类型：支持 INT8 和 FP8 两种量化方式

### When to Use This Tool:
- Quantizing an ONNX model for TensorRT deployment
- Optimizing Q/DQ placement for best performance
- The model has repeating structures (e.g., transformer blocks, ResNet layers)

### 什么时候使用：
- 为部署到 TensorRT 而对 ONNX 模型执行量化
- 优化 Q/DQ 节点位置以实现最优推理性能
- 模型包含重复结构（如 Transformer 模块、ResNet 网络层）

------------

## Quick Start

## Command-Line Interface
使用该自动调优工具最简单的方式是通过命令行   

```bash
# Basic usage - INT8 quantization (output default: ./autotuner_output)
python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx

# Specify output dir and FP8 with more schemes
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./results \
    --quant_type fp8       \
    --schemes_per_region 50
```

该命令会执行以下操作：
1. Discover regions in the model automatically  自动识别模型中的各个优化区域
2. Measure baseline performance (no quantization) 测试未量化模型的基准性能
3. Test different Q/DQ placement schemes for each region pattern 针对每一类区域结构，测试不同的 Q/DQ 节点放置方案
4. Select the best scheme based on TensorRT latency measurements 根据 TensorRT 实测时延选择最优方案
5. Export an optimized ONNX model with Q/DQ nodes 导出嵌入 Q/DQ 节点的优化版 ONNX 模型

### Output Files:
所有文件都会保存到输出目录下（默认路径为 `./autotuner_output`，也可通过 `--output_dir` 参数指定）。

```
autotuner_output/                         # 默认目录，或 --output_dir 指定的路径
├── autotuner_state.yaml                  # 断点续跑文件
├── autotuner_state_pattern_cache.yaml    # 用于后续运行的模式缓存文件
├── baseline.onnx                         # 未量化的基准模型
├── optimized_final.onnx                  # 最终优化后的模型
├── logs/                                 # TensorRT 构建日志目录
│   ├── baseline.log
│   ├── region_*_scheme_*.log
│   └── final.log
└── region_models/                        # 每个区域对应的最优模型
    └── region_*_level_*.onnx
```

## Python API
如需通过代码进行精细化控制，请使用工作流函数

```python
from pathlib import Path
from modelopt.onnx.quantization.autotune.workflows import (
    region_pattern_autotuning_workflow,
    init_benchmark_instance
)

# When using the CLI, the benchmark is initialized automatically. When calling the
# workflow from Python, call init_benchmark_instance first:
init_benchmark_instance(
    use_trtexec=False,
    timing_cache_file="timing.cache",
    warmup_runs=5,
    timing_runs=20,
)

# Run autotuning workflow
autotuner = region_pattern_autotuning_workflow(
    model_path="model.onnx",
    output_dir=Path("./results"),
    num_schemes_per_region=30,
    quant_type="int8",
)
```

## How It Works
该自动调优工具采用基于模式的方案，让优化过程兼顾效率与稳定性。

-----

### Region Discovery Phase
The model’s computation graph is automatically partitioned into **hierarchical regions**. Each region is a subgraph containing related operations (e.g. a Conv-BatchNorm-ReLU block).
### 区域识别阶段
工具会自动将模型计算图划分为**多层级的区域**，每个区域是一个子图由一组关联算子组成（例如 `卷积-批归一化-激活函数` 组合模块）。

-----

### Pattern Identification Phase
Regions with identical structural patterns are grouped together. For example, all Convolution->BatchNormalization->ReLU blocks in the model share the same pattern.
### 模式识别阶段
将结构完全一致的区域分为一组。例如模型中所有 `卷积→批归一化→ReLU` 模块都属于同一种结构模式。

----

### Scheme Generation Phase
For each unique pattern, multiple Q/DQ insertion **schemes** are generated. Each scheme specifies different locations to insert Q/DQ nodes.
### 方案生成阶段
针对每一种独有结构模式，生成多套 Q/DQ 节点插入方案，每套方案对应不同的节点插入位置。

------

### Performance Measurement Phase
### 性能测试阶段
每套方案（scheme）的评估流程如下：
1. Exporting the ONNX model with Q/DQ nodes applied 导出嵌入 Q/DQ 节点的 ONNX 模型
2. Building a TensorRT engine 构建 TensorRT 推理引擎
3. Measuring actual inference latency 实测推理时延

-----

### Best Scheme Selection
The scheme with the lowest latency is selected for each pattern. This scheme automatically applies to all regions matching that pattern.
### 最优方案选择
为每种结构模式挑选时延最低的方案，该方案会自动应用到所有匹配该模式的区域。

----

### Model Export
The final model includes the best Q/DQ scheme for each pattern, resulting in an optimized quantized model.
### 模型导出
最终模型会整合所有结构模式的最优 Q/DQ 方案，生成经过优化的量化模型。

-----

### Why pattern-based?
The autotuner optimizes each unique pattern once; the chosen scheme then applies to every region that matches that pattern. So runtime scales with the number of patterns, not regions. Models with repeated structure (e.g. transformers) benefit most; highly diverse graphs have more patterns and take longer.
### 为何采用基于模式的设计？
调优器仅对每一种独有结构模式执行一次优化，选中的方案会复用至所有同结构区域。因此运行耗时取决于**模式数量**，而**非区域总数**。该方式对 Transformer 这类存在大量重复结构的模型提升效果最显著；如果模型计算图结构繁杂、独有模式较多，优化耗时也会相应增加。

----

## Advanced Usage
### Warm-Start with Pattern Cache
Pattern cache files store the best Q/DQ schemes from previous optimization runs. These patterns can be reused on similar models or model versions:
### 借助模式缓存实现热启动
模式缓存文件会保存过往优化任务中得到的最优 Q/DQ 方案，可复用到同类型模型或模型迭代版本中。

```bash
# First optimization (cold start)
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model_v1.onnx                 \
    --output_dir ./run1

# The pattern cache is saved to ./run1/autotuner_state_pattern_cache.yaml
# Second optimization with warm-start
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model_v2.onnx \
    --output_dir ./run2       \
    --pattern_cache ./run1/autotuner_state_pattern_cache.yaml
```

The second run tests cached schemes first and can reach a good configuration faster.
第二次运行会优先测试缓存中的方案，能够更快得到优质配置。

----

#### When to use pattern cache:
- Optimizing multiple versions of the same model
- Optimizing models from the same family (e.g., different BERT variants)
- Transferring learned patterns across models
#### 模式缓存适用场景：
- 对同一模型的多个迭代版本进行优化
- 优化同系列模型（例如不同版本的 BERT 模型）
- 在不同模型之间复用已习得的优化模式

----

### Import Patterns from Existing QDQ Models
With a pre-quantized baseline model (e.g., from manual optimization or another tool), its Q/DQ patterns can be imported:
### 从已有 Q/DQ 模型导入优化模式
如果已有经过手动调优或其他工具量化的基准模型，可以直接导入其 Q/DQ 配置方案。

```bash
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx                    \
    --output_dir ./results                    \
    --qdq_baseline manually_quantized.onnx
```

The workflow extracts Q/DQ insertion points from the baseline, maps them to region patterns, and uses them as seed schemes. Useful when:
该流程会提取基准模型中的 Q/DQ 插入位置，将其映射到当前模型的区域结构中，并作为初始候选方案。适用于以下场景：
- Starting from expert-tuned quantization schemes 基于专家调试好的量化方案继续优化
- Comparing against reference implementations 与参考实现进行性能对比
- Fine-tuning existing quantized models 对已有量化模型做精细化调优

---

### Resume After Interruption
A long run can be interrupted (Ctrl+C, preemption, or crash) and resumed later:
### 中断后恢复运行
长时间运行的任务若被手动终止（Ctrl+C）、资源抢占或程序崩溃，支持后续断点续跑。

```bash
# Start optimization
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./results

# ... interrupted after 2 hours ...
# Resume from checkpoint (just run the same command)
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./results
```

When rerun with the same `--output_dir`, the autotuner detects `autotuner_state.yaml`, restores progress, and continues from the next unprofiled region.
使用相同的 `--output_dir` 重新执行命令时，调优器会读取 `autotuner_state.yaml` 文件，恢复运行进度，从下一个未测试的区域继续执行。

----

### Custom TensorRT Plugins
If the model uses custom TensorRT operations, provide the plugin libraries:
### 自定义 TensorRT 插件
如果模型使用了自定义 TensorRT 算子，需要指定插件库文件路径。

```bash
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./results \
    --plugin_libraries /path/to/plugin1.so /path/to/plugin2.so
```

----

### Remote Autotuning
TensorRT 10.15+ supports remote autotuning in safety mode (`--safe`), which allows TensorRT’s optimization process to be offloaded to a remote hardware. This is useful when optimizing models for different target GPUs without having direct access to them.
### 远程自动调优
TensorRT 10.15 及以上版本支持安全模式（`--safe`）下的远程调优，可将 TensorRT 优化任务分发到远端硬件执行。当需要为无法本地访问的目标 GPU 优化模型时，该功能十分实用。

To use remote autotuning during Q/DQ placement optimization, run with `trtexec` and pass extra args:
在 Q/DQ 节点位置优化过程中使用远程调优，需搭配 `trtexec` 并传入额外参数。

```bash
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx \
    --output_dir ./model_remote_autotuned \
    --schemes_per_region 50 \
    --use_trtexec \
    --trtexec_benchmark_args "--remoteAutoTuningConfig=\"<remote autotuning config>\" --safe --skipInference"
```

----

#### Requirements:
1. TensorRT 10.15 or later
2. Valid remote autotuning configuration
3. `--use_trtexec` must be set (benchmarking uses `trtexec` instead of the TensorRT Python API)
4. `--safe --skipInference` must be enabled via `--trtexec_benchmark_args`
#### 环境与参数要求：
1. TensorRT 版本为 10.15 或更高
2. 拥有合法的远程调优配置
3. 必须开启 `--use_trtexec`（使用 trtexec 而非 TensorRT Python 接口执行基准测试）
4. 必须在 `--trtexec_benchmark_args` 中启用 `--safe --skipInference`

Replace `<remote autotuning config>` with an actual remote autotuning string (see `trtexec --help` for more details). Other TensorRT benchmark options (e.g. `--timing_cache`, `--warmup_runs`, `--timing_runs`, `--plugin_libraries`) are also available; run `--help` for details.     
将 `<remote autotuning config>` 替换为实际的远程调优配置字符串（详情可查看 `trtexec --help`）。工具同样支持其他 TensorRT 基准测试参数（如 `--timing_cache`、`--warmup_runs` 等），执行 `--help` 可查看完整说明。

---

## Low-Level API Usage
For fine-grained control over the autotune process (e.g. driving it step-by-step or customizing regions and schemes), use the autotuner classes directly:
## 底层接口使用
如需对调优流程进行细粒度控制（例如分步执行、自定义区域与优化方案），可以直接调用调优器类。

### Basic Workflow
```python
import onnx
from modelopt.onnx.quantization.autotune import QDQAutotuner, Config
from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    benchmark_onnx_model,
)

# Initialize global benchmark (required before benchmark_onnx_model)
init_benchmark_instance(
    use_trtexec=False,
    timing_cache_file="timing.cache",
    warmup_runs=5,
    timing_runs=20,
)

# Load model
model = onnx.load("model.onnx")

# Initialize autotuner with automatic region discovery
autotuner = QDQAutotuner(model)
config = Config(default_quant_type="int8", verbose=True)
autotuner.initialize(config)

# Measure baseline (no Q/DQ)
autotuner.export_onnx("baseline.onnx", insert_qdq=False)
baseline_latency = benchmark_onnx_model("baseline.onnx")
autotuner.submit(baseline_latency)
print(f"Baseline: {baseline_latency:.2f} ms")

# Profile each region
regions = autotuner.regions
print(f"Found {len(regions)} regions to optimize")
for region_idx, region in enumerate(regions):
    print(f"\nRegion {region_idx + 1}/{len(regions)}")

    # Set current profile region
    autotuner.set_profile_region(region, commit=(region_idx > 0))

    # After set_profile_region(), None means this region's pattern was already
    # profiled (e.g. from a loaded state file). There are no new schemes to
    # generate, so skip to the next region.
    if autotuner.current_profile_pattern_schemes is None:
        print("  Already profiled, skipping")
        continue


    # Generate and test schemes
    for scheme_num in range(30):  # Test 30 schemes per region
        scheme_idx = autotuner.generate()

        if scheme_idx == -1:
            print(f"  No more unique schemes after {scheme_num}")
            break


        # Export model with Q/DQ nodes
        model_bytes = autotuner.export_onnx(None, insert_qdq=True)

        # Measure performance
        latency = benchmark_onnx_model(model_bytes)
        success = latency != float('inf')
        autotuner.submit(latency, success=success)

        if success:
            speedup = baseline_latency / latency
            print(f"  Scheme {scheme_idx}: {latency:.2f} ms ({speedup:.3f}x)")

    # Best scheme is automatically selected
    ps = autotuner.current_profile_pattern_schemes
    if ps and ps.best_scheme:
        print(f"  Best: {ps.best_scheme.latency_ms:.2f} ms")

# Commit final region
autotuner.set_profile_region(None, commit=True)

# Export optimized model
autotuner.export_onnx("optimized_final.onnx", insert_qdq=True)
print("\nOptimization complete!")
```

----

### 状态管理
保存和加载优化状态，实现故障恢复。

```python
# Save state after each region
autotuner.save_state("autotuner_state.yaml")

# Load state to resume
autotuner = QDQAutotuner(model)
autotuner.initialize(config)
autotuner.load_state("autotuner_state.yaml")

# Continue optimization from last checkpoint
# (regions already profiled will be skipped)
```

---

### Pattern Cache Management
Create and use pattern caches     
### 模式缓存管理
创建并使用模式缓存。

```python
from modelopt.onnx.quantization.autotune import PatternCache

# Load existing cache
cache = PatternCache.load("autotuner_state_pattern_cache.yaml")
print(f"Loaded {cache.num_patterns} patterns")

# Initialize autotuner with cache
autotuner = QDQAutotuner(model)
autotuner.initialize(config, pattern_cache=cache)

# After optimization, pattern cache is automatically saved
# when save_state() is called
autotuner.save_state("autotuner_state.yaml")
# This also saves: autotuner_state_pattern_cache.yaml
```

----

### Import from a Q/DQ Baseline
To seed the autotuner from a pre-quantized model (e.g. from another tool or manual tuning), extract quantized tensor names and pass them in:
### 从 Q/DQ 基准模型导入
想要基于已有量化模型（其他工具生成或手动调优）初始化调优器，可先提取量化张量名称并传入。

```python
import onnx
from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

# Load baseline model with Q/DQ nodes
baseline_model = onnx.load("quantized_baseline.onnx")

# Extract quantized tensor names
quantized_tensors = get_quantized_tensors(baseline_model)
print(f"Found {len(quantized_tensors)} quantized tensors")

# Import into autotuner
autotuner = QDQAutotuner(model)
autotuner.initialize(config)
autotuner.import_insertion_points(quantized_tensors)

# These patterns will be tested first during optimization
```

----

## Configuration Options
### Config Class
The `Config` class controls autotuner behavior:
## 配置项
### Config 配置类
`Config` 类用于控制调优器的运行行为。

```python
from modelopt.onnx.quantization.autotune import Config

config = Config(
    default_quant_type="int8",             # "int8" or "fp8"
    default_dq_dtype="float32",            # float16, float32, bfloat16 (bfloat16 needs NumPy with np.bfloat16)
    default_q_scale=0.1,
    default_q_zero_point=0,
    top_percent_to_mutate=0.1,
    minimum_schemes_to_mutate=10,
    maximum_mutations=3,
    maximum_generation_attempts=100,
    pattern_cache_minimum_distance=4,
    pattern_cache_max_entries_per_pattern=32,
    maximum_sequence_region_size=10,
    minimum_topdown_search_size=10,
    verbose=True,
)
```

----

### Command-Line Arguments
Arguments use underscores. Short options: `-m` (onnx_path), `-o` (output_dir), `-s` (schemes_per_region), `-v` (verbose). Run `python -m modelopt.onnx.quantization.autotune --help` for full help.

```
usage: python -m modelopt.onnx.quantization.autotune [-h] --onnx_path
                                                     ONNX_PATH
                                                     [--output_dir OUTPUT_DIR]
                                                     [--mode {quick,default,extensive}]
                                                     [--schemes_per_region NUM_SCHEMES]
                                                     [--pattern_cache PATTERN_CACHE_FILE]
                                                     [--qdq_baseline QDQ_BASELINE]
                                                     [--state_file STATE_FILE]
                                                     [--node_filter_list NODE_FILTER_LIST]
                                                     [--quant_type {int8,fp8}]
                                                     [--default_dq_dtype {float16,float32,bfloat16}]
                                                     [--use_trtexec]
                                                     [--timing_cache TIMING_CACHE]
                                                     [--warmup_runs WARMUP_RUNS]
                                                     [--timing_runs TIMING_RUNS]
                                                     [--plugin_libraries PLUGIN_LIBRARIES [PLUGIN_LIBRARIES ...]]
                                                     [--trtexec_benchmark_args TRTEXEC_BENCHMARK_ARGS]
                                                     [--verbose]
```

----

### Named Arguments
#### --verbose, -v
Enable verbose DEBUG logging
Default: `False`

#### Model and Output
##### --onnx_path, -m
Path to ONNX model file
##### --output_dir, -o
Output directory for results (default: ./autotuner_output)
Default: `'./autotuner_output'`

#### Autotuning Strategy
##### --mode
Possible choices: quick, default, extensive
Preset for schemes_per_region, warmup_runs, and timing_runs. ‘quick’: fewer schemes/runs for fast iteration; ‘default’: balanced; ‘extensive’: more schemes/runs for thorough tuning. Explicit –schemes_per_region, –warmup_runs, –timing_runs override the preset.
Default: `'default'`

##### --schemes_per_region, -s
Schemes per region (default: 50; preset from –mode if not set)
Default: `50`

##### --pattern_cache
Path to pattern cache YAML for warm-start (optional)

##### --qdq_baseline
Path to QDQ baseline ONNX model to import quantization patterns (optional)

##### --state_file
State file path for resume capability (default: <output_dir>/autotuner_state.yaml)

##### --node_filter_list
Path to a file containing wildcard patterns to filter ONNX nodes (one pattern per line). Regions without any matching nodes are skipped during autotuning.

#### 调优策略
##### --mode
可选值：quick、default、extensive
用于预设单区域方案数、预热次数、测速次数。  
quick：减少方案与运行次数，快速迭代；  
default：均衡配置；  
extensive：增加方案与运行次数，深度调优。  
手动指定 --schemes_per_region、--warmup_runs、--timing_runs 会覆盖该预设。
默认值：`'default'`

##### --schemes_per_region, -s
单个区域的测试方案数量（默认 50；未手动指定时，由 --mode 预设决定）
默认值：`50`

##### --pattern_cache
热启动所用的模式缓存 YAML 文件路径（选填）

##### --qdq_baseline
用于导入量化方案的 Q/DQ 基准 ONNX 模型路径（选填）

##### --state_file
断点文件路径（默认：<output_dir>/autotuner_state.yaml）

##### --node_filter_list
节点过滤规则文件路径，文件内每行一条通配符规则。调优过程中，不含匹配节点的区域会被跳过。

#### Quantization
##### --quant_type
Possible choices: int8, fp8
Quantization data type (default: int8)
Default: `'int8'`

##### --default_dq_dtype
Possible choices: float16, float32, bfloat16
Default DQ output dtype if cannot be deduced (optional)
Default: `'float32'`

#### 量化配置
##### --quant_type
可选值：int8、fp8
量化数据类型（默认 int8）
默认值：`'int8'`

##### --default_dq_dtype
可选值：float16、float32、bfloat16
无法自动推断时，反量化节点使用的默认数据类型（选填）
默认值：`'float32'`

#### TensorRT Benchmark
##### --use_trtexec
Use trtexec for benchmarking (default: False)
Default: `False`

##### --timing_cache
TensorRT timing cache file (default: /tmp/trtexec_timing.cache)
Default: `'/tmp/trtexec_timing.cache'`

##### --warmup_runs
Number of warmup runs (default: 50; preset from –mode applies if not set)
Default: `50`

##### --timing_runs
Number of timing runs (default: 100; preset from –mode applies if not set)
Default: `100`

##### --plugin_libraries, --plugins
TensorRT plugin libraries (.so files) to load (optional, space-separated)

##### --trtexec_benchmark_args
Additional command-line arguments to pass to trtexec as a single quoted string. Example: –trtexec_benchmark_args ‘–fp16 –workspace=4096 –verbose’

#### TensorRT 基准测试
##### --use_trtexec
使用 trtexec 执行基准测试（默认关闭）
默认值：`False`

##### --timing_cache
TensorRT 测速缓存文件（默认：/tmp/trtexec_timing.cache）
默认值：`'/tmp/trtexec_timing.cache'`

##### --warmup_runs
预热运行次数（默认 50；未手动指定时，由 --mode 预设决定）
默认值：`50`

##### --timing_runs
正式测速运行次数（默认 100；未手动指定时，由 --mode 预设决定）
默认值：`100`

##### --plugin_libraries, --plugins
需要加载的 TensorRT 插件库（.so 文件），多个文件以空格分隔（选填）

##### --trtexec_benchmark_args
传递给 trtexec 的额外命令行参数，整体作为一个带引号的字符串。示例：–trtexec_benchmark_args ‘–fp16 –workspace=4096 –verbose’

----

## Best Practices
### Choosing Scheme Count
The `--schemes_per_region` (or `-s`) parameter controls exploration depth. Typical values:
15–30 schemes (e.g. `-s 30`): Quick exploration; good for trying the tool or small models
50 schemes (default, `-s 50`): Default; Recommended for most cases
100–200+ schemes (e.g. `-s 200`): Extensive search; consider using a pattern cache to avoid re-exploring

Use fewer schemes when there are many small regions or limited time; use more for large or critical regions.

## 最佳实践
### 选择方案数量
`--schemes_per_region`（简写 `-s`）参数用于控制搜索深度，常用取值参考：
15–30 个方案（例如 `-s 30`）：快速探索，适合初次试用工具或小型模型
50 个方案   （默认，`-s 50`）：通用默认值，推荐大多数场景使用
100–200 及以上方案（例如 `-s 200`）：深度搜索，建议搭配模式缓存避免重复计算

如果模型存在大量小区域、或是时间有限，请减少方案数；针对大型核心区域，可增加方案数。

### Managing Optimization Time
Optimization time depends on:
Number of unique patterns (not total regions)
Schemes per region
TensorRT engine build time (model complexity)

Time Estimation Formula:
Total time ≈ (m unique patterns) × (n schemes per region) × (t seconds per benchmark) + baseline measurement

Where: - m = number of unique region patterns in the model - n = schemes per region (e.g., 30) - t = average benchmark time (typically 3-10 seconds, depends on model size)

Example Calculations:
Assuming t = 5 seconds per benchmark:
Small model: 10 patterns × 30 schemes × 5s = 25 minutes
Medium model: 50 patterns × 30 schemes × 5s = 2.1 hours
Large model: 100 patterns × 30 schemes × 5s = 4.2 hours

Note: Actual benchmark times may depend on TensorRT engine build complexity and GPU hardware.

Ways to reduce time: Use a pattern cache from a similar model (warm-start), use fewer schemes per region for initial runs, or rely on checkpoint/resume to split work across sessions.

### 优化耗时管控
优化耗时由以下因素决定：
独有结构模式的数量（而非区域总数）
单区域测试方案数
TensorRT 引擎构建耗时（取决于模型复杂度）

#### 耗时估算公式
总耗时 ≈（独有模式数 m）×（单区域方案数 n）×（单次测速耗时 t）+ 基准测试耗时

参数说明：m = 模型中独有区域模式数量；n = 单区域方案数（如 30）；t = 单次测速平均耗时（通常 3-10 秒，由模型大小决定）

#### 计算示例
假设单次测速耗时 t = 5 秒：
小型模型：10 种模式 × 30 套方案 × 5 秒 = 25 分钟
中型模型：50 种模式 × 30 套方案 × 5 秒 = 2.1 小时
大型模型：100 种模式 × 30 套方案 × 5 秒 = 4.2 小时

注：实际测速耗时还会受 TensorRT 引擎构建复杂度、GPU 硬件影响。

#### 缩短耗时的方法
使用同类型模型的模式缓存实现热启动；初次调试时减少单区域方案数；利用断点续跑功能，将长任务拆分到多个时段执行。

### Using the Pattern Cache Effectively
The pattern cache helps most when models share structure (e.g. BERT → RoBERTa), when iterating on the same model (v1 → v2), or when optimizing a family of models.

Example: building a pattern library

### 高效使用模式缓存
当模型结构相近（如 BERT 与 RoBERTa）、对同一模型迭代优化（v1 到 v2）、优化同系列模型时，模式缓存的效果最为显著。

#### 示例：搭建模式资源库
```bash
# Optimize first model and save patterns
python -m modelopt.onnx.quantization.autotune \
     --onnx_path bert_base.onnx \
     --output_dir ./bert_base_run \
     --schemes_per_region 50

# Use patterns for similar models
python -m modelopt.onnx.quantization.autotune \
     --onnx_path bert_large.onnx \
     --output_dir ./bert_large_run \
     --pattern_cache ./bert_base_run/autotuner_state_pattern_cache.yaml

python -m modelopt.onnx.quantization.autotune \
     --onnx_path roberta_base.onnx \
     --output_dir ./roberta_run \
     --pattern_cache ./bert_base_run/autotuner_state_pattern_cache.yaml
```

### Interpreting Results
The autotuner reports speedup ratios:
Baseline: 12.50 ms
Final: 9.80 ms (1.276x speedup)

What the speedup ratio means: Baseline ÷ final latency (e.g. 1.276x = final is about 22% faster than baseline).

If speedup is low (<1.1x):
Model may already be memory-bound (not compute-bound)
Q/DQ overhead dominates small operations
TensorRT may not fully exploit quantization for this architecture
Try FP8 instead of INT8

### 结果解读
调优器会输出加速比，示例如下：
基准时延：12.50 毫秒
优化后时延：9.80 毫秒（加速比 1.276 倍）

加速比计算方式：基准时延 ÷ 优化后时延。以上例为例，1.276 倍代表优化后模型速度相比原模型提升约 22%。

若加速比偏低（小于 1.1 倍），常见原因：
模型属于访存密集型，而非计算密集型
小规模算子中，Q/DQ 节点带来的开销占主导
TensorRT 无法在该模型架构上充分发挥量化优势
建议将量化类型从 INT8 更换为 FP8。

## Deploying Optimized Models
The optimized ONNX model includes Q/DQ nodes and can be used with TensorRT as follows.
## 部署优化后的模型
优化后的 ONNX 模型已嵌入 Q/DQ 节点，可按照以下方式配合 TensorRT 使用。

----

### Using Trtexec
```bash
# Build TensorRT engine from optimized ONNX
trtexec --onnx=optimized_final.onnx \
         --saveEngine=model.engine \
        --stronglyTyped

# Run inference
trtexec --loadEngine=model.engine
```

### Using TensorRT Python API
```python
import tensorrt as trt
import numpy as np

# Create builder and logger
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
      1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    | 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
)
parser = trt.OnnxParser(network, logger)

# Parse optimized ONNX model
with open("optimized_final.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX")

# Build engine
config = builder.create_builder_config()
engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError("TensorRT engine build failed")

# Save engine
with open("model.engine", "wb") as f:
    f.write(engine)

print("TensorRT engine built successfully!")
```

## Troubleshooting
### Common Issues
#### Issue: “Benchmark instance not initialized”  
```
# Solution: Initialize benchmark before running workflow
from modelopt.onnx.quantization.autotune.workflows import init_benchmark_instance
init_benchmark_instance()
```

## 问题排查
### 常见问题
#### 问题：“Benchmark instance not initialized”（基准环境未初始化）  
```
# 解决方案：运行工作流前初始化基准环境
from modelopt.onnx.quantization.autotune.workflows import init_benchmark_instance
init_benchmark_instance()
```
#### Issue: All schemes show inf latency    
Possible causes:  
```
TensorRT cannot parse the ONNX model
Model contains unsupported operations
Missing custom plugin libraries
cuda-python package not installed when using TensorRTPyBenchmark

# Solution: Check TensorRT logs in output_dir/logs/
# Add plugins if needed
python -m modelopt.onnx.quantization.autotune \
     --onnx_path model.onnx \
    --plugin_libraries /path/to/plugin.so
```

#### 问题：所有方案的时延均显示为无穷大
可能原因：   
```
TensorRT 无法解析该 ONNX 模型
模型包含不支持的算子
缺失自定义插件库
使用 TensorRTPyBenchmark 时未安装 cuda-python 包

# 解决方案：查看 output_dir/logs/ 下的 TensorRT 日志
# 如有需要，补充插件路径
python -m modelopt.onnx.quantization.autotune \
     --onnx_path model.onnx \
    --plugin_libraries /path/to/plugin.so
```

#### Issue: Optimization is very slow
Check number of unique patterns (shown at start)
Reduce schemes per region for faster exploration
Use pattern cache from similar models
```
# Faster exploration with fewer schemes
python -m modelopt.onnx.quantization.autotune \
     --onnx_path model.onnx \
     --schemes_per_region 15  
```     
#### 问题：优化过程速度极慢
排查方向：查看启动日志中的独有模式数量
减少单区域方案数以加快搜索速度
使用同类型模型的模式缓存
```
# 减少方案数，快速探索
python -m modelopt.onnx.quantization.autotune \
     --onnx_path model.onnx \
     --schemes_per_region 15
```
#### Issue: Out of GPU memory during optimization
TensorRT engine building is GPU memory intensive:
Close other GPU processes
Use smaller batch size in ONNX model if applicable
Run optimization on a GPU with more memory
#### 问题：优化过程中出现 GPU 显存溢出
TensorRT 引擎构建会占用大量 GPU 显存，解决办法：
+ 关闭其他占用 GPU 资源的进程    
+ 条件允许时，调小 ONNX 模型的批次大小    
+ 使用显存更大的 GPU 执行优化    

#### Issue: Final speedup is negative (slowdown)
The model may not benefit from quantization:
Try FP8 instead of INT8
Check if model is memory-bound (not compute-bound)
Verify TensorRT can optimize the quantized operations
#### 问题：优化后速度反而下降（加速比为负）
该模型可能无法从量化中获得收益：
+ 尝试将 INT8 量化替换为 FP8   
+ 确认模型是否为访存密集型（非计算密集型）   
+ 检查 TensorRT 是否能够正常优化量化算子    

#### Issue: Resume doesn’t work after interruption
Use the same `--output_dir` (and `--onnx_path`) as the original run
Confirm `autotuner_state.yaml` exists in that directory
If the state file is corrupted, remove it and start over
#### 问题：任务中断后无法续跑
续跑时必须使用和原任务一致的 `--output_dir` 与 `--onnx_path`
确认目录中存在 `autotuner_state.yaml` 文件
如果断点文件损坏，删除文件后重新开始优化

### Debugging
Enable verbose logging to see detailed information:
### 调试方法
开启详细日志查看完整信息：
```bash
python -m modelopt.onnx.quantization.autotune \
     --onnx_path model.onnx \
    --verbose
```

Check TensorRT build logs for each scheme (under the output directory, default `./autotuner_output`):
```bash 
# Logs are saved per scheme (replace autotuner_output with your --output_dir if different)
ls ./autotuner_output/logs/
# baseline.log
# region_0_scheme_0.log
# region_0_scheme_1.log
# ...

# View a specific log
cat ./autotuner_output/logs/region_0_scheme_0.log
查看每个方案对应的 TensorRT 构建日志（日志位于输出目录下，默认 `./autotuner_output`）：
# 每个方案都会生成独立日志（如需更换目录，请将 autotuner_output 替换为自定义的 --output_dir）
ls ./autotuner_output/logs/
# baseline.log
# region_0_scheme_0.log
# region_0_scheme_1.log
# ...

# 查看指定日志文件
cat ./autotuner_output/logs/region_0_scheme_0.log
```

### Inspect Region Discovery
To understand how the autotuner partitions the model into regions, use the region inspection tool:
### 查看区域划分结果
如需了解调优器对模型的区域划分逻辑，可以使用区域查看工具。

```bash
# Basic inspection - shows region hierarchy and statistics
python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx

# Verbose mode for detailed debug information
python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --verbose

# Custom maximum sequence size (default: 10)
python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --max-sequence-size 20

# Include all regions (even without quantizable operations)
python -m modelopt.onnx.quantization.autotune.region_inspect --model model.onnx --include-all-regions
```

What this tool shows:
Region hierarchy: How the model is partitioned into LEAF and COMPOSITE regions
Region types: Convergence patterns (divergence→branches→convergence) vs sequences
Node counts: Number of operations in each region
Input/output tensors: Data flow boundaries for each region
Coverage statistics: Percentage of nodes in the model covered by regions
Size distribution: Histogram showing region sizes

When to use:
Before optimization: Understand how many unique patterns to expect
Slow optimization: Check if model has too many unique patterns
Debugging: Verify region discovery is working correctly
Model analysis: Understand computational structure
该工具输出内容：
+ 区域层级：模型被划分为叶子区域与组合区域的结构
+ 区域类型：分支收敛结构、串行序列结构
+ 算子数量：每个区域内的算子个数
+ 输入/输出张量：每个区域的数据流边界
+ 覆盖率统计：被区域覆盖的模型算子占比
+ 尺寸分布：区域大小分布统计

使用场景：
+ 优化前：预估模型的独有模式数量
+ 优化缓慢：排查模型是否存在过多独有模式
+ 调试：验证区域划分功能是否正常
+ 模型分析：梳理模型的计算结构

------

#### Example output  
```
Phase 1 complete: 45 regions, 312/312 nodes (100.0%)
Phase 2 complete: refined 40 regions, skipped 5
Summary: 85 regions (80 LEAF, 5 COMPOSITE), 312/312 nodes (100.0%)
LEAF region sizes: min=1, max=15, avg=3.9

├─ Region 0 (Level 0, Type: COMPOSITE)
│  ├─ Direct nodes: 0
│  ├─ Total nodes (recursive): 28
│  ├─ Children: 4
│  ├─ Inputs: 3 tensors
│  └─ Outputs: 2 tensors
│    ├─ Region 1 (Level 1, Type: LEAF)
│    │  ├─ Direct nodes: 5
│    │  ├─ Nodes: Conv, BatchNormalization, Relu
│    ...

Use this to see how many unique patterns to expect (more patterns → longer optimization), whether region sizes need tuning (e.g. `--max-sequence-size` in region_inspect), and where branches or skip connections appear.
```

#### 输出示例
阶段1完成：共45个区域，覆盖 312/312 个算子（覆盖率 100.0%）
阶段2完成：优化 40 个区域，跳过 5 个
汇总：总计 85 个区域（80 个叶子区域，5 个组合区域），覆盖 312/312 个算子（覆盖率 100.0%）
叶子区域大小：最小值=1，最大值=15，平均值=3.9
```
├─ 区域 0 (层级 0, 类型: 组合区域)
│  ├─ 直属算子数: 0
│  ├─ 递归总算子数: 28
│  ├─ 子区域数量: 4
│  ├─ 输入张量: 3 个
│  └─ 输出张量: 2 个
│    ├─ 区域 1 (层级 1, 类型: 叶子区域)
│    │  ├─ 直属算子数: 5
│    │  ├─ 包含算子: 卷积、批归一化、ReLU
│    ...
```

通过该工具可以预判独有模式数量（模式越多，优化耗时越长）、判断是否需要调整区域尺寸参数（如 region_inspect 中的 `--max-sequence-size`），以及定位模型中的分支、跳跃连接等结构。

## Architecture and Workflow
The autotuner partitions the ONNX graph into regions, groups regions by structural pattern, and for each pattern tests multiple Q/DQ insertion schemes via TensorRT benchmarking. The following diagram summarizes the end-to-end process:
## 架构与工作流程
调优器会将 ONNX 计算图划分为多个区域，按照结构模式分组，并针对每种模式通过 TensorRT 基准测试验证多套 Q/DQ 插入方案。完整流程如下：

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Model Loading & Initialization                           │
│    • Load ONNX model                                        │
│    • Create QDQAutotuner instance                           │
│    • Run automatic region discovery                         │
│    • Load pattern cache (warm-start)                        │
│    • Import patterns from QDQ baseline (optional)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Baseline Measurement                                     │
│    • Export model without Q/DQ nodes                        │
│    • Build TensorRT engine                                  │
│    • Measure baseline latency                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Pattern-Based Region Profiling                           │
│    For each region: set profile region, generate schemes,   │
│    benchmark each scheme, commit best, save state           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Finalization                                             │
│    • Export optimized model with all best schemes           │
│    • Save state and pattern cache                           │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 模型加载与初始化                                          │
│    • 加载 ONNX 模型                                         │
│    • 创建 QDQAutotuner 实例                                 │
│    • 执行自动区域识别                                        │
│    • 加载模式缓存（热启动）                                   │
│    • 从 Q/DQ 基准模型导入模式（可选）                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 基准性能测试                                             │
│    • 导出不含 Q/DQ 节点的模型                                │
│    • 构建 TensorRT 引擎                                    │
│    • 测试基准推理时延                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 基于模式的区域测试                                          │
│    遍历所有区域：指定测试区域、生成方案、测速、保存最优解、记录状态    |
|                                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 收尾导出                                                 │
│    • 整合所有最优方案，导出优化模型                         │
│    • 保存运行状态与模式缓存                                 │
└─────────────────────────────────────────────────────────────┘
```

------

## Design Rationale
Pattern-based: One optimization per pattern; the chosen scheme applies to every matching region, reducing work and keeping behavior consistent.
Hierarchical regions: LEAF (single ops or short sequences) and COMPOSITE (nested subgraphs) allow tuning at different granularities.
Incremental state: Progress is saved after each region so runs can be resumed after interruption.
## 设计思路
+ 基于模式优化：每种结构模式仅优化一次，选中的方案应用到所有匹配区域，减少重复工作并保证行为一致。
+ 分层区域设计：分为叶子区域（单个算子或短算子序列）和组合区域（嵌套子图），支持不同粒度的调优。
+ 增量状态保存：每个区域优化完成后都会保存进度，支持任务中断后续跑。

## Limitations and Future Work
Current limitations:
Random scheme sampling may miss optimal configurations; number of schemes per region is fixed.
Structural similarity is assumed to imply similar performance; context (input/output) can vary.
Uniform quantization per scheme (no mixed-precision within a scheme).
TensorRT engine build time dominates; each scheme requires a full engine build.
Performance is measured with default/dummy inputs and may not generalize to all distributions.

Possible future enhancements:
Advanced search (e.g. Bayesian optimization, evolutionary algorithms).
Mixed-precision and per-layer bit-width.
Accuracy constraints and multi-objective (latency + accuracy) optimization.
## 局限性与未来规划
### 当前局限
+ 方案采用随机采样，**可能错过最优配置**；单区域测试方案数量为固定值。      
+ 默认结构相似则性能相近，但区域的输入、输出上下文可能存在差异。
+ 每套方案内使用统一量化精度，不支持方案内部混合精度。
+ TensorRT 引擎构建耗时占比较高，每一套方案都需要完整构建引擎。
+ 性能基于默认/测试输入测算，不一定能适配所有真实数据分布。

### 未来优化方向
+ 引入贝叶斯优化、进化算法等高级搜索策略。    
+ 支持混合精度与分层位宽设置。    
+ 增加精度约束，实现时延+精度的多目标优化。     

## Glossary
### Q/DQ Nodes
QuantizeLinear (Q) and DequantizeLinear (DQ) nodes in ONNX that convert between floating-point and quantized integer representations.  

## 术语表   
### Q/DQ 节点
ONNX 中的 QuantizeLinear（量化）与 DequantizeLinear（反量化）节点，用于完成浮点数据与量化整型数据的相互转换。

### Region
A hierarchical subgraph in an ONNX computation graph with well-defined input and output boundaries. Can be `LEAF (atomic)`, COMPOSITE (containing child regions), or ROOT.

### Pattern  
A structural signature of a region. Regions with identical patterns can share insertion schemes.

### Insertion Scheme  
A collection of insertion points specifying where to insert Q/DQ nodes within a region. Schemes use pattern-relative addressing for portability.

### Pattern Cache  
Collection of top-performing insertion schemes for multiple patterns, used to warm-start optimization on similar models.

### Baseline Latency  
Inference latency of the model without any Q/DQ nodes, used as reference for speedup.

### TensorRT Timing Cache   
Persistent cache of kernel performance measurements used by TensorRT to speed up engine builds.

------------

## Frequently Asked Questions    

Q: How long does optimization take?   

A: `Time ≈ (unique patterns) × (schemes per region) × (time per benchmark)`. See Managing Optimization Time for a formula and examples. Use a pattern cache when re-running on similar models to reduce time.

Q: Can I stop optimization early?    

A: Yes. Press `Ctrl+C` to interrupt. Progress is saved and the run can be resumed later.

Q: Do I need calibration data?   

A: No, the **autotuner** focuses on Q/DQ placement optimization, not calibration. Calibration scales are added when the Q/DQ nodes are inserted. **For best accuracy, run calibration separately after optimization**.

Q: Can I use this with PyTorch models?    

A: Export the PyTorch model to ONNX first using `torch.onnx.export()`, then run the autotuner on the ONNX model.

Q: What’s the difference from `modelopt.onnx.quantization.quantize()`?   

A: `quantize()` is a fast PTQ tool that uses heuristics for Q/DQ placement. The autotuner uses TensorRT measurements to optimize placement for best performance. Use `quantize()` for quick results, `autotuner` for maximum performance.

Q: Can I **customize region** discovery?   

A: Yes. Subclass QDQAutotunerBase and supply custom regions instead of using automatic discovery:

```py   
from modelopt.onnx.quantization.autotune import QDQAutotunerBase, Region

class CustomAutotuner(QDQAutotunerBase):
    def __init__(self, model, custom_regions):
        super().__init__(model)
        self.regions = custom_regions  # Custom regions   
```     

Q: Does this work with dynamic shapes?

A: The autotuner uses TensorRT for benchmarking, which requires fixed shapes. Set fixed input shapes in the ONNX model before optimization. If the model was exported with dynamic shapes, one option is to **use Polygraphy to fix them to static shapes**, for example:   

```bash  
$ polygraphy surgeon sanitize --override-input-shapes x:[128,3,1024,1024] -o model_bs128.onnx model.onnx   
```

Q: Can I optimize for accuracy instead of latency?    

A: Currently, the autotuner optimizes for latency only.

Example 1: Basic Optimization    

```bash
# Optimize a ResNet model with INT8 quantization
python -m modelopt.onnx.quantization.autotune \
    --onnx_path resnet50.onnx         \
    --output_dir ./resnet50_optimized \
    --quant_type int8                 \
    --schemes_per_region 30
```  

Example 2: Transfer Learning with Pattern Cache   

```bash
# Optimize GPT-2 small
python -m modelopt.onnx.quantization.autotune \
    --onnx_path gpt2_small.onnx   \
    --output_dir ./gpt2_small_run \
    --quant_type fp8              \
    --schemes_per_region 50

# Reuse patterns for GPT-2 medium (much faster)
python -m modelopt.onnx.quantization.autotune \
    --onnx_path gpt2_medium.onnx   \
    --output_dir ./gpt2_medium_run \
    --quant_type fp8               \
    --pattern_cache ./gpt2_small_run/autotuner_state_pattern_cache.yaml
```

Example 3: Import from Manual Baseline   

```bash  
# With a manually quantized baseline
# Import its patterns as starting point
python -m modelopt.onnx.quantization.autotune \
    --onnx_path model.onnx                 \
    --output_dir ./auto_optimized          \
    --qdq_baseline manually_quantized.onnx \
    --schemes_per_region 40
```

Example 4: Full Python Workflow   

```py   
from pathlib import Path
from modelopt.onnx.quantization.autotune.workflows import (
    region_pattern_autotuning_workflow,
    init_benchmark_instance
)

# Initialize TensorRT benchmark
init_benchmark_instance(
    timing_cache_file="/tmp/trt_cache.cache",
    warmup_runs=5,
    timing_runs=20
)

# Run optimization (only non-defaults shown; see API for all options)
autotuner = region_pattern_autotuning_workflow(
    model_path="model.onnx",
    output_dir=Path("./results"),
    num_schemes_per_region=30,
)

# Access results
print(f"Baseline latency: {autotuner.baseline_latency_ms:.2f} ms")
print(f"Number of patterns: {len(autotuner.profiled_patterns)}")

# Pattern cache is automatically saved during workflow
# Check the output directory for autotuner_state_pattern_cache.yaml
if autotuner.pattern_cache:
    print(f"Pattern cache contains {autotuner.pattern_cache.num_patterns} patterns")
```

## Conclusion   
The **modelopt.onnx.quantization.autotune** module provides a powerful automated approach to Q/DQ placement optimization. By combining automatic region discovery, pattern-based optimization, and TensorRT performance measurement, it finds optimal quantization strategies without manual tuning.

Next steps: Run the quick start on a model, try different `--schemes_per_region` values, build a pattern cache for the model family, then integrate the optimized model into the deployment pipeline.


--------------------

## Ref  
https://nvidia.github.io/Model-Optimizer/guides/9_autotune.html#   
