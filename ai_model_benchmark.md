orin-x, drive os6060    

|model|fp16     | ptq/best|    |      
| ----|-----    | ------- |----|    
| resnet50      |         |         |    |    
| yolo-v3      |         |         |    |     
| yolo-x       |         |         |    |      


  GR3D_FREQ xx% 是 GPU 驱动在一个很短的窗口内统计的忙碌比例（窗口多长取决于驱动实现）。tegrastats 每到采样间隔就读一次，所以每个点只反映采样那一刻附近的状态，不是两次采样之间的平均。

  推理负载本身是脉冲式的，采样点落在哪里就读到什么：

  - 落在 kernel 执行期间，读到接近 100%
  - 落在两帧之间的空档，读到接近 0%
  - 落在 engine 加载、反序列化、warmup、退出阶段，也是低值

  举例：模型 latency 1.8 ms，按 10Hz 跑，GPU 真实平均占用只有 1.8 × 10 / 1000 ≈ 1.8%，但 tegrastats 的单个点会在 0% 和很高之间随机跳。即使 trtexec 不停地连续跑，CPU enqueue、同步的间隙也会造成跳动。

  另外，这个百分比是时间维度的忙碌率，意思是"GPU 上有没有活在跑"，不代表 SM 被用满。kernel 只用了一半的 SM，也算 100% 忙。

  怎么得到推理过程中的利用率

  按精度从高到低，有三种方法：

  1. 用 trtexec 的结果直接算（最简单，也准）

  GPU 忙碌率 ≈ GPU Compute Time(mean) × Throughput(qps)

  perf.log 里这两项都有。比如 1.80 ms × 555.6 qps ≈ 100%，说明连续跑时 GPU 基本没有空档。要算业务工况，把 qps 换成实际帧率，例如 10Hz 就是 1.8 ms × 10 = 1.8%。

  这个公式只适用于单 stream。用了 --infStreams>1 之后多个 stream 会重叠，算出来会超过 100%，就不能这么用了。

  2. nsys 时间线（精确，还能看清空档在哪里）

  忙碌率 = 采集窗口内 kernel 执行区间的并集时长 / 窗口时长

  用 --capture-range cudaProfilerApi 采集，窗口天然只包含推理阶段，不含加载和 warmup。从 nsys stats -r cuda_gpu_trace 导出 kernel 的起止时间就能计算，相邻 kernel 之间的间隔就是 bubble。

  如果 --gpu-metrics-devices 在 Thor 上可用，还能拿到高频采样的 SM Active、SM Occupancy、DRAM 带宽曲线，这是"SM 用得满不满"层面的利用率，比 GR3D 更有意义。

  3. tegrastats 只作佐证，要做窗口统计，不要看单点

  - 采样只取 loop 稳定运行的区间，去掉开头的加载和 warmup，也去掉结尾
  - loop 要连续跑足够长（几十秒），采样点多了，平均值才收敛到真实忙碌率
  - 报告里写均值、p50、p90 和样本数，不写单次读数
  - GR3D 后面的 @[频率] 也要记下来，DVFS 降频时同样的负载忙碌率会变高；条件允许就先锁频再测

  窗口截取可以按 tegrastats 每行的时间戳（--logfile 会带）。也可以更简单：loop 先跑 10 秒再启动 tegrastats，loop 结束前停掉，然后：

  grep -o 'GR3D_FREQ [0-9]*%' tegrastats.log | tr -d '%' | awk '{print $2}' | sort -n | \
    awk '{a[NR]=$1; s+=$1} END{printf "n=%d mean=%.1f%% p50=%d%% p90=%d%%\n", NR, s/NR, a[int(NR*0.5)+1], a[int(NR*0.9)+1]}'
