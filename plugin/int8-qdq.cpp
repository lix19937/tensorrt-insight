

// clang-format off

inline __device__ char quantize(const float x, const float qScale)
{
  int32_t tmpq = __float2int_rn(qScale * x); // scale and round
  char tmpq8 = min(127, max(-127, tmpq));    // clip and cast
  return tmpq8;
}

int32_t enqueue(
    PluginTensorDesc const *inputDesc,
    PluginTensorDesc const *outputDesc,
    void const *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) noexcept {
  const int32_t ld    = inputDesc[0].dims.d[1];
  const int32_t total = inputDesc[0].dims.d[2];

  // dq
  float const dqScaleIn   = inputDesc[0].scale;
  float const dqScaleSkip = inputDesc[1].scale; // ------------
 
 // q 
  float const qScale     = 1.F / outputDesc[0].scale;
  float const qSkipScale = 1.F / outputDesc[1].scale; // ----------

  int8_t const *input = static_cast<int8_t const *>(inputs[0]);
  int8_t const *skip  = static_cast<int8_t const *>(inputs[1]);

  int8_t *output = static_cast<int8_t *>(outputs[0]);
  int8_t *preln  = static_cast<int8_t *>(outputs[1]);

  half const *gamma = static_cast<half const *>(mGammaDev.get());
  half const *beta  = static_cast<half const *>(mBetaDev.get());
}

  template <int32_t TPB, int32_t VPT>
  __global__ void skiplnDQQ_vec4(int32_t const ld, int8_t const *input, int8_t const *skip,
                                 int8_t *output, int8_t *preln,
                                 half const *beta,
                                 half const *gamma,
                                 float const dqScaleIn,
                                 float const dqScaleSkip,
                                 float const qScale,
                                 float const qSkipScale,
                                 int32_t const total)
  {
    int32_t const hinner = threadIdx.x % 4;
    int32_t const houter = threadIdx.x / 4;

    int32_t const tidx = threadIdx.x;
    int32_t const bidx = blockIdx.x;
    int32_t const idx = houter * total * 32 + bidx * 32 + hinner * VPT;

    int8_t inLocal[VPT];
    int8_t skipLocal[VPT];

    half inLocalDQ[VPT]; // dequantized input + skip
    half betaLocal[VPT];
    half gammaLocal[VPT];

    // load input tensors to local var
    copy<sizeof(int8_t) * VPT>(&input[idx], inLocal);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skipLocal);

    // load parameters
    copy<sizeof(half) * VPT>(&beta[tidx * VPT], betaLocal);
    copy<sizeof(half) * VPT>(&gamma[tidx * VPT], gammaLocal);

    half2 statsLocal = __floats2half2_rn(0.F, 0.F); // accumulator

    half const rld = half(1.F) / half(ld);

#pragma unroll
    for (int32_t it = 0; it < VPT; ++it)
    {
      // DQ input and skip
      float const tmpIn   = inLocal[it]; // from int8 cast to fp32
      float const tmpSkip = skipLocal[it];

      inLocalDQ[it] = dqScaleIn * tmpIn + dqScaleSkip * tmpSkip; /// dq to fp32

      half const tmp = rld * inLocalDQ[it];
      half2 const tmp2 = __halves2half2(tmp, tmp * inLocalDQ[it]);
      statsLocal = statsLocal + tmp2;
    }

    using BlockReduce = cub::BlockReduce<half2, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ half mu;     // mean
    __shared__ half rsigma; // 1 / std.dev.

    half2 const sum2 = BlockReduce(tempStorage).Reduce(statsLocal, cub::Sum());

    // Copy skip connection output before Layer Norm
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
      inLocal[it] = quantize(inLocalDQ[it], qSkipScale); /// q fp32 to int8
    }
    copy<sizeof(int8_t) * VPT>(inLocal, &preln[idx]);

    if (tidx == 0)
    {
      mu = __low2half(sum2);
      rsigma = rsqrtf(__high2half(sum2) - mu * mu);
    }

    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t outLocal[VPT / 4U];
#pragma unroll
    for (int32_t it = 0; it < VPT / 4U; it++)
    {
      float const tmp0 = gammaLocal[it * 4 + 0] * (inLocalDQ[it * 4 + 0] - mu) * rsigma + betaLocal[it * 4 + 0];
      float const tmp1 = gammaLocal[it * 4 + 1] * (inLocalDQ[it * 4 + 1] - mu) * rsigma + betaLocal[it * 4 + 1];
      float const tmp2 = gammaLocal[it * 4 + 2] * (inLocalDQ[it * 4 + 2] - mu) * rsigma + betaLocal[it * 4 + 2];
      float const tmp3 = gammaLocal[it * 4 + 3] * (inLocalDQ[it * 4 + 3] - mu) * rsigma + betaLocal[it * 4 + 3];
      outLocal[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
  }

  ////////
int32_t enqueue(
    PluginTensorDesc const *inputDesc,
    PluginTensorDesc const *outputDesc,
    void const *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) noexcept{
    float const dqScaleIn   = inputDesc[0].scale;
    float const dqScaleSkip = inputDesc[1].scale;

    PLUGIN_VALIDATE(outputDesc[0].scale != 0.0F);
    float const qScale      = 1.F / outputDesc[0].scale;

    auto const *const input = static_cast<int8_t const *>(inputs[0]);
    auto const *const skip  = static_cast<int8_t const *>(inputs[1]);

    auto *output = static_cast<int8_t *>(outputs[0]);

    auto const *const beta  = static_cast<half const *>(mBetaDev.get());
    auto const *const gamma = static_cast<half const *>(mGammaDev.get());

    status = computeSkipLayerNormDQQ<false>(stream, static_cast<int32_t>(mLd), inputVolume, input, skip,
                                            beta, gamma, output, bias, dqScaleIn, dqScaleSkip, qScale);
  }

  template <int32_t TPB, int32_t VPT, bool hasBias>
  __global__ void skiplnDQQ(int32_t const ld, int8_t const *input, int8_t const *skip, int8_t *output, __half const *beta,
                            __half const *gamma, __half const *bias, float const dqScaleIn, float const dqScaleSkip, float const qScale)
  {
    int32_t const idx = ld * blockIdx.x + threadIdx.x * VPT;
    int8_t inLocal[VPT];
    int8_t skipLocal[VPT];

    __half inLocalDQ[VPT]; // dequantized input + skip + bias
    __half biasLocal[VPT]; // bias and beta
    __half gammaLocal[VPT];

    copy<sizeof(int8_t) * VPT>(&input[idx], inLocal);
    copy<sizeof(int8_t) * VPT>(&skip[idx], skipLocal);
    copy<sizeof(__half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);

    __half2 loc = __floats2half2_rn(0.f, 0.f); // used for accumulator

    const __half rld = __half(1) / __half(ld);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
      // DQ input and skip
      float const tmpIn   = inLocal[it]; // int8 cast to fp32
      float const tmpSkip = skipLocal[it];

      inLocalDQ[it] = dqScaleIn * tmpIn + dqScaleSkip * tmpSkip; // dq to fp32

      const __half tmp = rld * inLocalDQ[it];
      const __half2 tmp2 = __halves2half2(tmp, tmp * inLocalDQ[it]);
      loc += tmp2;
    }

    // load parameters
    copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT],  biasLocal);
    copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], gammaLocal);

    using BlockReduce = cub::BlockReduce<__half2, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ __half mu;     // mean
    __shared__ __half rsigma; // 1 / std.dev.

    const __half2 sum2 = BlockReduce(tempStorage).Reduce(loc,
      [](auto const &lhs, auto const &rhs){ return lhs + rhs; });

    if (threadIdx.x == 0)
    {
      mu = __low2half(sum2);
      rsigma = rsqrt(__high2half(sum2) - mu * mu);
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, ""); // make sure %4
    uint32_t outLocal[VPT / 4U];
#pragma unroll
    for (int32_t it = 0; it < VPT / 4U; it++)
    {
      float const tmp0 = gammaLocal[it * 4 + 0] * (inLocalDQ[it * 4 + 0] - mu) * rsigma + biasLocal[it * 4 + 0];
      float const tmp1 = gammaLocal[it * 4 + 1] * (inLocalDQ[it * 4 + 1] - mu) * rsigma + biasLocal[it * 4 + 1];
      float const tmp2 = gammaLocal[it * 4 + 2] * (inLocalDQ[it * 4 + 2] - mu) * rsigma + biasLocal[it * 4 + 2];
      float const tmp3 = gammaLocal[it * 4 + 3] * (inLocalDQ[it * 4 + 3] - mu) * rsigma + biasLocal[it * 4 + 3];
      outLocal[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
  }

// clang-format on
