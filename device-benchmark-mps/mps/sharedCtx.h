/*
 * Copyright 2023 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_shared_ctx_h__
#define __cuda_etbl_shared_ctx_h__

#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Backdoor driver API for Shared Contexts
//------------------------------------------------------------------
// This etbl contains functions to create shared contexts on Tegra platforms
// which share the underlying compute engine.

CUuuid CU_ETID_SHARED_CONTEXT = {
    {0xd9, 0xbc, 0x88, 0x52, 0xd0, 0xcb, 0xcc, 0x4e, 0x8e, 0x63, 0xe4, 0x30, 0x69, 0x9a, 0x46, 0xb6}};

/* A Device Key which is unique across processes */
typedef unsigned long long int CUetblSharedCtx_deviceKey;

/* A Share Key used to create a shared context */
typedef struct CUetblSharedCtx_shareKey_st
{
    unsigned char key[16];
} CUetblSharedCtx_shareKey;

/* CUDA Shared Context create parameters */
typedef struct CU_ETBL_SHARED_CONTEXT_CREATE_PARAMS_st
{
    // Share Context Key
    CUetblSharedCtx_shareKey *shareKey;

    // Percentage of available threads on the device
    float activeThreadsPercentage;

    // Same as context creation flags
    unsigned int flags;
} CU_ETBL_SHARED_CONTEXT_CREATE_PARAMS;

typedef struct CUetblSharedCtx_st
{
    size_t struct_size;

    /** @brief Returns a unique key associated with the device.
     *
     * The device key can be used to generate a share token for \p dev.
     *
     * \param deviceKey[out] Returned device key.
     * \param dev[in] Device handle.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     */
    CUresult (*etiDeviceGetKey)(CUetblSharedCtx_deviceKey *deviceKey, CUdevice dev);

    /** @brief Returns a shared key to share a context.
     *
     * Create a shared key to share \p ctx across processes with device
     * associated with \p deviceKey.
     * \p shareKey can be used to create one shared context. To create
     * multiple shared contexts, multiple shared keys must be created.
     *
     * \param shareKey[out] Returned share key.
     * \param deviceKey[in] Unique device key.
     * \param ctx[in] CUDA context that must be shared.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_UNKNOWN
     */
    CUresult (*etiSharedCtxKeyCreate)(CUetblSharedCtx_shareKey *shareKey, CUetblSharedCtx_deviceKey deviceKey,
                                      CUcontext ctx);

    /** @brief Creates a shared context.
     *
     * If value of \p pParams->shareKey is NULL, then the API creates a
     * context similar to ::cuCtxCreate. If the value of \p pParams->shareKey
     * is non-NULL then the API creates a shared context associated with
     * \p pParams->shareKey.
     * Kernels running on a shared context share the compute engine parallely
     * with other context/contexts (associated with the share key).
     * The CUcontext handle returned can be passed to all APIs that accept
     * context handle.
     *
     * To destroy the context call ::cuCtxDestroy().
     *
     * Only a limited number of shared contexts (max 64) can be created that
     * share the HW concurrently. ::CUDA_ERROR_UNKNOWN is returned if the
     * context creation fails because of unavailability of resources.
     *
     * ::CUDA_DEVICE_MAX_CONNECTIONS can be used to configure the number of
     * connections in a shared context.
     * When the current context is a shared context, then the device
     * attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT returns the number
     * of enabled SMs set using \p pParams->activeThreadsPercentage.
     *
     * \param pCtx[out] Returned shared context handle.
     * \param device[in] The device on which shared context must be created.
     * \param pParams[in] A pointer to shared context create parameters.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_OUT_OF_MEMORY,
     * ::CUDA_ERROR_UNKNOWN
     */
    CUresult (*etiSharedCtxCreate)(CUcontext *pCtx, CUdevice device, CU_ETBL_SHARED_CONTEXT_CREATE_PARAMS *pParams);
    /** @brief Enables primary context to be a shared context.
     *
     * Enables primary context to be a shared context on \p device.
     * \p shareKey should be used to enable sharing the primary context
     * with an already created primary context in another process or thread.
     * If \p shareKey is NULL, CUDA_ERROR_INVALID_VALUE is returned.
     *
     * If the API is invoked with an initialized primary context,
     * CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
     *
     * The API does not create or initialize a primary context.
     *
     * Note that, a shared primary context follows the same properties
     * and constraints as that of a primary context.
     *
     * \param shareKey[in] The share key that should be used to share
     * the primary context with.
     * context is created.
     * \param device[in] The device on which shared primary context must
     * be created.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
     */
    CUresult (*etiEnableSharedPrimaryCtx)(
        CUetblSharedCtx_shareKey *shareKey,
        CUdevice device);
} CUetblSharedCtx;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __cuda_etbl_shared_ctx_h__
