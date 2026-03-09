// for comment out UPTK code
#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__

#define __UPTK_CUDA_PLATFORM_NVIDIA__

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <string.h>

#include <UPTK.h>
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

enum UPTKError CUresultToUPTKError(CUresult para);
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RUNTIME_HPP__
