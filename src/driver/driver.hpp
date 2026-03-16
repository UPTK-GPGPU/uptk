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

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

enum UPTKError CUresultToUPTKError(CUresult para);
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RUNTIME_HPP__
