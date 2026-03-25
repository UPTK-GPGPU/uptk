#ifndef __FFT_HPP__
#define __FFT_HPP__

#include "../runtime/runtime.hpp"
#include <UPTK_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>

// below is cuda header
#include <UPTK_fft.h>
#include <UPTK_fftXt.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cufftResult UPTKfftResultTocufftResult(UPTKfftResult para);
UPTKfftResult cufftResultToUPTKfftResult(cufftResult para);

cufftType UPTKfftTypeTocufftType(UPTKfftType para);

cudaDataType UPTKDataTypeTocudaDataType(UPTKDataType para);

cufftXtCallbackType UPTKfftXtCallbackTypeTocufftXtCallbackType(UPTKfftXtCallbackType para);

cufftXtSubFormat UPTKfftXtSubFormatTocuftXtSubFormat(UPTKfftXtSubFormat para);

cufftXtCopyType UPTKfftXtCopyTypeTocufftXtCopyType(UPTKfftXtCopyType para);

cufftXtQueryType UPTKfftXtQueryTypeTohcufftXtQueryType(UPTKfftXtQueryType para);

cufftXtWorkAreaPolicy UPTKfftXtWorkAreaPolicyTocufftXtWorkAreaPolicy(UPTKfftXtWorkAreaPolicy para);


#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __FFT_HPP__
