
 /* Copyright 2005-2021 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

/*!
* \file UPTKFFTXt.h
* \brief Public header file for the NVIDIA CUDA FFT library (UPTKFFT)
*/

#ifndef _UPTKFFTXT_H_
#define _UPTKFFTXT_H_
#include <UPTK_libxt.h>
#include "UPTK_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// UPTKfftXtSubFormat identifies the data layout of
// a memory descriptor owned by UPTKFFT.
// note that multi GPU UPTKFFT does not yet support out-of-place transforms
//

typedef enum UPTKfftXtSubFormat_t {
    UPTKFFT_XT_FORMAT_INPUT = 0x00,              //by default input is in linear order across GPUs
    UPTKFFT_XT_FORMAT_OUTPUT = 0x01,             //by default output is in scrambled order depending on transform
    UPTKFFT_XT_FORMAT_INPLACE = 0x02,            //by default inplace is input order, which is linear across GPUs
    UPTKFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,   //shuffled output order after execution of the transform
    UPTKFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,  //shuffled input order prior to execution of 1D transforms
    UPTKFFT_XT_FORMAT_DISTRIBUTED_INPUT = 0x05,
    UPTKFFT_XT_FORMAT_DISTRIBUTED_OUTPUT = 0x06,
    UPTKFFT_FORMAT_UNDEFINED = 0x07
} UPTKfftXtSubFormat;

//
// UPTKfftXtCopyType specifies the type of copy for UPTKfftXtMemcpy
//
typedef enum UPTKfftXtCopyType_t {
    UPTKFFT_COPY_HOST_TO_DEVICE = 0x00,
    UPTKFFT_COPY_DEVICE_TO_HOST = 0x01,
    UPTKFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    UPTKFFT_COPY_UNDEFINED = 0x03
} UPTKfftXtCopyType;

//
// UPTKfftXtQueryType specifies the type of query for UPTKfftXtQueryPlan
//
typedef enum UPTKfftXtQueryType_t {
    UPTKFFT_QUERY_1D_FACTORS = 0x00,
    UPTKFFT_QUERY_UNDEFINED = 0x01
} UPTKfftXtQueryType;

typedef struct UPTKfftXt1dFactors_t {
    long long int size;
    long long int stringCount;
    long long int stringLength;
    long long int substringLength;
    long long int factor1;
    long long int factor2;
    long long int stringMask;
    long long int substringMask;
    long long int factor1Mask;
    long long int factor2Mask;
    int stringShift;
    int substringShift;
    int factor1Shift;
    int factor2Shift;
} UPTKfftXt1dFactors;

//
// UPTKfftXtWorkAreaPolicy specifies policy for UPTKfftXtSetWorkAreaPolicy
//
typedef enum UPTKfftXtWorkAreaPolicy_t {
    UPTKFFT_WORKAREA_MINIMAL = 0, /* maximum reduction */
    UPTKFFT_WORKAREA_USER = 1, /* use workSize parameter as limit */
    UPTKFFT_WORKAREA_PERFORMANCE = 2, /* default - 1x overhead or more, maximum performance */
} UPTKfftXtWorkAreaPolicy;

// multi-GPU routines
UPTKfftResult UPTKFFTAPI UPTKfftXtSetGPUs(UPTKfftHandle handle, int nGPUs, int *whichGPUs);

UPTKfftResult UPTKFFTAPI UPTKfftXtMalloc(UPTKfftHandle plan,
                                   UPTKLibXtDesc ** descriptor,
                                   UPTKfftXtSubFormat format);

UPTKfftResult UPTKFFTAPI UPTKfftXtMemcpy(UPTKfftHandle plan,
                                   void *dstPointer,
                                   void *srcPointer,
                                   UPTKfftXtCopyType type);

UPTKfftResult UPTKFFTAPI UPTKfftXtFree(UPTKLibXtDesc *descriptor);

UPTKfftResult UPTKFFTAPI UPTKfftXtSetWorkArea(UPTKfftHandle plan, void **workArea);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorC2C(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output,
                                              int direction);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorR2C(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorC2R(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorZ2Z(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output,
                                              int direction);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorD2Z(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptorZ2D(UPTKfftHandle plan,
                                              UPTKLibXtDesc *input,
                                              UPTKLibXtDesc *output);

// Utility functions

UPTKfftResult UPTKFFTAPI UPTKfftXtQueryPlan(UPTKfftHandle plan, void *queryStruct, UPTKfftXtQueryType queryType);


// callbacks


typedef enum UPTKfftXtCallbackType_t {
    UPTKFFT_CB_LD_COMPLEX = 0x0,
    UPTKFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    UPTKFFT_CB_LD_REAL = 0x2,
    UPTKFFT_CB_LD_REAL_DOUBLE = 0x3,
    UPTKFFT_CB_ST_COMPLEX = 0x4,
    UPTKFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    UPTKFFT_CB_ST_REAL = 0x6,
    UPTKFFT_CB_ST_REAL_DOUBLE = 0x7,
    UPTKFFT_CB_UNDEFINED = 0x8

} UPTKfftXtCallbackType;

typedef UPTKfftComplex (*UPTKfftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef UPTKfftDoubleComplex (*UPTKfftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef UPTKfftReal (*UPTKfftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef UPTKfftDoubleReal(*UPTKfftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);

typedef void (*UPTKfftCallbackStoreC)(void *dataOut, size_t offset, UPTKfftComplex element, void *callerInfo, void *sharedPointer);
typedef void (*UPTKfftCallbackStoreZ)(void *dataOut, size_t offset, UPTKfftDoubleComplex element, void *callerInfo, void *sharedPointer);
typedef void (*UPTKfftCallbackStoreR)(void *dataOut, size_t offset, UPTKfftReal element, void *callerInfo, void *sharedPointer);
typedef void (*UPTKfftCallbackStoreD)(void *dataOut, size_t offset, UPTKfftDoubleReal element, void *callerInfo, void *sharedPointer);


UPTKfftResult UPTKFFTAPI UPTKfftXtSetCallback(UPTKfftHandle plan, void **callback_routine, UPTKfftXtCallbackType cbType, void **caller_info);
UPTKfftResult UPTKFFTAPI UPTKfftXtClearCallback(UPTKfftHandle plan, UPTKfftXtCallbackType cbType);
UPTKfftResult UPTKFFTAPI UPTKfftXtSetCallbackSharedSize(UPTKfftHandle plan, UPTKfftXtCallbackType cbType, size_t sharedSize);

UPTKfftResult UPTKFFTAPI UPTKfftXtMakePlanMany(UPTKfftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         UPTKDataType inputtype,
                                         long long int *onembed,
                                         long long int ostride,
                                         long long int odist,
                                         UPTKDataType outputtype,
                                         long long int batch,
                                         size_t *workSize,
                                       	 UPTKDataType executiontype);

UPTKfftResult UPTKFFTAPI UPTKfftXtGetSizeMany(UPTKfftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride,
                                        long long int idist,
                                        UPTKDataType inputtype,
                                        long long int *onembed,
                                        long long int ostride,
                                        long long int odist,
                                        UPTKDataType outputtype,
                                        long long int batch,
                                        size_t *workSize,
                                        UPTKDataType executiontype);


UPTKfftResult UPTKFFTAPI UPTKfftXtExec(UPTKfftHandle plan,
                                 void *input,
                                 void *output,
                                 int direction);

UPTKfftResult UPTKFFTAPI UPTKfftXtExecDescriptor(UPTKfftHandle plan,
                                           UPTKLibXtDesc *input,
                                           UPTKLibXtDesc *output,
                                           int direction);

UPTKfftResult UPTKFFTAPI UPTKfftXtSetWorkAreaPolicy(UPTKfftHandle plan, UPTKfftXtWorkAreaPolicy policy, size_t *workSize);

typedef struct UPTKfftBox3d_t {
    size_t lower[3];
    size_t upper[3];
    size_t strides[3];
} UPTKfftBox3d;

UPTKfftResult UPTKFFTAPI UPTKfftXtSetDistribution(UPTKfftHandle plan,
                                            const UPTKfftBox3d *box_in,
                                            const UPTKfftBox3d *box_out);

#ifdef __cplusplus
}
#endif

#endif
