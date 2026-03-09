// for comment out UPTK code
#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__
#define __UPTK_CUDA_PLATFORM_NVIDIA__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h> //printf报错
#include <algorithm> //min报错
#include <string.h> //strncpy报错

// below is UPTK header
#include <UPTK_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)
#define ERROR_INVALID_OR_UNSUPPORTED_ENUM() do{printf("[ERROR] The enumeration passed in is invalid, or the functionality for the enumeration is currently not supported. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

const int SM_VERSION_MAJOR = 7;
const int SM_VERSION_MINOR = 5;

cudaError_t UPTKErrorTocudaError(enum UPTKError para);
//cudaClusterSchedulingPolicy UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para);
enum cudaSynchronizationPolicy UPTKSynchronizationPolicyTocudaSynchronizationPolicy(enum UPTKSynchronizationPolicy para);
cudaAccessProperty UPTKAccessPropertyTocudaAccessProperty(enum UPTKAccessProperty para);
void UPTKAccessPolicyWindowTocudaAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, cudaAccessPolicyWindow * cuda_para);
enum UPTKError cudaErrorToUPTKError(cudaError_t para);
cudaMemcpyKind UPTKMemcpyKindTocudaMemcpyKind(enum UPTKMemcpyKind para);
cudaStreamCaptureMode UPTKStreamCaptureModeTocudaStreamCaptureMode(enum UPTKStreamCaptureMode para);
enum UPTKStreamCaptureStatus cudaStreamCaptureStatusToUPTKStreamCaptureStatus(cudaStreamCaptureStatus para);
void UPTKIpcMemHandleTocudaIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, cudaIpcMemHandle_t * cuda_para);
void UPTKDevicePropTocudaDeviceProp(const struct UPTKDeviceProp * UPTK_para, cudaDeviceProp * cuda_para);
void cudaDevicePropToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para);
enum UPTKFuncCache cudaFuncCacheToUPTKFuncCache(cudaFuncCache para);
CUlimit UPTKlimitToCUlimit(UPTKlimit para);
void UPTKKernelNodeParamsTocudaKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, cudaKernelNodeParams * cuda_para);
void UPTKPosTocudaPos(const struct UPTKPos * UPTK_para, cudaPos * cuda_para);
void UPTKPitchedPtrTocudaPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, cudaPitchedPtr * cuda_para);
void UPTKExtentTocudaExtent(const struct UPTKExtent * UPTK_para, cudaExtent * cuda_para);
enum UPTKGraphExecUpdateResult cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cudaGraphExecUpdateResult para);
void UPTKMemcpy3DParmsTocudaMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, cudaMemcpy3DParms * cuda_para);
//void UPTKLaunchConfig_tTocudaLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, cudaLaunchConfig_t * cuda_para);
//void UPTKLaunchAttributeValueTocudaLaunchAttributeValue(const UPTKLaunchAttributeValue *UPTK_para, cudaLaunchAttributeValue *cuda_para, UPTKLaunchAttributeID para);
//cudaLaunchAttribute UPTKLaunchAttributeTocudaLaunchAttribute(UPTKLaunchAttribute UPTK_para);
//cudaLaunchAttributeID UPTKLaunchAttributeIDTocudaLaunchAttributeID(enum UPTKLaunchAttributeID para);
void cudaFuncAttributesToUPTKFuncAttributes(const cudaFuncAttributes * cuda_para, struct UPTKFuncAttributes * UPTK_para);

UPTKresult cudaErrorToUPTKresult(CUresult para);
CUarray_format UPTKarray_formatToCUarray_format(UPTKarray_format para);
void UPTK_ARRAY_DESCRIPTORToCUDA_ARRAY_DESCRIPTOR(const UPTK_ARRAY_DESCRIPTOR * UPTK_para, CUDA_ARRAY_DESCRIPTOR * cuda_para);
CUjit_option  UPTKjit_optionTonvrtcJIT_option(UPTKjit_option para);
UPTKresult cudaResultToUPTKresult(CUresult para);
CUjitInputType UPTKjitInputTypeTonvrtcJITInputType(UPTKjitInputType para);
cudaDeviceAttr UPTKDeviceAttrTocudaDeviceAttribute(enum UPTKDeviceAttr para);
enum UPTKGraphNodeType cudaGraphNodeTypeToUPTKGraphNodeType(cudaGraphNodeType para);
//void UPTKKernelNodeParamsV2TocudaKernelNodeParams(const UPTKKernelNodeParamsV2 *UPTK_para, cudaKernelNodeParamsV2 *cuda_para);
cudaGraphNodeType UPTKGraphNodeTypeTocudaGraphNodeType(enum UPTKGraphNodeType para);
//void UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(const UPTKMemcpyNodeParams *UPTK_para, cudaMemcpyNodeParams *cuda_para);
//void UPTKMemsetParamsV2TocudaMemsetParams(const UPTKMemsetParamsV2 *UPTK_para, cudaMemsetParamsV2 *cuda_para);
//void UPTKHostNodeParamsV2TocudaHostNodeParams(const UPTKHostNodeParamsV2 *UPTK_para, cudaHostNodeParamsV2 *);
//void UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(const UPTKChildGraphNodeParams *UPTK_para, cudaChildGraphNodeParams *cuda_para);
//void UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(const UPTKEventWaitNodeParams *UPTK_para, cudaEventWaitNodeParams *cuda_para);
//void UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(const UPTKEventRecordNodeParams *UPTK_para, cudaEventRecordNodeParams *cuda_para);
cudaExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para);
cudaExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para);
//cudaExternalSemaphoreSignalNodeParamsV2 UPTKExternalSemaphoreSignalNodeParamsV2TocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para);
cudaMemAllocationType UPTKMemAllocationTypeTocudaMemAllocationType(enum UPTKMemAllocationType para);
cudaMemAllocationHandleType UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(enum UPTKMemAllocationHandleType para);
cudaMemLocationType UPTKMemLocationTypeTocudaMemLocationType(enum UPTKMemLocationType para);
void UPTKMemLocationTocudaMemLocation(const struct UPTKMemLocation * UPTK_para, cudaMemLocation * cuda_para);
//cudaExternalSemaphoreWaitNodeParamsV2 UPTKExternalSemaphoreWaitNodeParamsV2TocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para);
void UPTKMemPoolPropsTocudaMemPoolProps(const UPTKMemPoolProps * UPTK_para, cudaMemPoolProps * cuda_para);
cudaMemAccessFlags UPTKMemAccessFlagsTocudaMemAccessFlags(UPTKMemAccessFlags para);
cudaMemAccessDesc UPTKMemAccessDescTocudaMemAccessDesc(struct UPTKMemAccessDesc UPTK_para);
//void UPTKMemAllocNodeParamsV2TocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, cudaMemAllocNodeParamsV2 *cuda_para);
//void UPTKMemFreeNodeParamsTocudaMemFreeNodeParams(const UPTKMemFreeNodeParams *UPTK_para, cudaMemFreeNodeParams *cuda_para);
//void UPTKConditionalNodeParamsTocudaConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, cudaConditionalNodeParams *cuda_para);
//cudaGraphConditionalNodeType UPTKConditionalNodeTypeTocudaConditionalNodeType(enum UPTKGraphConditionalNodeType para);
//void UPTKGraphNodeParamsTocudaGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, cudaGraphNodeParams * cuda_para);
//void UPTKGraphNodeParamsTomcGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, mcGraphNodeParams * cuda_para);
CUfunction_attribute UPTKfunction_attributeTocudaFunction_attribute(UPTKfunction_attribute para);
cudaFuncAttribute UPTKFuncAttributeTocudaFuncAttribute(enum UPTKFuncAttribute para);

// Based on the newly added application interface

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RUNTIME_HPP__





