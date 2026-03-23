// for comment out UPTK code
#ifndef __UPTK_RUNTIME_HPP__
#define __UPTK_RUNTIME_HPP__
#define __UPTK_CUDA_PLATFORM_NVIDIA__

#include <cuda_runtime_api.h>
//#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h> //printf报错
#include <algorithm> //min报错
#include <string.h> //strncpy报错

// below is UPTK header
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)
#define ERROR_INVALID_OR_UNSUPPORTED_ENUM() do{printf("[ERROR] The enumeration passed in is invalid, or the functionality for the enumeration is currently not supported. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_BOLD "\x1b[1m"
#define ANSI_COLOR_HIGH_SATURATION "\x1b[91m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define Debug() do { \
    printf(ANSI_COLOR_BOLD ANSI_COLOR_HIGH_SATURATION ANSI_COLOR_MAGENTA "Warning: The %s is currently not supported.The %s is invalid.\n" ANSI_COLOR_RESET, __FUNCTION__, __FUNCTION__); \
} while (0)

const int SM_VERSION_MAJOR = 7;
const int SM_VERSION_MINOR = 5;

#define UPTKStreamAttrIDTocudaStreamAttrID UPTKLaunchAttributeIDTocudaLaunchAttributeID
#define cudaStreamAttrValueToUPTKStreamAttrValue cudaLaunchAttributeValueToUPTKLaunchAttributeValue
#define UPTKStreamAttrValueTocudaStreamAttrValue UPTKLaunchAttributeValueTocudaLaunchAttributeValue

cudaError_t UPTKErrorTocudaError(enum UPTKError para);
enum UPTKError cudaErrorToUPTKError(cudaError_t para);
void UPTKLaunchParamsTocudaLaunchParams(const struct UPTKLaunchParams * UPTK_para, cudaLaunchParams * cuda_para);
void UPTKFuncAttributesTocudaFuncAttributes(const struct UPTKFuncAttributes * UPTK_para, cudaFuncAttributes * cuda_para);

enum cudaComputeMode UPTKComputeModeTocudaComputeMode(enum UPTKComputeMode para);
cudaDeviceAttr UPTKDeviceAttrTocudaDeviceAttr(enum UPTKDeviceAttr para);
cudaDeviceP2PAttr UPTKDeviceP2PAttrTocudaDeviceP2PAttr(enum UPTKDeviceP2PAttr para);

cudaFuncAttribute UPTKFuncAttributeTocudaFuncAttribute(enum UPTKFuncAttribute para);
cudaFuncCache UPTKFuncCacheTocudaFuncCache(enum UPTKFuncCache para);
enum UPTKFuncCache cudaFuncCacheToUPTKFuncCache(cudaFuncCache para);

cudaSharedMemConfig UPTKSharedMemConfigTocudaSharedMemConfig(enum UPTKSharedMemConfig para);
enum UPTKSharedMemConfig cudaSharedMemConfigToUPTKSharedMemConfig(cudaSharedMemConfig para);
cudaStreamCaptureMode UPTKStreamCaptureModeTocudaStreamCaptureMode(enum UPTKStreamCaptureMode para);
enum UPTKStreamCaptureMode cudaStreamCaptureModeToUPTKStreamCaptureMode(cudaStreamCaptureMode para);
cudaStreamCaptureStatus UPTKStreamCaptureStatusTocudaStreamCaptureStatus(enum UPTKStreamCaptureStatus para);
//enum UPTKStreamCaptureStatus cudaStreamCaptureStatusToUPTKStreamCaptureStatus(cudaStreamCaptureStatus para);
cudaExternalMemoryHandleType UPTKExternalMemoryHandleTypeTocudaExternalMemoryHandleType(enum UPTKExternalMemoryHandleType para);
cudaExternalSemaphoreHandleType UPTKExternalSemaphoreHandleTypeTocudaExternalSemaphoreHandleType(enum UPTKExternalSemaphoreHandleType para);


enum cudaLimit UPTKLimitTocudaLimit(enum UPTKLimit para);
cudaMemoryAdvise UPTKMemoryAdviseTocudaMemoryAdvise(enum UPTKMemoryAdvise para);
cudaMemLocationType UPTKMemLocationTypeTocudaMemLocationType(enum UPTKMemLocationType para);
cudaMemRangeAttribute UPTKMemRangeAttributeTocudaMemRangeAttribute(enum UPTKMemRangeAttribute para);
enum UPTKMemRangeAttribute cudaMemRangeAttributeToUPTKMemRangeAttribute(cudaMemRangeAttribute para);
cudaMemcpyKind UPTKMemcpyKindTocudaMemcpyKind(enum UPTKMemcpyKind para);
enum UPTKMemcpyKind cudaMemcpyKindToUPTKMemcpyKind(cudaMemcpyKind para);
UPTKMemoryType cudaMemoryTypeToUPTKMemoryType(enum cudaMemoryType para);

void UPTKDevicePropTocudaDeviceProp(const struct UPTKDeviceProp * UPTK_para, cudaDeviceProp * cuda_para);
void cudaDevicePropToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para);
void cudaDeviceProp_v2ToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para);
void UPTKTextureDescTocudaTextureDesc(const struct UPTKTextureDesc * UPTK_para, cudaTextureDesc * cuda_para);

void cudaFuncAttributesToUPTKFuncAttributes(const cudaFuncAttributes * cuda_para, struct UPTKFuncAttributes * UPTK_para);
void UPTKKernelNodeParamsTocudaKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, cudaKernelNodeParams * cuda_para);
void cudaKernelNodeParamsToUPTKKernelNodeParams(const cudaKernelNodeParams * cuda_para, struct UPTKKernelNodeParams * UPTK_para);
void UPTKResourceViewDescTocudaResourceViewDesc(const struct UPTKResourceViewDesc * UPTK_para, struct cudaResourceViewDesc * cuda_para);
void UPTKExtentTocudaExtent(const struct UPTKExtent * UPTK_para, cudaExtent * cuda_para);
void cudaExtentToUPTKExtent(const cudaExtent * cuda_para, struct UPTKExtent * UPTK_para);
void cudaPointerAttributesToUPTKPointerAttributes(const struct cudaPointerAttributes * cuda_para, UPTKPointerAttributes * UPTK_para);
void UPTKMemsetParamsTocudaMemsetParams(const struct UPTKMemsetParams * UPTK_para, cudaMemsetParams * cuda_para);
void cudaMemsetParamsToUPTKMemsetParams(const cudaMemsetParams * cuda_para, struct UPTKMemsetParams * UPTK_para);
void UPTKMemcpy3DParmsTocudaMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, cudaMemcpy3DParms * cuda_para);
void cudaMemcpy3DParmsToUPTKMemcpy3DParms(const cudaMemcpy3DParms * cuda_para, struct UPTKMemcpy3DParms * UPTK_para);
void UPTKHostNodeParamsTocudaHostNodeParams(const struct UPTKHostNodeParams * UPTK_para, cudaHostNodeParams * cuda_para);
void cudaHostNodeParamsToUPTKHostNodeParams(const cudaHostNodeParams * cuda_para, struct UPTKHostNodeParams * UPTK_para);
void UPTKMemLocationTocudaMemLocation(const struct UPTKMemLocation * UPTK_para, cudaMemLocation * cuda_para);
void UPTKPitchedPtrTocudaPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, cudaPitchedPtr * cuda_para);
void cudaPitchedPtrToUPTKPitchedPtr(const cudaPitchedPtr * cuda_para, struct UPTKPitchedPtr * UPTK_para);
void UPTKResourceDescTocudaResourceDesc(const struct UPTKResourceDesc * UPTK_para, cudaResourceDesc * cuda_para);
void cudaResourceDescToUPTKResourceDesc(const cudaResourceDesc * cuda_para, struct UPTKResourceDesc * UPTK_para);

void UPTKPosTocudaPos(const struct UPTKPos * UPTK_para, cudaPos * cuda_para);
void cudaPosToUPTKPos(const cudaPos * cuda_para, struct UPTKPos * UPTK_para);
cudaGraphicsRegisterFlags UPTKGraphicsRegisterFlagsTocudaGraphicsRegisterFlags(enum UPTKGraphicsRegisterFlags para);
cudaKernelNodeAttrID UPTKKernelNodeAttrIDTocudaKernelNodeAttrID(enum UPTKKernelNodeAttrID para);
cudaGraphExecUpdateResult UPTKGraphExecUpdateResultTocudaGraphExecUpdateResult(enum UPTKGraphExecUpdateResult para);
enum UPTKGraphExecUpdateResult cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cudaGraphExecUpdateResult para);
cudaGraphNodeType UPTKGraphNodeTypeTocudaGraphNodeType(enum UPTKGraphNodeType para);
enum UPTKGraphNodeType cudaGraphNodeTypeToUPTKGraphNodeType(cudaGraphNodeType para);
void UPTKIpcEventHandleTocudaIpcEventHandle(const UPTKIpcEventHandle_t * UPTK_para, cudaIpcEventHandle_t * cuda_para);
void cudaIpcEventHandleToUPTKIpcEventHandle(const cudaIpcEventHandle_t * cuda_para, UPTKIpcEventHandle_t * UPTK_para);
void UPTKIpcMemHandleTocudaIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, cudaIpcMemHandle_t * cuda_para);
void cudaIpcMemHandleToUPTKIpcMemHandle(const cudaIpcMemHandle_t * cuda_para, UPTKIpcMemHandle_t * UPTK_para);
enum cudaTextureAddressMode UPTKTextureAddressModeTocudaTextureAddressMode(enum UPTKTextureAddressMode para);
enum cudaSurfaceBoundaryMode UPTKSurfaceBoundaryModeTocudaSurfaceBoundaryMode(enum UPTKSurfaceBoundaryMode para);
cudaChannelFormatKind UPTKChannelFormatKindTocudaChannelFormatKind(enum UPTKChannelFormatKind para);
enum UPTKChannelFormatKind cudaChannelFormatKindToUPTKChannelFormatKind(cudaChannelFormatKind para);

void UPTKChannelFormatDescTocudaChannelFormatDesc(const struct UPTKChannelFormatDesc * UPTK_para, cudaChannelFormatDesc * cuda_para);
void cudaChannelFormatDescToUPTKChannelFormatDesc(const cudaChannelFormatDesc * cuda_para, struct UPTKChannelFormatDesc * UPTK_para);
void UPTKExternalMemoryBufferDescTocudaExternalMemoryBufferDesc(const struct UPTKExternalMemoryBufferDesc * UPTK_para, cudaExternalMemoryBufferDesc * cuda_para);
void UPTKExternalMemoryHandleDescTocudaExternalMemoryHandleDesc(const struct UPTKExternalMemoryHandleDesc * UPTK_para, cudaExternalMemoryHandleDesc * cuda_para);
void UPTKExternalSemaphoreHandleDescTocudaExternalSemaphoreHandleDesc(const struct UPTKExternalSemaphoreHandleDesc * UPTK_para, cudaExternalSemaphoreHandleDesc * cuda_para);
enum cudaTextureFilterMode UPTKTextureFilterModeTocudaTextureFilterMode(enum UPTKTextureFilterMode para);
enum cudaTextureReadMode UPTKTextureReadModeTocudaTextureReadMode(enum UPTKTextureReadMode para);
cudaResourceViewFormat UPTKResourceViewFormatTocudaResourceViewFormat(enum UPTKResourceViewFormat para);
cudaResourceType UPTKResourceTypeTocudaResourceType(enum UPTKResourceType para);
enum UPTKResourceType cudaResourceTypeToUPTKResourceType(cudaResourceType para);

void UPTKKernelNodeAttrValueTocudaKernelNodeAttrValue(const UPTKKernelNodeAttrValue *UPTK_para, cudaKernelNodeAttrValue *cuda_para, UPTKKernelNodeAttrID attr);
void cudaKernelNodeAttrValueToUPTKKernelNodeAttrValue(const cudaKernelNodeAttrValue *cuda_para, UPTKKernelNodeAttrValue *UPTK_para, UPTKKernelNodeAttrID attr);
cudaMemAccessFlags UPTKMemAccessFlagsTocudaMemAccessFlags(UPTKMemAccessFlags para);
UPTKMemAccessFlags cudaMemAccessFlagsToUPTKMemAccessFlags(cudaMemAccessFlags para);
cudaMemAllocationType UPTKMemAllocationTypeTocudaMemAllocationType(enum UPTKMemAllocationType para);
cudaMemAllocationHandleType UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(enum UPTKMemAllocationHandleType para);
cudaAccessProperty UPTKAccessPropertyTocudaAccessProperty(enum UPTKAccessProperty para);
enum UPTKAccessProperty cudaAccessPropertyToUPTKAccessProperty(cudaAccessProperty para);
void UPTKAccessPolicyWindowTocudaAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, cudaAccessPolicyWindow * cuda_para);
void cudaAccessPolicyWindowToUPTKAccessPolicyWindow(const cudaAccessPolicyWindow * cuda_para, struct UPTKAccessPolicyWindow * UPTK_para);
cudaGraphMemAttributeType UPTKGraphMemAttributeTypeTocudaGraphMemAttributeType(enum UPTKGraphMemAttributeType para);
void UPTKMemPoolPropsTocudaMemPoolProps(const UPTKMemPoolProps * UPTK_para, cudaMemPoolProps * cuda_para);
void CUmemPoolPropsTocudaMemPoolProps(const CUmemPoolProps * UPTK_para, cudaMemPoolProps * cuda_para);
cudaMemAccessDesc UPTKMemAccessDescTocudaMemAccessDesc(struct UPTKMemAccessDesc UPTK_para);
cudaMemPoolAttr UPTKMemPoolAttrTocudaMemPoolAttr(enum UPTKMemPoolAttr para);
void UPTKMemAllocNodeParamsTocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParams *UPTK_para, cudaMemAllocNodeParams *cuda_para);
void cudaMemAllocNodeParamsToUPTKMemAllocNodeParams(const struct cudaMemAllocNodeParams *cuda_para, UPTKMemAllocNodeParams *UPTK_para);
void cudaMemPoolPropsToUPTKMemPoolProps(const cudaMemPoolProps * cuda_para, UPTKMemPoolProps * UPTK_para);
UPTKMemAllocationType cudaMemAllocationTypeToUPTKMemAllocationType(enum cudaMemAllocationType para);
UPTKMemAllocationHandleType cudaMemAllocationHandleTypeToUPTKMemAllocationHandleType(enum cudaMemAllocationHandleType para);
void cudaMemLocationToUPTKMemLocation(const struct cudaMemLocation * cuda_para, UPTKMemLocation * UPTK_para);
UPTKMemLocationType cudaMemLocationTypeToUPTKMemLocationType(enum cudaMemLocationType para);
UPTKMemAccessDesc cudaMemAccessDescTUPTKMemAccessDesc(struct cudaMemAccessDesc cuda_para);
cudaGLDeviceList UPTKGLDeviceListTocudaGLDeviceList(UPTKGLDeviceList para);
cudaExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para);
UPTKExternalSemaphoreWaitParams cudaExternalSemaphoreWaitParamsToUPTKExternalSemaphoreWaitParams(cudaExternalSemaphoreWaitParams cuda_para);
cudaExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para);
void UPTKMemcpy3DPeerParmsTocudaMemcpy3DPeerParms(const struct UPTKMemcpy3DPeerParms * UPTK_para, cudaMemcpy3DPeerParms * cuda_para);
void UPTKArrayMemoryRequirementsTocudaArrayMemoryRequirements(const struct UPTKArrayMemoryRequirements * UPTK_para, cudaArrayMemoryRequirements * cuda_para);
void cudaArrayMemoryRequirementsToUPTKArrayMemoryRequirements(const struct cudaArrayMemoryRequirements * cuda_para, UPTKArrayMemoryRequirements * UPTK_para);
void UPTKLaunchConfig_tTocudaLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, cudaLaunchConfig_t * cuda_para);
cudaLaunchAttribute UPTKLaunchAttributeTocudaLaunchAttribute(UPTKLaunchAttribute  UPTK_para);
cudaLaunchAttributeID UPTKLaunchAttributeIDTocudaLaunchAttributeID(enum UPTKLaunchAttributeID para);
void UPTKLaunchAttributeValueTocudaLaunchAttributeValue(const UPTKLaunchAttributeValue * UPTK_para, cudaLaunchAttributeValue * cuda_para, UPTKLaunchAttributeID attr);
void cudaLaunchAttributeValueToUPTKLaunchAttributeValue(const cudaLaunchAttributeValue * cuda_para, UPTKLaunchAttributeValue * UPTK_para, cudaLaunchAttributeID para);
enum cudaSynchronizationPolicy UPTKSynchronizationPolicyTocudaSynchronizationPolicy(enum UPTKSynchronizationPolicy para);
cudaClusterSchedulingPolicy UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para);
enum UPTKSynchronizationPolicy cudaSynchronizationPolicyToUPTKSynchronizationPolicy(enum cudaSynchronizationPolicy para);
UPTKClusterSchedulingPolicy cudaClusterSchedulingPolicyToUPTKClusterSchedulingPolicy(cudaClusterSchedulingPolicy para);
//size_t getElementSize(const hipArray_const_t para);
enum UPTKTextureAddressMode cudaTextureAddressModeToUPTKTextureAddressMode(enum cudaTextureAddressMode para);
enum UPTKTextureFilterMode cudaTextureFilterModeToUPTKTextureFilterMode(enum cudaTextureFilterMode para);
enum UPTKTextureReadMode cudaTextureReadModeToUPTKTextureReadMode(enum cudaTextureReadMode para);
UPTKExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParamsToUPTKExternalSemaphoreSignalParams(cudaExternalSemaphoreSignalParams cuda_para);
void UPTKExternalMemoryMipmappedArrayDescTocudaExternalMemoryMipmappedArrayDesc(const UPTKExternalMemoryMipmappedArrayDesc * UPTK_para, cudaExternalMemoryMipmappedArrayDesc * cuda_para);
cudaExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsTocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParams UPTK_para);
UPTKExternalSemaphoreSignalNodeParams cudaExternalSemaphoreSignalNodeParamsToUPTKExternalSemaphoreSignalNodeParams(cudaExternalSemaphoreSignalNodeParams cuda_para);
cudaExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsTocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParams UPTK_para);
UPTKExternalSemaphoreWaitNodeParams cudaExternalSemaphoreWaitNodeParamsToUPTKExternalSemaphoreWaitNodeParams(cudaExternalSemaphoreWaitNodeParams cuda_para);
UPTKDriverEntryPointQueryResult cudaDriverEntryPointQueryResultToUPTKDriverEntryPointQueryResult(cudaDriverEntryPointQueryResult para);
void UPTKGraphInstantiateParamsTocudaGraphInstantiateParams(const UPTKGraphInstantiateParams * UPTK_para, cudaGraphInstantiateParams * cuda_para);
cudaGraphInstantiateResult UPTKGraphInstantiateResultTocudaGraphInstantiateResult(UPTKGraphInstantiateResult para);
cudaMemLocation UPTKMemLocationTocudaMemLocation_v2(struct UPTKMemLocation UPTK_para);
void cudaToUPTKCallback(cudaAsyncNotificationInfo_t* cudaInfo, void* wrappedUserData, cudaAsyncCallbackHandle_t cudaHandle);
UPTKAsyncNotificationType cudaAsyncNotificationTypeToUPTKAsyncNotificationType(cudaAsyncNotificationType para);
typedef struct {
  UPTKAsyncCallback userCallback;
  void* userData;
} CallbackWrapperData;
void cudaGraphExecUpdateResultInfoToUPTKGraphExecUpdateResultInfo(const cudaGraphExecUpdateResultInfo * cuda_para, UPTKGraphExecUpdateResultInfo * UPTK_para);

void UPTKGraphNodeParamsTocudaGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, cudaGraphNodeParams * cuda_para);
//void UPTKKernelNodeParamsV2TocudaKernelNodeParams(const UPTKKernelNodeParamsV2 * UPTK_para, cudaKernelNodeParams * cuda_para);
void UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(const UPTKMemcpyNodeParams * UPTK_para, cudaMemcpyNodeParams * cuda_para);
void UPTKMemsetParamsV2TocudaMemsetParams(const UPTKMemsetParamsV2 * UPTK_para, cudaMemsetParams * cuda_para);
void UPTKHostNodeParamsV2TocudaHostNodeParams(const UPTKHostNodeParamsV2 * UPTK_para, cudaHostNodeParams * cuda_para);
void UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(const UPTKChildGraphNodeParams* UPTK_para, cudaChildGraphNodeParams * cuda_para);
void UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(const UPTKEventWaitNodeParams * UPTK_para, cudaEventWaitNodeParams * cuda_para);
void UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(const UPTKEventRecordNodeParams * UPTK_para, cudaEventRecordNodeParams * cuda_para);
cudaExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsV2TocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para);
cudaExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsV2TocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para);
void UPTKMemAllocNodeParamsV2TocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, cudaMemAllocNodeParams *cuda_para);
void UPTKMemFreeNodeParamsTocudaMemFreeNodeParams(const UPTKMemFreeNodeParams * UPTK_para, cudaMemFreeNodeParams * cuda_para);
void UPTKGraphEdgeDataTocudaGraphEdgeData(const UPTKGraphEdgeData* UPTK_para, cudaGraphEdgeData* cuda_para);
void UPTKConditionalNodeParamsTocudaConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, cudaConditionalNodeParams *cuda_para);
cudaGraphConditionalNodeType UPTKConditionalNodeTypeTocudaConditionalNodeType(enum UPTKGraphConditionalNodeType para);
const char* UPTK_symbolTocuda_symbol_v2(const char *UPTK_symbol);
typedef struct {
    const char *UPTK_symbol;
    const char *cuda_symbol;
} SymbolMapping;
const char* UPTK_symbolTocuda_symbol_v2_ptsz(const char *UPTK_symbol);

// Based on the newly added application interface

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RUNTIME_HPP__





