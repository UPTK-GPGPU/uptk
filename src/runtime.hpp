// for comment out UPTK code
#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__

#define __UPTK_HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

// below is UPTK header
#include <UPTK_runtime_api.h>


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ERROR_INVALID_ENUM() do{printf("Error invalid enum. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)
#define ERROR_INVALID_OR_UNSUPPORTED_ENUM() do{printf("[ERROR] The enumeration passed in is invalid, or the functionality for the enumeration is currently not supported. Fun: %s, para: %d\n", __FUNCTION__, para); abort(); }while(0)

const int SM_VERSION_MAJOR = 7;
const int SM_VERSION_MINOR = 5;

hipError_t UPTKErrorTohipError(enum UPTKError para);
hipClusterSchedulingPolicy UPTKClusterSchedulingPolicyTohipClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para);
enum hipSynchronizationPolicy UPTKSynchronizationPolicyTohipSynchronizationPolicy(enum UPTKSynchronizationPolicy para);
hipAccessProperty UPTKAccessPropertyTohipAccessProperty(enum UPTKAccessProperty para);
void UPTKAccessPolicyWindowTohipAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, hipAccessPolicyWindow * hip_para);
enum UPTKError hipErrorToUPTKError(hipError_t para);
hipMemcpyKind UPTKMemcpyKindTohipMemcpyKind(enum UPTKMemcpyKind para);
hipStreamCaptureMode UPTKStreamCaptureModeTohipStreamCaptureMode(enum UPTKStreamCaptureMode para);
enum UPTKStreamCaptureStatus hipStreamCaptureStatusToUPTKStreamCaptureStatus(hipStreamCaptureStatus para);
void UPTKIpcMemHandleTohipIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, hipIpcMemHandle_t * hip_para);
void UPTKDevicePropTohipDeviceProp(const struct UPTKDeviceProp * UPTK_para, hipDeviceProp_t * hip_para);
void hipDeviceProp_v2ToUPTKDeviceProp(const hipDeviceProp_t_v2 * hip_para, struct UPTKDeviceProp * UPTK_para);
enum UPTKFuncCache hipFuncCacheToUPTKFuncCache(hipFuncCache_t para);
enum hipLimit_t UPTKlimitTohipLimit(UPTKlimit para);
void UPTKKernelNodeParamsTohipKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, hipKernelNodeParams * hip_para);
void UPTKPosTohipPos(const struct UPTKPos * UPTK_para, hipPos * hip_para);
void UPTKPitchedPtrTohipPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, hipPitchedPtr * hip_para);
void UPTKExtentTohipExtent(const struct UPTKExtent * UPTK_para, hipExtent * hip_para);
enum UPTKGraphExecUpdateResult hipGraphExecUpdateResultToUPTKGraphExecUpdateResult(hipGraphExecUpdateResult para);
void UPTKMemcpy3DParmsTohipMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, hipMemcpy3DParms * hip_para);
void UPTKLaunchConfig_tTohipLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, hipLaunchConfig_t * hip_para);
void UPTKLaunchAttributeValueTohipLaunchAttributeValue(const UPTKLaunchAttributeValue *UPTK_para, hipLaunchAttributeValue *hip_para, UPTKLaunchAttributeID para);
hipLaunchAttribute UPTKLaunchAttributeTohipLaunchAttribute(UPTKLaunchAttribute UPTK_para);
hipLaunchAttributeID UPTKLaunchAttributeIDTohipLaunchAttributeID(enum UPTKLaunchAttributeID para);
void hipFuncAttributesToUPTKFuncAttributes(const hipFuncAttributes * hip_para, struct UPTKFuncAttributes * UPTK_para);

UPTKresult hipErrorToUPTKresult(hipError_t para);
hipArray_Format UPTKarray_formatTohipArray_Format(UPTKarray_format para);
void UPTK_ARRAY_DESCRIPTORToHIP_ARRAY_DESCRIPTOR(const UPTK_ARRAY_DESCRIPTOR * UPTK_para, HIP_ARRAY_DESCRIPTOR * hip_para);
hiprtcJIT_option UPTKjit_optionTohiprtcJIT_option(UPTKjit_option para);
UPTKresult hiprtcResultToUPTKresult(hiprtcResult para);
hiprtcJITInputType UPTKjitInputTypeTohiprtcJITInputType(UPTKjitInputType para);
hipDeviceAttribute_t UPTKDeviceAttrTohipDeviceAttribute(enum UPTKDeviceAttr para);
enum UPTKGraphNodeType hipGraphNodeTypeToUPTKGraphNodeType(hipGraphNodeType para);
void UPTKKernelNodeParamsV2TohipKernelNodeParams(const UPTKKernelNodeParamsV2 *UPTK_para, hipKernelNodeParams *hip_para);
hipGraphNodeType UPTKGraphNodeTypeTohipGraphNodeType(enum UPTKGraphNodeType para);
void UPTKMemcpyNodeParamsTohipMemcpyNodeParams(const UPTKMemcpyNodeParams *UPTK_para, hipMemcpyNodeParams *hip_para);
void UPTKMemsetParamsV2TohipMemsetParams(const UPTKMemsetParamsV2 *UPTK_para, hipMemsetParams *hip_para);
void UPTKHostNodeParamsV2TohipHostNodeParams(const UPTKHostNodeParamsV2 *UPTK_para, hipHostNodeParams *hip_para);
void UPTKChildGraphNodeParamsTohipChildGraphNodeParams(const UPTKChildGraphNodeParams *UPTK_para, hipChildGraphNodeParams *hip_para);
void UPTKEventWaitNodeParamsTohipEventWaitNodeParams(const UPTKEventWaitNodeParams *UPTK_para, hipEventWaitNodeParams *hip_para);
void UPTKEventRecordNodeParamsTohipEventRecordNodeParams(const UPTKEventRecordNodeParams *UPTK_para, hipEventRecordNodeParams *hip_para);
hipExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTohipExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para);
hipExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTohipExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para);
hipExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsV2TohipExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para);
hipMemAllocationType UPTKMemAllocationTypeTohipMemAllocationType(enum UPTKMemAllocationType para);
hipMemAllocationHandleType UPTKMemAllocationHandleTypeTohipMemAllocationHandleType(enum UPTKMemAllocationHandleType para);
hipMemLocationType UPTKMemLocationTypeTohipMemLocationType(enum UPTKMemLocationType para);
void UPTKMemLocationTohipMemLocation(const struct UPTKMemLocation * UPTK_para, hipMemLocation * hip_para);
hipExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsV2TohipExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para);
void UPTKMemPoolPropsTohipMemPoolProps(const UPTKMemPoolProps * UPTK_para, hipMemPoolProps * hip_para);
hipMemAccessFlags UPTKMemAccessFlagsTohipMemAccessFlags(UPTKMemAccessFlags para);
hipMemAccessDesc UPTKMemAccessDescTohipMemAccessDesc(struct UPTKMemAccessDesc UPTK_para);
void UPTKMemAllocNodeParamsV2TohipMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, hipMemAllocNodeParams *hip_para);
void UPTKMemFreeNodeParamsTohipMemFreeNodeParams(const UPTKMemFreeNodeParams *UPTK_para, hipMemFreeNodeParams *hip_para);
void UPTKConditionalNodeParamsTohipConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, hipConditionalNodeParams *hip_para);
hipGraphConditionalNodeType UPTKConditionalNodeTypeTohipConditionalNodeType(enum UPTKGraphConditionalNodeType para);
void UPTKGraphNodeParamsTohipGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, hipGraphNodeParams * hip_para);
hipFunction_attribute UPTKfunction_attributeTohipFunction_attribute(UPTKfunction_attribute para);
hipFuncAttribute UPTKFuncAttributeTohipFuncAttribute(enum UPTKFuncAttribute para);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RUNTIME_HPP__





