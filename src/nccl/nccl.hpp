#ifndef __FFT_HPP__
#define __FFT_HPP__

#include "../runtime/runtime.hpp"
#include <UPTK_runtime_api.h>
#include <nccl.h>
// below is cuda header
#include <UPTK_nccl.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

ncclResult_t UPTKncclResultToncclResult(UPTKncclResult_t para);
UPTKncclResult_t ncclResultToUPTKncclResult(ncclResult_t para);

ncclRedOp_dummy_t UPTKncclRedOp_dummyToncclRedOp_dummy(UPTKncclRedOp_dummy_t para);

ncclRedOp_t UPTKncclRedOpToncclRedOp(UPTKncclRedOp_t para);

ncclDataType_t UPTKncclDataTypeToncclDataType(UPTKncclDataType_t para);

ncclScalarResidence_t UPTKncclScalarResidenceToncclScalarResidence(UPTKncclScalarResidence_t para);

void UPTKncclUniqueIdToncclUniqueId(const UPTKncclUniqueId * UPTK_para, ncclUniqueId * cuda_para);
void ncclUniqueIdToUPTKncclUniqueId(const ncclUniqueId * cuda_para, UPTKncclUniqueId * UPTK_para);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __FFT_HPP__
