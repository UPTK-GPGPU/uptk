#ifndef __RAND_HPP__
#define __RAND_HPP__

#include "../runtime/runtime.hpp"

#include <curand.h>
#include <UPTK_rand.h>


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
* UPTKrand status convert function
*/
curandMethod_t UPTKrandMethodTocurandMethod(UPTKrandMethod_t para);

curandStatus_t UPTKrandStatusTocurandStatus(UPTKrandStatus_t para);
UPTKrandStatus_t curandStatusToUPTKrandStatus(curandStatus_t para);

curandDirectionVectorSet_t UPTKrandDirectionVectorSetTocurandDirectionVectorSet(UPTKrandDirectionVectorSet_t para);
UPTKrandDirectionVectorSet_t curandDirectionVectorSetToUPTKrandDirectionVectorSet(curandDirectionVectorSet_t para);

curandOrdering_t UPTKrandOrderingTocurandOrdering(UPTKrandOrdering_t para);

curandRngType_t UPTKrandRngTypeTocurandRngType(UPTKrandRngType_t para);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // __RAND_HPP__
