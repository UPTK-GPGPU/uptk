#include "rand.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

curandMethod_t UPTKrandMethodTocurandMethod(UPTKrandMethod_t para) {
    switch (para) {
        case UPTKRAND_3RD:
            return CURAND_3RD;
        case UPTKRAND_BINARY_SEARCH:
            return CURAND_BINARY_SEARCH;
        case UPTKRAND_CHOOSE_BEST:
            return CURAND_CHOOSE_BEST;
        case UPTKRAND_DEFINITION:
            return CURAND_DEFINITION;
        case UPTKRAND_DEVICE_API:
            return CURAND_DEVICE_API;
        case UPTKRAND_DISCRETE_GAUSS:
            return CURAND_DISCRETE_GAUSS;
        case UPTKRAND_FAST_REJECTION:
            return CURAND_FAST_REJECTION;
        case UPTKRAND_HITR:
            return CURAND_HITR;
        case UPTKRAND_ITR:
            return CURAND_ITR;
        case UPTKRAND_KNUTH:
            return CURAND_KNUTH;
        case UPTKRAND_M1:
            return CURAND_M1;
        case UPTKRAND_M2:
            return CURAND_M2;
        case UPTKRAND_POISSON:
            return CURAND_POISSON;
        case UPTKRAND_REJECTION:
            return CURAND_REJECTION;
        default:
            ERROR_INVALID_ENUM();
    }
}

curandDirectionVectorSet_t UPTKrandDirectionVectorSetTocurandDirectionVectorSet(UPTKrandDirectionVectorSet_t para) {
    switch (para) {
        case UPTKRAND_DIRECTION_VECTORS_32_JOEKUO6:
            return CURAND_DIRECTION_VECTORS_32_JOEKUO6;
        case UPTKRAND_DIRECTION_VECTORS_64_JOEKUO6:
            return CURAND_DIRECTION_VECTORS_64_JOEKUO6;
        case UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            return CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
        case UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            return CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKrandDirectionVectorSet_t curandDirectionVectorSetToUPTKrandDirectionVectorSet(curandDirectionVectorSet_t para) {
    switch (para) {
        case CURAND_DIRECTION_VECTORS_32_JOEKUO6:
            return UPTKRAND_DIRECTION_VECTORS_32_JOEKUO6;
        case CURAND_DIRECTION_VECTORS_64_JOEKUO6:
            return UPTKRAND_DIRECTION_VECTORS_64_JOEKUO6;
        case CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            return UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
        case CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            return UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
        default:
            ERROR_INVALID_ENUM();
    }
}

curandOrdering_t UPTKrandOrderingTocurandOrdering(UPTKrandOrdering_t para) {
    switch (para) {
        case UPTKRAND_ORDERING_PSEUDO_BEST:
            return CURAND_ORDERING_PSEUDO_BEST;
        case UPTKRAND_ORDERING_PSEUDO_DEFAULT:
            return CURAND_ORDERING_PSEUDO_DEFAULT;
        case UPTKRAND_ORDERING_PSEUDO_SEEDED:
            return CURAND_ORDERING_PSEUDO_SEEDED;
        case UPTKRAND_ORDERING_QUASI_DEFAULT:
            return CURAND_ORDERING_QUASI_DEFAULT;
        case UPTKRAND_ORDERING_PSEUDO_LEGACY:
            return CURAND_ORDERING_PSEUDO_LEGACY;
        case UPTKRAND_ORDERING_PSEUDO_DYNAMIC:
            return CURAND_ORDERING_PSEUDO_DYNAMIC;
        default:
            ERROR_INVALID_ENUM();
    }
}

curandRngType_t UPTKrandRngTypeTocurandRngType(UPTKrandRngType_t para) {
    switch (para) {
        case UPTKRAND_RNG_PSEUDO_DEFAULT:
            return CURAND_RNG_PSEUDO_DEFAULT;
        case UPTKRAND_RNG_PSEUDO_MRG32K3A:
            return CURAND_RNG_PSEUDO_MRG32K3A;
        case UPTKRAND_RNG_PSEUDO_MT19937:
            return CURAND_RNG_PSEUDO_MT19937;
        case UPTKRAND_RNG_PSEUDO_MTGP32:
            return CURAND_RNG_PSEUDO_MTGP32;
        case UPTKRAND_RNG_PSEUDO_PHILOX4_32_10:
            return CURAND_RNG_PSEUDO_PHILOX4_32_10;
        case UPTKRAND_RNG_PSEUDO_XORWOW:
            return CURAND_RNG_PSEUDO_XORWOW;
        case UPTKRAND_RNG_QUASI_DEFAULT:
            return CURAND_RNG_QUASI_DEFAULT;
        case UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
            return CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        case UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
            return CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        case UPTKRAND_RNG_QUASI_SOBOL32:
            return CURAND_RNG_QUASI_SOBOL32;
        case UPTKRAND_RNG_QUASI_SOBOL64:
            return CURAND_RNG_QUASI_SOBOL64;
        case UPTKRAND_RNG_TEST:
            return CURAND_RNG_TEST;
        default:
            ERROR_INVALID_ENUM();
    }
}

curandStatus_t UPTKrandStatusTocurandStatus(UPTKrandStatus_t para) {
    switch (para) {
        case UPTKRAND_STATUS_ALLOCATION_FAILED:
            return CURAND_STATUS_ALLOCATION_FAILED;
        case UPTKRAND_STATUS_ARCH_MISMATCH:
            return CURAND_STATUS_ARCH_MISMATCH;
        case UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case UPTKRAND_STATUS_INITIALIZATION_FAILED:
            return CURAND_STATUS_INITIALIZATION_FAILED;
        case UPTKRAND_STATUS_INTERNAL_ERROR:
            return CURAND_STATUS_INTERNAL_ERROR;
        case UPTKRAND_STATUS_LAUNCH_FAILURE:
            return CURAND_STATUS_LAUNCH_FAILURE;
        case UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE:
            return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
        case UPTKRAND_STATUS_NOT_INITIALIZED:
            return CURAND_STATUS_NOT_INITIALIZED;
        case UPTKRAND_STATUS_OUT_OF_RANGE:
            return CURAND_STATUS_OUT_OF_RANGE;
        case UPTKRAND_STATUS_PREEXISTING_FAILURE:
            return CURAND_STATUS_PREEXISTING_FAILURE;
        case UPTKRAND_STATUS_SUCCESS:
            return CURAND_STATUS_SUCCESS;
        case UPTKRAND_STATUS_TYPE_ERROR:
            return CURAND_STATUS_TYPE_ERROR;
        case UPTKRAND_STATUS_VERSION_MISMATCH:
            return CURAND_STATUS_VERSION_MISMATCH;
        case UPTKRAND_STATUS_NOT_IMPLEMENTED:    // 1000
            return CURAND_STATUS_NOT_IMPLEMENTED;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKrandStatus_t curandStatusToUPTKrandStatus(curandStatus_t para) {
    switch (para) {
        case CURAND_STATUS_ALLOCATION_FAILED:
            return UPTKRAND_STATUS_ALLOCATION_FAILED;
        case CURAND_STATUS_ARCH_MISMATCH:
            return UPTKRAND_STATUS_ARCH_MISMATCH;
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return UPTKRAND_STATUS_INITIALIZATION_FAILED;
        case CURAND_STATUS_INTERNAL_ERROR:
            return UPTKRAND_STATUS_INTERNAL_ERROR;
        case CURAND_STATUS_LAUNCH_FAILURE:
            return UPTKRAND_STATUS_LAUNCH_FAILURE;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case CURAND_STATUS_NOT_INITIALIZED:
            return UPTKRAND_STATUS_NOT_INITIALIZED;
        case CURAND_STATUS_OUT_OF_RANGE:
            return UPTKRAND_STATUS_OUT_OF_RANGE;
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return UPTKRAND_STATUS_PREEXISTING_FAILURE;
        case CURAND_STATUS_SUCCESS:
            return UPTKRAND_STATUS_SUCCESS;
        case CURAND_STATUS_TYPE_ERROR:
            return UPTKRAND_STATUS_TYPE_ERROR;
        case CURAND_STATUS_VERSION_MISMATCH:
            return UPTKRAND_STATUS_VERSION_MISMATCH;
        case CURAND_STATUS_NOT_IMPLEMENTED:    // 1000
            return UPTKRAND_STATUS_NOT_IMPLEMENTED;
        default:
            ERROR_INVALID_ENUM();
    }
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */