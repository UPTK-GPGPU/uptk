#include "rand.hpp"

UPTKrandStatus_t CURANDAPI UPTKrandCreateGenerator(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type)
{
    curandRngType_t cuda_rng_type = UPTKrandRngTypeTocurandRngType(rng_type);
    curandStatus_t cuda_res;
    cuda_res = curandCreateGenerator((curandGenerator_t *)generator, cuda_rng_type);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandCreateGeneratorHost(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type)
{
    curandRngType_t cuda_rng_type = UPTKrandRngTypeTocurandRngType(rng_type);
    curandStatus_t cuda_res;
    cuda_res = curandCreateGeneratorHost((curandGenerator_t *)generator, cuda_rng_type);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandCreatePoissonDistribution(double lambda, UPTKrandDiscreteDistribution_t *discrete_distribution)
{
    curandStatus_t cuda_res;
    cuda_res = curandCreatePoissonDistribution(lambda, (curandDiscreteDistribution_t *)discrete_distribution);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandDestroyDistribution(UPTKrandDiscreteDistribution_t discrete_distribution)
{
    curandStatus_t cuda_res;
    cuda_res = curandDestroyDistribution((curandDiscreteDistribution_t)discrete_distribution);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandDestroyGenerator(UPTKrandGenerator_t generator)
{
    curandStatus_t cuda_res;
    cuda_res = curandDestroyGenerator((curandGenerator_t)generator);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerate(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerate((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateLogNormal(UPTKrandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateLogNormal((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateLogNormalDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateLogNormalDouble((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateNormal(UPTKrandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateNormal((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateNormalDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateNormalDouble((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGeneratePoisson(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda)
{
    curandStatus_t cuda_res;
    cuda_res = curandGeneratePoisson((curandGenerator_t)generator, outputPtr, n, lambda);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateSeeds(UPTKrandGenerator_t generator)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateSeeds((curandGenerator_t)generator);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateUniform(UPTKrandGenerator_t generator, float *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateUniform((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGenerateUniformDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateUniformDouble((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGetVersion(int *version)
{
    if (nullptr == version)
    {
        return UPTKRAND_STATUS_INTERNAL_ERROR;
    }

    *version = UPTKRAND_VERSION;

    return UPTKRAND_STATUS_SUCCESS;
}

UPTKrandStatus_t CURANDAPI UPTKrandSetGeneratorOffset(UPTKrandGenerator_t generator, unsigned long long offset)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetGeneratorOffset((curandGenerator_t)generator, offset);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandSetPseudoRandomGeneratorSeed(UPTKrandGenerator_t generator, unsigned long long seed)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetPseudoRandomGeneratorSeed((curandGenerator_t)generator, seed);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandSetQuasiRandomGeneratorDimensions(UPTKrandGenerator_t generator, unsigned int num_dimensions)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetQuasiRandomGeneratorDimensions((curandGenerator_t)generator, num_dimensions);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandSetStream(UPTKrandGenerator_t generator, UPTKStream_t stream)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetStream((curandGenerator_t)generator, (cudaStream_t)stream);
    return curandStatusToUPTKrandStatus(cuda_res);
}

// Not Support
UPTKrandStatus_t CURANDAPI UPTKrandGenerateLongLong(UPTKrandGenerator_t generator, unsigned long long *outputPtr, size_t num)
{
    return curandStatusToUPTKrandStatus(curandGenerateLongLong((curandGenerator_t)generator, outputPtr, num));
}

UPTKrandStatus_t CURANDAPI UPTKrandGeneratePoissonMethod(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda, UPTKrandMethod_t method)
{
    curandStatus_t cuda_res;
    cuda_res = curandGeneratePoissonMethod((curandGenerator_t)generator, outputPtr, n, lambda, UPTKrandMethodTocurandMethod(method));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGetDirectionVectors32(UPTKrandDirectionVectors32_t *vectors[], UPTKrandDirectionVectorSet_t set)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetDirectionVectors32((curandDirectionVectors32_t **)vectors, UPTKrandDirectionVectorSetTocurandDirectionVectorSet(set));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGetDirectionVectors64(UPTKrandDirectionVectors64_t *vectors[], UPTKrandDirectionVectorSet_t set)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetDirectionVectors64((curandDirectionVectors64_t **)vectors, UPTKrandDirectionVectorSetTocurandDirectionVectorSet(set));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGetProperty(libraryPropertyType type, int *value)
{
    if (nullptr == value)
    {
        return UPTKRAND_STATUS_INTERNAL_ERROR;
    }

    switch (type)
    {
    case MAJOR_VERSION:
        *value = UPTKRAND_VER_MAJOR;
        break;
    case MINOR_VERSION:
        *value = UPTKRAND_VER_MINOR;
        break;
    case PATCH_LEVEL:
        *value = UPTKRAND_VER_PATCH;
        break;
    default:
        return UPTKRAND_STATUS_TYPE_ERROR;
    }

    return UPTKRAND_STATUS_SUCCESS;
}

UPTKrandStatus_t CURANDAPI UPTKrandGetScrambleConstants32(unsigned int **constants)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetScrambleConstants32(constants);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandGetScrambleConstants64(unsigned long long **constants)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetScrambleConstants64(constants);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t CURANDAPI UPTKrandSetGeneratorOrdering(UPTKrandGenerator_t generator, UPTKrandOrdering_t order)
{
    return curandStatusToUPTKrandStatus(curandSetGeneratorOrdering((curandGenerator_t)generator, UPTKrandOrderingTocurandOrdering(order)));
}

#ifdef UPTK_NOT_SUPPORT
// cuda dynamic library Static libraries do not have this symbol
UPTKrandStatus_t CURANDAPI
UPTKrandGenerateBinomial(UPTKrandGenerator_t generator, unsigned int *outputPtr,
                         size_t num, unsigned int n, double p)
{
    Debug();
    return UPTKRAND_STATUS_NOT_IMPLEMENTED;
}

// cuda dynamic library Static libraries do not have this symbol
UPTKrandStatus_t CURANDAPI
UPTKrandGenerateBinomialMethod(UPTKrandGenerator_t generator,
                               unsigned int *outputPtr,
                               size_t num, unsigned int n, double p,
                               UPTKrandMethod_t method)
{
    Debug();
    return UPTKRAND_STATUS_NOT_IMPLEMENTED;
}
#endif