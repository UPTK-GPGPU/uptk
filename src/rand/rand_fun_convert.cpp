#include "rand.hpp"

UPTKrandStatus_t UPTKRANDAPI UPTKrandCreateGenerator(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type)
{
    curandRngType_t cuda_rng_type = UPTKrandRngTypeTocurandRngType(rng_type);
    curandStatus_t cuda_res;
    cuda_res = curandCreateGenerator((curandGenerator_t *)generator, cuda_rng_type);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandCreateGeneratorHost(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type)
{
    curandRngType_t cuda_rng_type = UPTKrandRngTypeTocurandRngType(rng_type);
    curandStatus_t cuda_res;
    cuda_res = curandCreateGeneratorHost((curandGenerator_t *)generator, cuda_rng_type);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandCreatePoissonDistribution(double lambda, UPTKrandDiscreteDistribution_t *discrete_distribution)
{
    curandStatus_t cuda_res;
    cuda_res = curandCreatePoissonDistribution(lambda, (curandDiscreteDistribution_t *)discrete_distribution);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandDestroyDistribution(UPTKrandDiscreteDistribution_t discrete_distribution)
{
    curandStatus_t cuda_res;
    cuda_res = curandDestroyDistribution((curandDiscreteDistribution_t)discrete_distribution);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandDestroyGenerator(UPTKrandGenerator_t generator)
{
    curandStatus_t cuda_res;
    cuda_res = curandDestroyGenerator((curandGenerator_t)generator);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerate(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerate((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateLogNormal(UPTKrandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateLogNormal((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateLogNormalDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateLogNormalDouble((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateNormal(UPTKrandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateNormal((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateNormalDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateNormalDouble((curandGenerator_t)generator, outputPtr, n, mean, stddev);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGeneratePoisson(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda)
{
    curandStatus_t cuda_res;
    cuda_res = curandGeneratePoisson((curandGenerator_t)generator, outputPtr, n, lambda);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateSeeds(UPTKrandGenerator_t generator)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateSeeds((curandGenerator_t)generator);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateUniform(UPTKrandGenerator_t generator, float *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateUniform((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateUniformDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t num)
{
    curandStatus_t cuda_res;
    cuda_res = curandGenerateUniformDouble((curandGenerator_t)generator, outputPtr, num);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetVersion(int *version)
{
    if (nullptr == version)
    {
        return UPTKRAND_STATUS_INTERNAL_ERROR;
    }

    *version = UPTKRAND_VERSION;

    return UPTKRAND_STATUS_SUCCESS;
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandSetGeneratorOffset(UPTKrandGenerator_t generator, unsigned long long offset)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetGeneratorOffset((curandGenerator_t)generator, offset);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandSetPseudoRandomGeneratorSeed(UPTKrandGenerator_t generator, unsigned long long seed)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetPseudoRandomGeneratorSeed((curandGenerator_t)generator, seed);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandSetQuasiRandomGeneratorDimensions(UPTKrandGenerator_t generator, unsigned int num_dimensions)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetQuasiRandomGeneratorDimensions((curandGenerator_t)generator, num_dimensions);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandSetStream(UPTKrandGenerator_t generator, UPTKStream_t stream)
{
    curandStatus_t cuda_res;
    cuda_res = curandSetStream((curandGenerator_t)generator, (cudaStream_t)stream);
    return curandStatusToUPTKrandStatus(cuda_res);
}

// Not Support
UPTKrandStatus_t UPTKRANDAPI UPTKrandGenerateLongLong(UPTKrandGenerator_t generator, unsigned long long *outputPtr, size_t num)
{
    return curandStatusToUPTKrandStatus(curandGenerateLongLong((curandGenerator_t)generator, outputPtr, num));
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGeneratePoissonMethod(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda, UPTKrandMethod_t method)
{
    curandStatus_t cuda_res;
    cuda_res = curandGeneratePoissonMethod((curandGenerator_t)generator, outputPtr, n, lambda, UPTKrandMethodTocurandMethod(method));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetDirectionVectors32(UPTKrandDirectionVectors32_t *vectors[], UPTKrandDirectionVectorSet_t set)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetDirectionVectors32((curandDirectionVectors32_t **)vectors, UPTKrandDirectionVectorSetTocurandDirectionVectorSet(set));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetDirectionVectors64(UPTKrandDirectionVectors64_t *vectors[], UPTKrandDirectionVectorSet_t set)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetDirectionVectors64((curandDirectionVectors64_t **)vectors, UPTKrandDirectionVectorSetTocurandDirectionVectorSet(set));
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetProperty(libraryPropertyType type, int *value)
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

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetScrambleConstants32(unsigned int **constants)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetScrambleConstants32(constants);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandGetScrambleConstants64(unsigned long long **constants)
{
    curandStatus_t cuda_res;
    cuda_res = curandGetScrambleConstants64(constants);
    return curandStatusToUPTKrandStatus(cuda_res);
}

UPTKrandStatus_t UPTKRANDAPI UPTKrandSetGeneratorOrdering(UPTKrandGenerator_t generator, UPTKrandOrdering_t order)
{
    return curandStatusToUPTKrandStatus(curandSetGeneratorOrdering((curandGenerator_t)generator, UPTKrandOrderingTocurandOrdering(order)));
}

#ifdef UPTK_NOT_SUPPORT
// cuda dynamic library Static libraries do not have this symbol
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateBinomial(UPTKrandGenerator_t generator, unsigned int *outputPtr,
                         size_t num, unsigned int n, double p)
{
    Debug();
    return UPTKRAND_STATUS_NOT_IMPLEMENTED;
}

// cuda dynamic library Static libraries do not have this symbol
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateBinomialMethod(UPTKrandGenerator_t generator,
                               unsigned int *outputPtr,
                               size_t num, unsigned int n, double p,
                               UPTKrandMethod_t method)
{
    Debug();
    return UPTKRAND_STATUS_NOT_IMPLEMENTED;
}
#endif