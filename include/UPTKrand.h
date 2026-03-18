#if !defined(UPTKRAND_H_)
#define UPTKRAND_H_

#include <UPTK_runtime_api.h>

#define UPTKRANDAPI

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define UPTKRAND_VER_MAJOR 1
#define UPTKRAND_VER_MINOR 0
#define UPTKRAND_VER_PATCH 0
#define UPTKRAND_VER_BUILD 0
#define UPTKRAND_VERSION (UPTKRAND_VER_MAJOR * 1000 + \
                        UPTKRAND_VER_MINOR *  100 + \
                        UPTKRAND_VER_PATCH)
/* UPTKRAND Host API datatypes */

/**
 * @{
 */

/**
 * UPTKRAND function call status types
 */
enum UPTKrandStatus {
    UPTKRAND_STATUS_SUCCESS = 0, ///< No errors
    UPTKRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    UPTKRAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not initialized
    UPTKRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
    UPTKRAND_STATUS_TYPE_ERROR = 103, ///< Generator is wrong type
    UPTKRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Length requested is not a multple of dimension
    UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision required by MRG32k3a
    UPTKRAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
    UPTKRAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
    UPTKRAND_STATUS_INITIALIZATION_FAILED = 203, ///< Initialization of CUDA failed
    UPTKRAND_STATUS_ARCH_MISMATCH = 204, ///< Architecture mismatch, GPU does not support requested feature
    UPTKRAND_STATUS_INTERNAL_ERROR = 999, ///< Internal library error
    UPTKRAND_STATUS_NOT_IMPLEMENTED = 1000 ///< Feature not implemented yet
};

/*
 * UPTKRAND function call status types
*/
/** \cond UNHIDE_TYPEDEFS */
typedef enum UPTKrandStatus UPTKrandStatus_t;
/** \endcond */

/**
 * UPTKRAND generator types
 */
enum UPTKrandRngType {
    UPTKRAND_RNG_TEST = 0,
    UPTKRAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
    UPTKRAND_RNG_PSEUDO_XORWOW = 101, ///< XORWOW pseudorandom generator
    UPTKRAND_RNG_PSEUDO_MRG32K3A = 121, ///< MRG32k3a pseudorandom generator
    UPTKRAND_RNG_PSEUDO_MTGP32 = 141, ///< Mersenne Twister MTGP32 pseudorandom generator
    UPTKRAND_RNG_PSEUDO_MT19937 = 142, ///< Mersenne Twister MT19937 pseudorandom generator
    UPTKRAND_RNG_PSEUDO_PHILOX4_32_10 = 161, ///< PHILOX-4x32-10 pseudorandom generator
    UPTKRAND_RNG_QUASI_DEFAULT = 200, ///< Default quasirandom generator
    UPTKRAND_RNG_QUASI_SOBOL32 = 201, ///< Sobol32 quasirandom generator
    UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  ///< Scrambled Sobol32 quasirandom generator
    UPTKRAND_RNG_QUASI_SOBOL64 = 203, ///< Sobol64 quasirandom generator
    UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  ///< Scrambled Sobol64 quasirandom generator
};

/*
 * UPTKRAND generator types
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum UPTKrandRngType UPTKrandRngType_t;
/** \endcond */

/**
 * UPTKRAND ordering of results in memory
 */
enum UPTKrandOrdering {
    UPTKRAND_ORDERING_PSEUDO_BEST = 100, ///< Best ordering for pseudorandom results
    UPTKRAND_ORDERING_PSEUDO_DEFAULT = 101, ///< Specific default thread sequence for pseudorandom results, same as UPTKRAND_ORDERING_PSEUDO_BEST
    UPTKRAND_ORDERING_PSEUDO_SEEDED = 102, ///< Specific seeding pattern for fast lower quality pseudorandom results
    UPTKRAND_ORDERING_PSEUDO_LEGACY = 103, ///< Specific legacy sequence for pseudorandom results, guaranteed to remain the same for all cuRAND release
    UPTKRAND_ORDERING_PSEUDO_DYNAMIC = 104, ///< Specific ordering adjusted to the device it is being executed on, provides the best performance
    UPTKRAND_ORDERING_QUASI_DEFAULT = 201 ///< Specific n-dimensional ordering for quasirandom results
};

/*
 * UPTKRAND ordering of results in memory
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum UPTKrandOrdering UPTKrandOrdering_t;
/** \endcond */

/**
 * UPTKRAND choice of direction vector set
 */
enum UPTKrandDirectionVectorSet {
    UPTKRAND_DIRECTION_VECTORS_32_JOEKUO6 = 101, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
    UPTKRAND_DIRECTION_VECTORS_64_JOEKUO6 = 103, ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104 ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
};

/*
 * UPTKRAND choice of direction vector set
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum UPTKrandDirectionVectorSet UPTKrandDirectionVectorSet_t;
/** \endcond */

/**
 * UPTKRAND array of 32-bit direction vectors
 */
/** \cond UNHIDE_TYPEDEFS */
typedef unsigned int UPTKrandDirectionVectors32_t[32];
/** \endcond */

 /**
 * UPTKRAND array of 64-bit direction vectors
 */
/** \cond UNHIDE_TYPEDEFS */
typedef unsigned long long UPTKrandDirectionVectors64_t[64];
/** \endcond **/

/**
 * UPTKRAND generator (opaque)
 */
struct UPTKrandGenerator_st;

/**
 * UPTKRAND generator
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct UPTKrandGenerator_st *UPTKrandGenerator_t;
/** \endcond */

/**
 * UPTKRAND distribution
 */
/** \cond UNHIDE_TYPEDEFS */
typedef double UPTKrandDistribution_st;
typedef UPTKrandDistribution_st *UPTKrandDistribution_t;
typedef struct UPTKrandDistributionShift_st *UPTKrandDistributionShift_t;
/** \endcond */
/**
 * UPTKRAND distribution M2
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct UPTKrandDistributionM2Shift_st *UPTKrandDistributionM2Shift_t;
typedef struct UPTKrandHistogramM2_st *UPTKrandHistogramM2_t;
typedef unsigned int UPTKrandHistogramM2K_st;
typedef UPTKrandHistogramM2K_st *UPTKrandHistogramM2K_t;
typedef UPTKrandDistribution_st UPTKrandHistogramM2V_st;
typedef UPTKrandHistogramM2V_st *UPTKrandHistogramM2V_t;

typedef struct UPTKrandDiscreteDistribution_st *UPTKrandDiscreteDistribution_t;
/** \endcond */

/*
 * UPTKRAND METHOD
 */
/** \cond UNHIDE_ENUMS */
enum UPTKrandMethod {
    UPTKRAND_CHOOSE_BEST = 0, // choose best depends on args
    UPTKRAND_ITR = 1,
    UPTKRAND_KNUTH = 2,
    UPTKRAND_HITR = 3,
    UPTKRAND_M1 = 4,
    UPTKRAND_M2 = 5,
    UPTKRAND_BINARY_SEARCH = 6,
    UPTKRAND_DISCRETE_GAUSS = 7,
    UPTKRAND_REJECTION = 8,
    UPTKRAND_DEVICE_API = 9,
    UPTKRAND_FAST_REJECTION = 10,
    UPTKRAND_3RD = 11,
    UPTKRAND_DEFINITION = 12,
    UPTKRAND_POISSON = 13
};

typedef enum UPTKrandMethod UPTKrandMethod_t;
/** \endcond */

/**
 * @}
 */

/**
 * \brief Create new random number generator.
 *
 * Creates a new random number generator of type \p rng_type
 * and returns it in \p *generator.
 *
 * Legal values for \p rng_type are:
 * - UPTKRAND_RNG_PSEUDO_DEFAULT
 * - UPTKRAND_RNG_PSEUDO_XORWOW
 * - UPTKRAND_RNG_PSEUDO_MRG32K3A
 * - UPTKRAND_RNG_PSEUDO_MTGP32
 * - UPTKRAND_RNG_PSEUDO_MT19937
 * - UPTKRAND_RNG_PSEUDO_PHILOX4_32_10
 * - UPTKRAND_RNG_QUASI_DEFAULT
 * - UPTKRAND_RNG_QUASI_SOBOL32
 * - UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL32
 * - UPTKRAND_RNG_QUASI_SOBOL64
 * - UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64
 *
 * When \p rng_type is UPTKRAND_RNG_PSEUDO_DEFAULT, the type chosen
 * is UPTKRAND_RNG_PSEUDO_XORWOW.  \n
 * When \p rng_type is UPTKRAND_RNG_QUASI_DEFAULT,
 * the type chosen is UPTKRAND_RNG_QUASI_SOBOL32.
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_XORWOW are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MRG32K3A are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MTGP32 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MT19937 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_PHILOX4_32_10 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SOBOL32 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SOBOL64 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SCRAMBBLED_SOBOL32 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * \param generator - Pointer to generator
 * \param rng_type - Type of generator to create
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \n
 * - UPTKRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
 * - UPTKRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - UPTKRAND_STATUS_SUCCESS if generator was created successfully \n
 *
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandCreateGenerator(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type);

/**
 * \brief Create new host CPU random number generator.
 *
 * Creates a new host CPU random number generator of type \p rng_type
 * and returns it in \p *generator.
 *
 * Legal values for \p rng_type are:
 * - UPTKRAND_RNG_PSEUDO_DEFAULT
 * - UPTKRAND_RNG_PSEUDO_XORWOW
 * - UPTKRAND_RNG_PSEUDO_MRG32K3A
 * - UPTKRAND_RNG_PSEUDO_MTGP32
 * - UPTKRAND_RNG_PSEUDO_MT19937
 * - UPTKRAND_RNG_PSEUDO_PHILOX4_32_10
 * - UPTKRAND_RNG_QUASI_DEFAULT
 * - UPTKRAND_RNG_QUASI_SOBOL32
 * - UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL32
 * - UPTKRAND_RNG_QUASI_SOBOL64
 * - UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64
 *
 * When \p rng_type is UPTKRAND_RNG_PSEUDO_DEFAULT, the type chosen
 * is UPTKRAND_RNG_PSEUDO_XORWOW.  \n
 * When \p rng_type is UPTKRAND_RNG_QUASI_DEFAULT,
 * the type chosen is UPTKRAND_RNG_QUASI_SOBOL32.
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_XORWOW are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MRG32K3A are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MTGP32 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_MT19937 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * * The default values for \p rng_type = UPTKRAND_RNG_PSEUDO_PHILOX4_32_10 are:
 * - \p seed = 0
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_PSEUDO_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SOBOL32 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SOBOL64 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL32 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * The default values for \p rng_type = UPTKRAND_RNG_QUASI_SCRAMBLED_SOBOL64 are:
 * - \p dimensions = 1
 * - \p offset = 0
 * - \p ordering = UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * \param generator - Pointer to generator
 * \param rng_type - Type of generator to create
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
 * - UPTKRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - UPTKRAND_STATUS_SUCCESS if generator was created successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandCreateGeneratorHost(UPTKrandGenerator_t *generator, UPTKrandRngType_t rng_type);

/**
 * \brief Destroy an existing generator.
 *
 * Destroy an existing generator and free all memory associated with its state.
 *
 * \param generator - Generator to destroy
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_SUCCESS if generator was destroyed successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandDestroyGenerator(UPTKrandGenerator_t generator);

/**
 * \brief Return the version number of the library.
 *
 * Return in \p *version the version number of the dynamically linked UPTKRAND
 * library.  The format is the same as CUDART_VERSION from the CUDA Runtime.
 * The only supported configuration is UPTKRAND version equal to CUDA Runtime
 * version.
 *
 * \param version - UPTKRAND library version
 *
 * \return
 * - UPTKRAND_STATUS_SUCCESS if the version number was successfully returned \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetVersion(int *version);

/**
* \brief Return the value of the UPTKrand property.
*
* Return in \p *value the number for the property described by \p type of the
* dynamically linked UPTKRAND library.
*
* \param type - CUDA library property
* \param value - integer value for the requested property
*
* \return
* - UPTKRAND_STATUS_SUCCESS if the property value was successfully returned \n
* - UPTKRAND_STATUS_OUT_OF_RANGE if the property type is not recognized \n
*/
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetProperty(libraryPropertyType type, int *value);


/**
 * \brief Set the current stream for UPTKRAND kernel launches.
 *
 * Set the current stream for UPTKRAND kernel launches.  All library functions
 * will use this stream until set again.
 *
 * \param generator - Generator to modify
 * \param stream - Stream to use or ::NULL for null stream
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_SUCCESS if stream was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandSetStream(UPTKrandGenerator_t generator, UPTKStream_t stream);

/**
 * \brief Set the seed value of the pseudo-random number generator.
 *
 * Set the seed value of the pseudorandom number generator.
 * All values of seed are valid.  Different seeds will produce different sequences.
 * Different seeds will often not be statistically correlated with each other,
 * but some pairs of seed values may generate sequences which are statistically correlated.
 *
 * \param generator - Generator to modify
 * \param seed - Seed value
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the generator is not a pseudorandom number generator \n
 * - UPTKRAND_STATUS_SUCCESS if generator seed was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandSetPseudoRandomGeneratorSeed(UPTKrandGenerator_t generator, unsigned long long seed);

/**
 * \brief Set the absolute offset of the pseudo or quasirandom number generator.
 *
 * Set the absolute offset of the pseudo or quasirandom number generator.
 *
 * All values of offset are valid.  The offset position is absolute, not
 * relative to the current position in the sequence.
 *
 * \param generator - Generator to modify
 * \param offset - Absolute offset position
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_SUCCESS if generator offset was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandSetGeneratorOffset(UPTKrandGenerator_t generator, unsigned long long offset);

/**
 * \brief Set the ordering of results of the pseudo or quasirandom number generator.
 *
 * Set the ordering of results of the pseudo or quasirandom number generator.
 *
 * Legal values of \p order for pseudorandom generators are:
 * - UPTKRAND_ORDERING_PSEUDO_DEFAULT
 * - UPTKRAND_ORDERING_PSEUDO_BEST
 * - UPTKRAND_ORDERING_PSEUDO_SEEDED
 * - UPTKRAND_ORDERING_PSEUDO_LEGACY
 *
 * Legal values of \p order for quasirandom generators are:
 * - UPTKRAND_ORDERING_QUASI_DEFAULT
 *
 * \param generator - Generator to modify
 * \param order - Ordering of results
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_OUT_OF_RANGE if the ordering is not valid \n
 * - UPTKRAND_STATUS_SUCCESS if generator ordering was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandSetGeneratorOrdering(UPTKrandGenerator_t generator, UPTKrandOrdering_t order);

/**
 * \brief Set the number of dimensions.
 *
 * Set the number of dimensions to be generated by the quasirandom number
 * generator.
 *
 * Legal values for \p num_dimensions are 1 to 20000.
 *
 * \param generator - Generator to modify
 * \param num_dimensions - Number of dimensions
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_OUT_OF_RANGE if num_dimensions is not valid \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the generator is not a quasirandom number generator \n
 * - UPTKRAND_STATUS_SUCCESS if generator ordering was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandSetQuasiRandomGeneratorDimensions(UPTKrandGenerator_t generator, unsigned int num_dimensions);

/**
 * \brief Generate 32-bit pseudo or quasirandom numbers.
 *
 * Use \p generator to generate \p num 32-bit results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 32-bit values with every bit random.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param num - Number of random 32-bit values to generate
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *     a previous kernel launch \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the generator is a 64 bit quasirandom generator.
 * (use ::UPTKrandGenerateLongLong() with 64 bit quasirandom generators)
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerate(UPTKrandGenerator_t generator, unsigned int *outputPtr, size_t num);

/**
 * \brief Generate 64-bit quasirandom numbers.
 *
 * Use \p generator to generate \p num 64-bit results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 64-bit values with every bit random.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param num - Number of random 64-bit values to generate
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *     a previous kernel launch \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_TYPE_ERROR if the generator is not a 64 bit quasirandom generator\n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateLongLong(UPTKrandGenerator_t generator, unsigned long long *outputPtr, size_t num);

/**
 * \brief Generate uniformly distributed floats.
 *
 * Use \p generator to generate \p num float results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 32-bit floating point values between \p 0.0f and \p 1.0f,
 * excluding \p 0.0f and including \p 1.0f.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param num - Number of floats to generate
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateUniform(UPTKrandGenerator_t generator, float *outputPtr, size_t num);

/**
 * \brief Generate uniformly distributed doubles.
 *
 * Use \p generator to generate \p num double results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 64-bit double precision floating point values between
 * \p 0.0 and \p 1.0, excluding \p 0.0 and including \p 1.0.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param num - Number of doubles to generate
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension \n
 * - UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateUniformDouble(UPTKrandGenerator_t generator, double *outputPtr, size_t num);

/**
 * \brief Generate normally distributed doubles.
 *
 * Use \p generator to generate \p n float results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 32-bit floating point values with mean \p mean and standard
 * deviation \p stddev.
 *
 * Normally distributed results are generated from pseudorandom generators
 * with a Box-Muller transform, and so require \p n to be even.
 * Quasirandom generators use an inverse cumulative distribution
 * function to preserve dimensionality.
 *
 * There may be slight numerical differences between results generated
 * on the GPU with generators created with ::UPTKrandCreateGenerator()
 * and results calculated on the CPU with generators created with
 * ::UPTKrandCreateGeneratorHost().  These differences arise because of
 * differences in results for transcendental functions.  In addition,
 * future versions of UPTKRAND may use newer versions of the CUDA math
 * library, so different versions of UPTKRAND may give slightly different
 * numerical values.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param n - Number of floats to generate
 * \param mean - Mean of normal distribution
 * \param stddev - Standard deviation of normal distribution
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension, or is not a multiple
 *    of two for pseudorandom generators \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateNormal(UPTKrandGenerator_t generator, float *outputPtr,
                     size_t n, float mean, float stddev);

/**
 * \brief Generate normally distributed doubles.
 *
 * Use \p generator to generate \p n double results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 64-bit floating point values with mean \p mean and standard
 * deviation \p stddev.
 *
 * Normally distributed results are generated from pseudorandom generators
 * with a Box-Muller transform, and so require \p n to be even.
 * Quasirandom generators use an inverse cumulative distribution
 * function to preserve dimensionality.
 *
 * There may be slight numerical differences between results generated
 * on the GPU with generators created with ::UPTKrandCreateGenerator()
 * and results calculated on the CPU with generators created with
 * ::UPTKrandCreateGeneratorHost().  These differences arise because of
 * differences in results for transcendental functions.  In addition,
 * future versions of UPTKRAND may use newer versions of the CUDA math
 * library, so different versions of UPTKRAND may give slightly different
 * numerical values.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param n - Number of doubles to generate
 * \param mean - Mean of normal distribution
 * \param stddev - Standard deviation of normal distribution
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension, or is not a multiple
 *    of two for pseudorandom generators \n
 * - UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateNormalDouble(UPTKrandGenerator_t generator, double *outputPtr,
                     size_t n, double mean, double stddev);

/**
 * \brief Generate log-normally distributed floats.
 *
 * Use \p generator to generate \p n float results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 32-bit floating point values with log-normal distribution based on
 * an associated normal distribution with mean \p mean and standard deviation \p stddev.
 *
 * Normally distributed results are generated from pseudorandom generators
 * with a Box-Muller transform, and so require \p n to be even.
 * Quasirandom generators use an inverse cumulative distribution
 * function to preserve dimensionality.
 * The normally distributed results are transformed into log-normal distribution.
 *
 * There may be slight numerical differences between results generated
 * on the GPU with generators created with ::UPTKrandCreateGenerator()
 * and results calculated on the CPU with generators created with
 * ::UPTKrandCreateGeneratorHost().  These differences arise because of
 * differences in results for transcendental functions.  In addition,
 * future versions of UPTKRAND may use newer versions of the CUDA math
 * library, so different versions of UPTKRAND may give slightly different
 * numerical values.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param n - Number of floats to generate
 * \param mean - Mean of associated normal distribution
 * \param stddev - Standard deviation of associated normal distribution
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension, or is not a multiple
 *    of two for pseudorandom generators \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateLogNormal(UPTKrandGenerator_t generator, float *outputPtr,
                     size_t n, float mean, float stddev);

/**
 * \brief Generate log-normally distributed doubles.
 *
 * Use \p generator to generate \p n double results into the device memory at
 * \p outputPtr.  The device memory must have been previously allocated and be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 64-bit floating point values with log-normal distribution based on
 * an associated normal distribution with mean \p mean and standard deviation \p stddev.
 *
 * Normally distributed results are generated from pseudorandom generators
 * with a Box-Muller transform, and so require \p n to be even.
 * Quasirandom generators use an inverse cumulative distribution
 * function to preserve dimensionality.
 * The normally distributed results are transformed into log-normal distribution.
 *
 * There may be slight numerical differences between results generated
 * on the GPU with generators created with ::UPTKrandCreateGenerator()
 * and results calculated on the CPU with generators created with
 * ::UPTKrandCreateGeneratorHost().  These differences arise because of
 * differences in results for transcendental functions.  In addition,
 * future versions of UPTKRAND may use newer versions of the CUDA math
 * library, so different versions of UPTKRAND may give slightly different
 * numerical values.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param n - Number of doubles to generate
 * \param mean - Mean of normal distribution
 * \param stddev - Standard deviation of normal distribution
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension, or is not a multiple
 *    of two for pseudorandom generators \n
 * - UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateLogNormalDouble(UPTKrandGenerator_t generator, double *outputPtr,
                     size_t n, double mean, double stddev);

/**
 * \brief Construct the histogram array for a Poisson distribution.
 *
 * Construct the histogram array for the Poisson distribution with lambda \p lambda.
 * For lambda greater than 2000, an approximation with a normal distribution is used.
 *
 * \param lambda - lambda for the Poisson distribution
 *
 *
 * \param discrete_distribution - pointer to the histogram in device memory
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU does not support double precision \n
 * - UPTKRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the distribution pointer was null \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000 \n
 * - UPTKRAND_STATUS_SUCCESS if the histogram was generated successfully \n
 */

UPTKrandStatus_t UPTKRANDAPI
UPTKrandCreatePoissonDistribution(double lambda, UPTKrandDiscreteDistribution_t *discrete_distribution);



/**
 * \brief Destroy the histogram array for a discrete distribution (e.g. Poisson).
 *
 * Destroy the histogram array for a discrete distribution created by UPTKrandCreatePoissonDistribution.
 *
 * \param discrete_distribution - pointer to device memory where the histogram is stored
 *
 * \return
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the histogram was never created \n
 * - UPTKRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandDestroyDistribution(UPTKrandDiscreteDistribution_t discrete_distribution);


/**
 * \brief Generate Poisson-distributed unsigned ints.
 *
 * Use \p generator to generate \p n unsigned int results into device memory at
 * \p outputPtr.  The device memory must have been previously allocated and must be
 * large enough to hold all the results.  Launches are done with the stream
 * set using ::UPTKrandSetStream(), or the null stream if no stream has been set.
 *
 * Results are 32-bit unsigned int point values with Poisson distribution, with lambda \p lambda.
 *
 * \param generator - Generator to use
 * \param outputPtr - Pointer to device memory to store CUDA-generated results, or
 *                 Pointer to host memory to store CPU-generated results
 * \param n - Number of unsigned ints to generate
 * \param lambda - lambda for the Poisson distribution
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *    a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_LENGTH_NOT_MULTIPLE if the number of output samples is
 *    not a multiple of the quasirandom dimension\n
 * - UPTKRAND_STATUS_DOUBLE_PRECISION_REQUIRED if the GPU or sm does not support double precision \n
 * - UPTKRAND_STATUS_OUT_OF_RANGE if lambda is non-positive or greater than 400,000 \n
 * - UPTKRAND_STATUS_SUCCESS if the results were generated successfully \n
 */

UPTKrandStatus_t UPTKRANDAPI
UPTKrandGeneratePoisson(UPTKrandGenerator_t generator, unsigned int *outputPtr,
                     size_t n, double lambda);
// just for internal usage
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGeneratePoissonMethod(UPTKrandGenerator_t generator, unsigned int *outputPtr,
                     size_t n, double lambda, UPTKrandMethod_t method);

/**
 * \brief Setup starting states.
 *
 * Generate the starting state of the generator.  This function is
 * automatically called by generation functions such as
 * ::UPTKrandGenerate() and ::UPTKrandGenerateUniform().
 * It can be called manually for performance testing reasons to separate
 * timings for starting state generation and random number generation.
 *
 * \param generator - Generator to update
 *
 * \return
 * - UPTKRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - UPTKRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - UPTKRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *     a previous kernel launch \n
 * - UPTKRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - UPTKRAND_STATUS_SUCCESS if the seeds were generated successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGenerateSeeds(UPTKrandGenerator_t generator);

/**
 * \brief Get direction vectors for 32-bit quasirandom number generation.
 *
 * Get a pointer to an array of direction vectors that can be used
 * for quasirandom number generation.  The resulting pointer will
 * reference an array of direction vectors in host memory.
 *
 * The array contains vectors for many dimensions.  Each dimension
 * has 32 vectors.  Each individual vector is an unsigned int.
 *
 * Legal values for \p set are:
 * - UPTKRAND_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
 * - UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 (20,000 dimensions)
 *
 * \param vectors - Address of pointer in which to return direction vectors
 * \param set - Which set of direction vectors to use
 *
 * \return
 * - UPTKRAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \n
 * - UPTKRAND_STATUS_SUCCESS if the pointer was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetDirectionVectors32(UPTKrandDirectionVectors32_t *vectors[], UPTKrandDirectionVectorSet_t set);

/**
 * \brief Get scramble constants for 32-bit scrambled Sobol' .
 *
 * Get a pointer to an array of scramble constants that can be used
 * for quasirandom number generation.  The resulting pointer will
 * reference an array of unsinged ints in host memory.
 *
 * The array contains constants for many dimensions.  Each dimension
 * has a single unsigned int constant.
 *
 * \param constants - Address of pointer in which to return scramble constants
 *
 * \return
 * - UPTKRAND_STATUS_SUCCESS if the pointer was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetScrambleConstants32(unsigned int * * constants);

/**
 * \brief Get direction vectors for 64-bit quasirandom number generation.
 *
 * Get a pointer to an array of direction vectors that can be used
 * for quasirandom number generation.  The resulting pointer will
 * reference an array of direction vectors in host memory.
 *
 * The array contains vectors for many dimensions.  Each dimension
 * has 64 vectors.  Each individual vector is an unsigned long long.
 *
 * Legal values for \p set are:
 * - UPTKRAND_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
 * - UPTKRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 (20,000 dimensions)
 *
 * \param vectors - Address of pointer in which to return direction vectors
 * \param set - Which set of direction vectors to use
 *
 * \return
 * - UPTKRAND_STATUS_OUT_OF_RANGE if the choice of set is invalid \n
 * - UPTKRAND_STATUS_SUCCESS if the pointer was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetDirectionVectors64(UPTKrandDirectionVectors64_t *vectors[], UPTKrandDirectionVectorSet_t set);

/**
 * \brief Get scramble constants for 64-bit scrambled Sobol' .
 *
 * Get a pointer to an array of scramble constants that can be used
 * for quasirandom number generation.  The resulting pointer will
 * reference an array of unsinged long longs in host memory.
 *
 * The array contains constants for many dimensions.  Each dimension
 * has a single unsigned long long constant.
 *
 * \param constants - Address of pointer in which to return scramble constants
 *
 * \return
 * - UPTKRAND_STATUS_SUCCESS if the pointer was set successfully \n
 */
UPTKrandStatus_t UPTKRANDAPI
UPTKrandGetScrambleConstants64(unsigned long long * * constants);

/** @} */

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(UPTKRAND_H_) */
