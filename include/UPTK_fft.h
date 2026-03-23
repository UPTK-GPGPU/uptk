#ifndef _UPTKFFT_H_
#define _UPTKFFT_H_


#include "cuComplex.h"
#include "UPTK_library_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UPTKFFT_VER_MAJOR 1
#define UPTKFFT_VER_MINOR 0
#define UPTKFFT_VER_PATCH 0
#define UPTKFFT_VER_BUILD 0

#define UPTKFFT_VERSION 10000

// CUFFT API function return values
typedef enum UPTKfftResult_t {
  UPTKFFT_SUCCESS        = 0x0,
  UPTKFFT_INVALID_PLAN   = 0x1,
  UPTKFFT_ALLOC_FAILED   = 0x2,
  UPTKFFT_INVALID_TYPE   = 0x3,
  UPTKFFT_INVALID_VALUE  = 0x4,
  UPTKFFT_INTERNAL_ERROR = 0x5,
  UPTKFFT_EXEC_FAILED    = 0x6,
  UPTKFFT_SETUP_FAILED   = 0x7,
  UPTKFFT_INVALID_SIZE   = 0x8,
  UPTKFFT_UNALIGNED_DATA = 0x9,
  UPTKFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  UPTKFFT_INVALID_DEVICE = 0xB,
  UPTKFFT_PARSE_ERROR = 0xC,
  UPTKFFT_NO_WORKSPACE = 0xD,
  UPTKFFT_NOT_IMPLEMENTED = 0xE,
  UPTKFFT_LICENSE_ERROR = 0x0F,
  UPTKFFT_NOT_SUPPORTED = 0x10

} UPTKfftResult;

#define MAX_UPTKFFT_ERROR 0x11
#define UPTKFFTAPI

// CUFFT defines and supports the following data types


// UPTKfftReal is a single-precision, floating-point real data type.
// UPTKfftDoubleReal is a double-precision, real data type.
typedef float UPTKfftReal;
typedef double UPTKfftDoubleReal;

// UPTKfftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// UPTKfftDoubleComplex is the double-precision equivalent.
typedef cuComplex UPTKfftComplex;
typedef cuDoubleComplex UPTKfftDoubleComplex;

// CUFFT transform directions
#define UPTKFFT_FORWARD -1 // Forward FFT
#define UPTKFFT_INVERSE  1 // Inverse FFT

// CUFFT supports the following transform types
typedef enum UPTKfftType_t {
  UPTKFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  UPTKFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  UPTKFFT_C2C = 0x29,     // Complex to Complex, interleaved
  UPTKFFT_D2Z = 0x6a,     // Double to Double-Complex
  UPTKFFT_Z2D = 0x6c,     // Double-Complex to Double
  UPTKFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
} UPTKfftType;

// CUFFT supports the following data layouts
typedef enum UPTKfftCompatibility_t {
    UPTKFFT_COMPATIBILITY_FFTW_PADDING    = 0x01    // The default value
} UPTKfftCompatibility;

#define UPTKFFT_COMPATIBILITY_DEFAULT   UPTKFFT_COMPATIBILITY_FFTW_PADDING

//
// structure definition used by the shim between old and new APIs
//
#define MAX_SHIM_RANK 3

// UPTKfftHandle is a handle type used to store and access CUFFT plans.
// typedef int UPTKfftHandle;
// PS: UPTKfftHandle aligns hipfftHandle for DTK-CUDA.
typedef unsigned long long UPTKfftHandle;


UPTKfftResult UPTKFFTAPI UPTKfftPlan1d(UPTKfftHandle *plan,
                                 int nx,
                                 UPTKfftType type,
                                 int batch);

UPTKfftResult UPTKFFTAPI UPTKfftPlan2d(UPTKfftHandle *plan,
                                 int nx, int ny,
                                 UPTKfftType type);

UPTKfftResult UPTKFFTAPI UPTKfftPlan3d(UPTKfftHandle *plan,
                                 int nx, int ny, int nz,
                                 UPTKfftType type);

UPTKfftResult UPTKFFTAPI UPTKfftPlanMany(UPTKfftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   UPTKfftType type,
                                   int batch);

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan1d(UPTKfftHandle plan,
                                     int nx,
                                     UPTKfftType type,
                                     int batch,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan2d(UPTKfftHandle plan,
                                     int nx, int ny,
                                     UPTKfftType type,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftMakePlan3d(UPTKfftHandle plan,
                                     int nx, int ny, int nz,
                                     UPTKfftType type,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftMakePlanMany(UPTKfftHandle plan,
                                       int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       UPTKfftType type,
                                       int batch,
                                       size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftMakePlanMany64(UPTKfftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         long long int *onembed,
                                         long long int ostride, long long int odist,
                                         UPTKfftType type,
                                         long long int batch,
                                         size_t * workSize);

UPTKfftResult UPTKFFTAPI UPTKfftGetSizeMany64(UPTKfftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride, long long int idist,
                                        long long int *onembed,
                                        long long int ostride, long long int odist,
                                        UPTKfftType type,
                                        long long int batch,
                                        size_t *workSize);




UPTKfftResult UPTKFFTAPI UPTKfftEstimate1d(int nx,
                                     UPTKfftType type,
                                     int batch,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftEstimate2d(int nx, int ny,
                                     UPTKfftType type,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftEstimate3d(int nx, int ny, int nz,
                                     UPTKfftType type,
                                     size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftEstimateMany(int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       UPTKfftType type,
                                       int batch,
                                       size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftCreate(UPTKfftHandle * handle);

UPTKfftResult UPTKFFTAPI UPTKfftGetSize1d(UPTKfftHandle handle,
                                    int nx,
                                    UPTKfftType type,
                                    int batch,
                                    size_t *workSize );

UPTKfftResult UPTKFFTAPI UPTKfftGetSize2d(UPTKfftHandle handle,
                                    int nx, int ny,
                                    UPTKfftType type,
                                    size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftGetSize3d(UPTKfftHandle handle,
                                    int nx, int ny, int nz,
                                    UPTKfftType type,
                                    size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftGetSizeMany(UPTKfftHandle handle,
                                      int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      UPTKfftType type, int batch, size_t *workArea);

UPTKfftResult UPTKFFTAPI UPTKfftGetSize(UPTKfftHandle handle, size_t *workSize);

UPTKfftResult UPTKFFTAPI UPTKfftSetWorkArea(UPTKfftHandle plan, void *workArea);

UPTKfftResult UPTKFFTAPI UPTKfftSetAutoAllocation(UPTKfftHandle plan, int autoAllocate);

UPTKfftResult UPTKFFTAPI UPTKfftExecC2C(UPTKfftHandle plan,
                                  UPTKfftComplex *idata,
                                  UPTKfftComplex *odata,
                                  int direction);

UPTKfftResult UPTKFFTAPI UPTKfftExecR2C(UPTKfftHandle plan,
                                  UPTKfftReal *idata,
                                  UPTKfftComplex *odata);

UPTKfftResult UPTKFFTAPI UPTKfftExecC2R(UPTKfftHandle plan,
                                  UPTKfftComplex *idata,
                                  UPTKfftReal *odata);

UPTKfftResult UPTKFFTAPI UPTKfftExecZ2Z(UPTKfftHandle plan,
                                  UPTKfftDoubleComplex *idata,
                                  UPTKfftDoubleComplex *odata,
                                  int direction);

UPTKfftResult UPTKFFTAPI UPTKfftExecD2Z(UPTKfftHandle plan,
                                  UPTKfftDoubleReal *idata,
                                  UPTKfftDoubleComplex *odata);

UPTKfftResult UPTKFFTAPI UPTKfftExecZ2D(UPTKfftHandle plan,
                                  UPTKfftDoubleComplex *idata,
                                  UPTKfftDoubleReal *odata);


// utility functions
UPTKfftResult UPTKFFTAPI UPTKfftSetStream(UPTKfftHandle plan,
                                    UPTKStream_t stream);

UPTKfftResult UPTKFFTAPI UPTKfftDestroy(UPTKfftHandle plan);

UPTKfftResult UPTKFFTAPI UPTKfftGetVersion(int *version);

UPTKfftResult UPTKFFTAPI UPTKfftGetProperty(UPTKlibraryPropertyType type,
                                      int *value);

//
// Set/Get PlanProperty APIs configures per-plan behavior 
//
typedef enum UPTKfftProperty_t {
    UPTKFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 0x1,
    UPTKFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = 0x2,
} UPTKfftProperty;

UPTKfftResult UPTKFFTAPI UPTKfftSetPlanPropertyInt64(UPTKfftHandle plan, 
                                               UPTKfftProperty property, 
                                               const long long int inputValueInt);

UPTKfftResult UPTKFFTAPI UPTKfftGetPlanPropertyInt64(UPTKfftHandle plan, 
                                               UPTKfftProperty property, 
                                               long long int* returnPtrValue);

UPTKfftResult UPTKFFTAPI UPTKfftResetPlanProperty(UPTKfftHandle plan, UPTKfftProperty property);

#ifdef __cplusplus
}
#endif

#endif /* _UPTKFFT_H_ */
