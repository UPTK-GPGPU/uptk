/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UPTK_NCCL_H_
#define UPTK_NCCL_H_

#include <UPTK_runtime_api.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#define UPTKNCCLAPI
#define UPTK_NCCL_MAJOR 1
#define UPTK_NCCL_MINOR 0
#define UPTK_NCCL_PATCH 0
#define UPTK_NCCL_SUFFIX ""

#define UPTK_NCCL_VERSION_CODE 10000 
#define UPTK_NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#define UPTK_RCCL_BFLOAT16 1
#define UPTK_RCCL_FLOAT8 1
#define UPTK_RCCL_GATHER_SCATTER 1
#define UPTK_RCCL_ALLTOALLV 1

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/*! @brief      Opaque handle to communicator
    @details    A communicator contains information required to facilitate collective communications calls */
typedef struct UPTKncclComm* UPTKncclComm_t;
#define UPTK_NCCL_COMM_NULL NULL

#define UPTK_NCCL_UNIQUE_ID_BYTES 128
/*! @brief      Opaque unique id used to initialize communicators
    @details    The UPTKncclUniqueId must be passed to all participating ranks */
typedef struct { char internal[UPTK_NCCL_UNIQUE_ID_BYTES]; /*!< Opaque array>*/} UPTKncclUniqueId;

/*! @defgroup   rccl_result_code Result Codes
    @details    The various result codes that RCCL API calls may return
    @{ */

/*! @brief      Result type
    @details    Return codes aside from UPTKncclSuccess indicate that a call has failed */
  typedef enum {
    UPTKncclSuccess                 =  0, /*!< No error */
    UPTKncclUnhandledCudaError      =  1, /*!< Unhandled HIP error */
    UPTKncclSystemError             =  2, /*!< Unhandled system error */
    UPTKncclInternalError           =  3, /*!< Internal Error - Please report to RCCL developers */
    UPTKncclInvalidArgument         =  4, /*!< Invalid argument */
    UPTKncclInvalidUsage            =  5, /*!< Invalid usage */
    UPTKncclRemoteError             =  6, /*!< Remote process exited or there was a network error */
    UPTKncclInProgress              =  7, /*!< RCCL operation in progress */
    UPTKncclNumResults              =  8  /*!< Number of result types */
  } UPTKncclResult_t;
/*! @} */

#define UPTK_NCCL_CONFIG_UNDEF_INT INT_MIN
#define UPTK_NCCL_CONFIG_UNDEF_PTR NULL
#define UPTK_NCCL_SPLIT_NOCOLOR -1
#define UPTK_NCCL_UNDEF_FLOAT -1.0f

/*! @defgroup   rccl_config_type Communicator Configuration
    @details    Structure that allows for customizing Communicator behavior via ncclCommInitRankConfig
    @{ */

/*! @brief      Communicator configuration
    @details    Users can assign value to attributes to specify the behavior of a communicator */
typedef struct UPTKncclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;                 /*!< Should not be touched */
  unsigned int magic;          /*!< Should not be touched */
  unsigned int version;        /*!< Should not be touched */
  /* attributes that users are able to customize. */
  int blocking;                /*!< Whether or not calls should block or not */
  int cgaClusterSize;          /*!< Cooperative group array cluster size */
  int minCTAs;                 /*!< Minimum number of cooperative thread arrays (blocks) */
  int maxCTAs;                 /*!< Maximum number of cooperative thread arrays (blocks) */
  const char *netName;         /*!< Force NCCL to use a specfic network */
  int splitShare;              /*!< Allow communicators to share resources */
  const char * graph;          /*!< Topo graph */
} UPTKncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in an error. */
#define UPTK_NCCL_CONFIG_INITIALIZER {                                        \
  sizeof(UPTKncclConfig_t),                             /* size */           \
  0xcafebeef,                                       /* magic */          \
  UPTKNCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */        \
  UPTK_NCCL_CONFIG_UNDEF_INT,                            /* blocking */       \
  UPTK_NCCL_CONFIG_UNDEF_INT,                            /* cgaClusterSize */ \
  UPTK_NCCL_CONFIG_UNDEF_INT,                            /* minCTAs */        \
  UPTK_NCCL_CONFIG_UNDEF_INT,                            /* maxCTAs */        \
  UPTK_NCCL_CONFIG_UNDEF_PTR,                            /* netName */        \
  UPTK_NCCL_CONFIG_UNDEF_INT,                            /* splitShare */     \
  UPTK_NCCL_CONFIG_UNDEF_PTR                             /* graph */          \
}
/*! @} */

/* This struct will be used by ncclGroupSimulateEnd() API to query information about simulation. */
typedef struct UPTKncclSimInfo_v22200 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    float estimatedTime;
} UPTKncclSimInfo_t;

/* NCCL_SIM_INFO_INITIALIZER must be assigned to initialize simInfo structure when it is created.
 * Not initialized simInfo will result in NCCL error. */
#define UPTK_NCCL_SIM_INFO_INITIALIZER {                                         \
  sizeof(UPTKncclSimInfo_t),                            /* size */              \
  0x74685283,                                       /* magic */             \
  UPTK_NCCL_VERSION(UPTK_NCCL_MAJOR, UPTK_NCCL_MINOR, UPTK_NCCL_PATCH), /* version */           \
  UPTK_NCCL_UNDEF_FLOAT                                  /* estimated time */    \
}

/*! @defgroup   rccl_api_version Version Information
    @details    API call that returns RCCL version
    @{ */

/*! @brief      Return the RCCL_VERSION_CODE of RCCL in the supplied integer.
    @details    This integer is coded with the MAJOR, MINOR and PATCH level of RCCL.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] version       Pointer to where version will be stored */

UPTKncclResult_t UPTKNCCLAPI UPTKncclGetVersion(int *version);

/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_communicator Communicator Initialization/Destruction
    @details    API calls that operate on communicators.
                Communicators objects are used to launch collective communication
                operations.  Unique ranks between 0 and N-1 must be assigned to
                each HIP device participating in the same Communicator.
                Using the same HIP device for multiple ranks of the same Communicator
                is not supported at this time.
    @{ */

/*! @brief      Generates an ID for UPTKncclCommInitRank.
    @details    Generates an ID to be used in UPTKncclCommInitRank.
                UPTKncclGetUniqueId should be called once by a single rank and the
                ID should be distributed to all ranks in the communicator before
                using it as a parameter for UPTKncclCommInitRank.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] uniqueId      Pointer to where uniqueId will be stored */
UPTKncclResult_t UPTKNCCLAPI UPTKncclGetUniqueId(UPTKncclUniqueId* uniqueId);

/*! @endcond */

/*! @brief      Creates a new communicator (multi thread/process version).
    @details    Rank must be between 0 and nranks-1 and unique within a communicator clique.
                Each rank is associated to a CUDA device, which has to be set before calling
                UPTKncclCommInitRank.  UPTKncclCommInitRank implicitly syncronizes with other ranks,
                so it must be called by different threads/processes or use UPTKncclGroupStart/UPTKncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] comm          Pointer to created communicator
    @param[in]  nranks        Total number of ranks participating in this communicator
    @param[in]  commId        UniqueId required for initialization
    @param[in]  rank          Current rank to create communicator for */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommInitRank(UPTKncclComm_t* comm, int nranks, UPTKncclUniqueId commId, int rank);

/*! @endcond */

/*! @brief      Creates a clique of communicators (single process version).
    @details    This is a convenience function to create a single-process communicator clique.
                Returns an array of ndev newly initialized communicators in comm.
                comm should be pre-allocated with size at least ndev*sizeof(UPTKncclComm_t).
                If devlist is NULL, the first ndev HIP devices are used.
                Order of devlist defines user-order of processors within the communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] comm          Pointer to array of created communicators
    @param[in]  ndev          Total number of ranks participating in this communicator
    @param[in]  devlist       Array of GPU device indices to create for */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommInitAll(UPTKncclComm_t* comm, int ndev, const int* devlist);

/*! @endcond */

/*! @brief      Frees local resources associated with communicator object.
    @details    Destroy all local resources associated with the passed in communicator object
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to destroy */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommDestroy(UPTKncclComm_t comm);

/*! @endcond */

/*! @brief      Abort any in-progress calls and destroy the communicator object.
    @details    Frees resources associated with communicator object and aborts any operations
                that might still be running on the device.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to abort and destroy */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommAbort(UPTKncclComm_t comm);

/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_errcheck Error Checking Calls
    @details    API calls that check for errors
    @{ */

/*! @brief      Returns a string for each result code.
    @details    Returns a human-readable string describing the given result code.
    @return     String containing description of result code.

    @param[in]  result        Result code to get description for */
const char* UPTKNCCLAPI UPTKncclGetErrorString(UPTKncclResult_t result);

/*! @endcond */

/*! @brief      Checks whether the comm has encountered any asynchronous errors
    @details    Query whether the provided communicator has encountered any asynchronous errors
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] asyncError    Pointer to where result code will be stored */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommGetAsyncError(UPTKncclComm_t comm, UPTKncclResult_t *asyncError);

/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_comminfo Communicator Information
    @details    API calls that query communicator information
    @{ */

/*! @brief      Gets the number of ranks in the communicator clique.
    @details    Returns the number of ranks in the communicator clique (as set during initialization)
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] count         Pointer to where number of ranks will be stored */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommCount(const UPTKncclComm_t comm, int* count);

/*~ @endcond */

/*! @brief      Get the ROCm device index associated with a communicator
    @details    Returns the ROCm device number associated with the provided communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] device        Pointer to where the associated ROCm device index will be stored */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommCuDevice(const UPTKncclComm_t comm, int* device);

/*! @endcond */

/*! @brief      Get the rank associated with a communicator
    @details    Returns the user-ordered "rank" associated with the provided communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] rank          Pointer to where the associated rank will be stored */
UPTKncclResult_t UPTKNCCLAPI UPTKncclCommUserRank(const UPTKncclComm_t comm, int* rank);

/*! @endcond */

/*! @defgroup   rccl_api_enumerations API Enumerations
    @details    Enumerations used by collective communication calls
    @{ */

/*! @brief      Dummy reduction enumeration
    @details    Dummy reduction enumeration used to determine value for ncclMaxRedOp */
typedef enum { UPTKncclNumOps_dummy = 5 } UPTKncclRedOp_dummy_t;

/*! @brief      Reduction operation selector
    @details    Enumeration used to specify the various reduction operations
                ncclNumOps is the number of built-in UPTKncclRedOp_t values and serves as
                the least possible value for dynamic UPTKncclRedOp_t values constructed by
                ncclRedOpCreate functions.

                ncclMaxRedOp is the largest valid value for UPTKncclRedOp_t and is defined
                to be the largest signed value (since compilers are permitted to use
                signed enums) that won't grow sizeof(UPTKncclRedOp_t) when compared to previous
                RCCL versions to maintain ABI compatibility. */
typedef enum { UPTKncclSum        = 0, /*!< Sum */
               UPTKncclProd       = 1, /*!< Product */
               UPTKncclMax        = 2, /*!< Max */
               UPTKncclMin        = 3, /*!< Min */
               UPTKncclAvg        = 4, /*!< Average */
               UPTKncclNumOps     = 5, /*!< Number of built-in reduction ops */
               UPTKncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(UPTKncclRedOp_dummy_t)) /*!< Largest value for UPTKncclRedOp_t */
             } UPTKncclRedOp_t;

/*! @brief      Data types
    @details    Enumeration of the various supported datatype */
typedef enum { UPTKncclInt8       = 0, UPTKncclChar       = 0,
               UPTKncclUint8      = 1,
               UPTKncclInt32      = 2, UPTKncclInt        = 2,
               UPTKncclUint32     = 3,
               UPTKncclInt64      = 4,
               UPTKncclUint64     = 5,
               UPTKncclFloat16    = 6, UPTKncclHalf       = 6,
               UPTKncclFloat32    = 7, UPTKncclFloat      = 7,
               UPTKncclFloat64    = 8, UPTKncclDouble     = 8,
               UPTKncclBfloat16   = 9,
#if defined(UPTKRCCL_FLOAT8)
               UPTKncclFp8E4M3    = 10,
               UPTKncclFp8E5M2    = 11,
               UPTKncclNumTypes   = 12 } UPTKncclDataType_t;
#else
               UPTKncclNumTypes   = 10 } UPTKncclDataType_t;
#endif
/*! @} */

/*! @defgroup   rccl_api_custom_redop Custom Reduction Operator
    @details    API calls relating to creation/destroying custom reduction operator
                that pre-multiplies local source arrays prior to reduction
    @{ */

/*! @brief      Location and dereferencing logic for scalar arguments.
    @details    Enumeration specifying memory location of the scalar argument.
                Based on where the value is stored, the argument will be dereferenced either
                while the collective is running (if in device memory), or before the ncclRedOpCreate()
                function returns (if in host memory). */
typedef enum {
  UPTKncclScalarDevice        = 0, /*!< Scalar is in device-visible memory */
  UPTKncclScalarHostImmediate = 1  /*!< Scalar is in host-visible memory */
} UPTKncclScalarResidence_t;

/*! @endcond */
/*! @} */

/*! @defgroup   rccl_collective_api Collective Communication Operations
    @details    Collective communication operations must be called separately for each
                communicator in a communicator clique.

                They return when operations have been enqueued on the HIP stream.
                Since they may perform inter-CPU synchronization, each call has to be done
                from a different thread or process, or need to use Group Semantics (see
                below).
    @{ */

/*! @brief      Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* into *recvbuff* using *op*
                operation.
                *recvbuff* may be NULL on all calls except for root device.
                *root* is the rank (not the HIP device) where data will reside after the
                 operation is complete.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Local device data buffer to be reduced
    @param[out] recvbuff      Data buffer where result is stored (only for *root* rank).  May be null for other ranks.
    @param[in]  count         Number of elements in every send buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator type
    @param[in]  root          Rank where result data array will be stored
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclReduce(const void* sendbuff, void* recvbuff, size_t count, UPTKncclDataType_t datatype,
    UPTKncclRedOp_t op, int root, UPTKncclComm_t comm, UPTKStream_t stream);

/*! @endcond */

/*! @brief      Broadcast
    @details    Copies *count* values from *sendbuff* on *root* to *recvbuff* on all devices.
                *root* is the rank (not the HIP device) where data resides before the operation is started.
                *sendbuff* may be NULL on ranks other than *root*.
                In-place operation will happen if *sendbuff* == *recvbuff*.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to copy (if *root*).  May be NULL for other ranks
    @param[in]  recvbuff      Data array to store received array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank of broadcast root
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, UPTKncclDataType_t datatype, int root,
    UPTKncclComm_t comm, UPTKStream_t stream);

/*! @endcond */

/*! @brief      All-Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* using *op* operation, and
                leaves identical copies of result on each *recvbuff*.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    UPTKncclDataType_t datatype, UPTKncclRedOp_t op, UPTKncclComm_t comm, UPTKStream_t stream);

/*! @endcond */

/*! @brief      Reduce-Scatter
    @details    Reduces data in *sendbuff* using *op* operation and leaves reduced result
                scattered over the devices so that *recvbuff* on rank i will contain the i-th
                block of the result.
                Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
                should have a size of at least nranks*recvcount elements.
                In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result subarray
    @param[in]  recvcount     Number of elements each rank receives
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, UPTKncclDataType_t datatype, UPTKncclRedOp_t op, UPTKncclComm_t comm,
    UPTKStream_t stream);

/*! @endcond */

/*! @brief      All-Gather
    @details    Each device gathers *sendcount* values from other GPUs into *recvbuff*,
                receiving data from rank i at offset i*sendcount.
                Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
                should have a size of at least nranks*sendcount elements.
                In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to send
    @param[out] recvbuff      Data array to store the gathered result
    @param[in]  sendcount     Number of elements each rank sends
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    UPTKncclDataType_t datatype, UPTKncclComm_t comm, UPTKStream_t stream);

/*! @endcond */

/*! @brief      Send
    @details    Send data from *sendbuff* to rank *peer*.
                Rank *peer* needs to call UPTKncclRecv with the same *datatype* and the same *count*
                as this rank.
                This operation is blocking for the GPU. If multiple UPTKncclSend and UPTKncclRecv operations
                need to progress concurrently to complete, they must be fused within a UPTKncclGroupStart /
                UPTKncclGroupEnd section.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send
    @param[in]  count         Number of elements to send
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclSend(const void* sendbuff, size_t count, UPTKncclDataType_t datatype, int peer,
    UPTKncclComm_t comm, UPTKStream_t stream);

/*! @endcond */

/*! @brief      Receive
    @details    Receive data from rank *peer* into *recvbuff*.
                Rank *peer* needs to call UPTKncclSend with the same datatype and the same count
                as this rank.
                This operation is blocking for the GPU. If multiple UPTKncclSend and UPTKncclRecv operations
                need to progress concurrently to complete, they must be fused within a UPTKncclGroupStart/
                UPTKncclGroupEnd section.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] recvbuff      Data array to receive
    @param[in]  count         Number of elements to receive
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
UPTKncclResult_t UPTKNCCLAPI UPTKncclRecv(void* recvbuff, size_t count, UPTKncclDataType_t datatype, int peer,
    UPTKncclComm_t comm, UPTKStream_t stream);


/*! @endcond */

/*! @defgroup   rccl_group_api Group semantics
    @details    When managing multiple GPUs from a single thread, and since RCCL collective
                calls may perform inter-CPU synchronization, we need to "group" calls for
                different ranks/devices into a single call.

                Grouping RCCL calls as being part of the same collective operation is done
                using UPTKncclGroupStart and UPTKncclGroupEnd. UPTKncclGroupStart will enqueue all
                collective calls until the UPTKncclGroupEnd call, which will wait for all calls
                to be complete. Note that for collective communication, UPTKncclGroupEnd only
                guarantees that the operations are enqueued on the streams, not that
                the operation is effectively done.

                Both collective communication and UPTKncclCommInitRank can be used in conjunction
                of UPTKncclGroupStart/UPTKncclGroupEnd, but not together.

                Group semantics also allow to fuse multiple operations on the same device
                to improve performance (for aggregated collective calls), or to permit
                concurrent progress of multiple send/receive operations.
    @{ */

/*! @brief      Group Start
    @details    Start a group call. All calls to RCCL until UPTKncclGroupEnd will be fused into
                a single RCCL operation. Nothing will be started on the HIP stream until
                UPTKncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details. */
UPTKncclResult_t UPTKNCCLAPI UPTKncclGroupStart();

/*! @endcond */

/*! @brief      Group End
    @details    End a group call. Start a fused RCCL operation consisting of all calls since
                UPTKncclGroupStart. Operations on the HIP stream depending on the RCCL operations
                need to be called after UPTKncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details. */
UPTKncclResult_t UPTKNCCLAPI UPTKncclGroupEnd();



#ifdef __cplusplus
} // end extern "C"
#endif

#endif // UPTK_NCCL_H_ end include guard
