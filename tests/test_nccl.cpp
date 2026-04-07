#include "test_common.h"
#include "nccl/nccl.hpp"

int main()
{
    printf("=============================================\n");
    printf("  UPTK NCCL Test Suite\n");
    printf("=============================================\n");

    /* ============================================================
     *  Section 1: Type Converter Tests (no GPU required)
     * ============================================================ */
    TEST_SECTION("NCCL Type Converters");

    /* --- UPTKncclResult_t <-> ncclResult_t --- */
    TEST_ENUM_ROUNDTRIP("Result Success",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclSuccess, ncclSuccess);
    TEST_ENUM_ROUNDTRIP("Result UnhandledCudaError",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclUnhandledCudaError, ncclUnhandledCudaError);
    TEST_ENUM_ROUNDTRIP("Result SystemError",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclSystemError, ncclSystemError);
    TEST_ENUM_ROUNDTRIP("Result InternalError",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclInternalError, ncclInternalError);
    TEST_ENUM_ROUNDTRIP("Result InvalidArgument",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclInvalidArgument, ncclInvalidArgument);
    TEST_ENUM_ROUNDTRIP("Result InvalidUsage",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclInvalidUsage, ncclInvalidUsage);
    TEST_ENUM_ROUNDTRIP("Result RemoteError",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclRemoteError, ncclRemoteError);
    TEST_ENUM_ROUNDTRIP("Result InProgress",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclInProgress, ncclInProgress);
    TEST_ENUM_ROUNDTRIP("Result NumResults",
        UPTKncclResultToncclResult, ncclResultToUPTKncclResult,
        UPTKncclNumResults, ncclNumResults);

    /* --- UPTKncclRedOp_t -> ncclRedOp_t --- */
    TEST_ENUM_CONVERT("RedOp Sum",
        UPTKncclRedOpToncclRedOp, UPTKncclSum, ncclSum);
    TEST_ENUM_CONVERT("RedOp Prod",
        UPTKncclRedOpToncclRedOp, UPTKncclProd, ncclProd);
    TEST_ENUM_CONVERT("RedOp Max",
        UPTKncclRedOpToncclRedOp, UPTKncclMax, ncclMax);
    TEST_ENUM_CONVERT("RedOp Min",
        UPTKncclRedOpToncclRedOp, UPTKncclMin, ncclMin);
    TEST_ENUM_CONVERT("RedOp Avg",
        UPTKncclRedOpToncclRedOp, UPTKncclAvg, ncclAvg);
    TEST_ENUM_CONVERT("RedOp NumOps",
        UPTKncclRedOpToncclRedOp, UPTKncclNumOps, ncclNumOps);
    TEST_ENUM_CONVERT("RedOp MaxRedOp",
        UPTKncclRedOpToncclRedOp, UPTKncclMaxRedOp, ncclMaxRedOp);

    /* --- UPTKncclRedOp_dummy_t -> ncclRedOp_dummy_t --- */
    TEST_ENUM_CONVERT("RedOp_dummy NumOps_dummy",
        UPTKncclRedOp_dummyToncclRedOp_dummy,
        UPTKncclNumOps_dummy, ncclNumOps_dummy);

    /* --- UPTKncclDataType_t -> ncclDataType_t --- */
    TEST_ENUM_CONVERT("DataType Int8",
        UPTKncclDataTypeToncclDataType, UPTKncclInt8, ncclInt8);
    TEST_ENUM_CONVERT("DataType Uint8",
        UPTKncclDataTypeToncclDataType, UPTKncclUint8, ncclUint8);
    TEST_ENUM_CONVERT("DataType Int32",
        UPTKncclDataTypeToncclDataType, UPTKncclInt32, ncclInt32);
    TEST_ENUM_CONVERT("DataType Uint32",
        UPTKncclDataTypeToncclDataType, UPTKncclUint32, ncclUint32);
    TEST_ENUM_CONVERT("DataType Int64",
        UPTKncclDataTypeToncclDataType, UPTKncclInt64, ncclInt64);
    TEST_ENUM_CONVERT("DataType Uint64",
        UPTKncclDataTypeToncclDataType, UPTKncclUint64, ncclUint64);
    TEST_ENUM_CONVERT("DataType Float16",
        UPTKncclDataTypeToncclDataType, UPTKncclFloat16, ncclFloat16);
    TEST_ENUM_CONVERT("DataType Float32",
        UPTKncclDataTypeToncclDataType, UPTKncclFloat32, ncclFloat32);
    TEST_ENUM_CONVERT("DataType Float64",
        UPTKncclDataTypeToncclDataType, UPTKncclFloat64, ncclFloat64);
    TEST_ENUM_CONVERT("DataType Bfloat16",
        UPTKncclDataTypeToncclDataType, UPTKncclBfloat16, ncclBfloat16);

    /* --- UPTKncclScalarResidence_t -> ncclScalarResidence_t --- */
    TEST_ENUM_CONVERT("ScalarRes Device",
        UPTKncclScalarResidenceToncclScalarResidence,
        UPTKncclScalarDevice, ncclScalarDevice);
    TEST_ENUM_CONVERT("ScalarRes HostImmediate",
        UPTKncclScalarResidenceToncclScalarResidence,
        UPTKncclScalarHostImmediate, ncclScalarHostImmediate);

    /* --- UPTKncclUniqueId roundtrip --- */
    {
        UPTKncclUniqueId uptkId;
        memset(&uptkId, 0xAB, sizeof(uptkId));

        ncclUniqueId cudaId;
        UPTKncclUniqueIdToncclUniqueId(&uptkId, &cudaId);

        UPTKncclUniqueId roundtrip;
        ncclUniqueIdToUPTKncclUniqueId(&cudaId, &roundtrip);

        int len = UPTK_NCCL_UNIQUE_ID_BYTES < NCCL_UNIQUE_ID_BYTES
                      ? UPTK_NCCL_UNIQUE_ID_BYTES
                      : NCCL_UNIQUE_ID_BYTES;
        bool match = (memcmp(&uptkId, &roundtrip, len) == 0);

        TEST_CHECK("UniqueId roundtrip",
            "fill 0xAB -> to nccl -> back",
            "match original", match ? "match" : "MISMATCH", match);
    }

    /* ============================================================
     *  Section 2: API Function Tests
     * ============================================================ */
    TEST_SECTION("NCCL API Functions");

    /* --- UPTKncclGetVersion --- */
    {
        int version = 0;
        UPTKncclResult_t res = UPTKncclGetVersion(&version);
        TEST_API_STATUS("UPTKncclGetVersion",
            "UPTKncclGetVersion(&version)", res, UPTKncclSuccess);

        char exp[64], act[64];
        int expected = UPTK_NCCL_VERSION(UPTK_NCCL_MAJOR, UPTK_NCCL_MINOR,
                                          UPTK_NCCL_PATCH);
        snprintf(exp, sizeof(exp), "%d", expected);
        snprintf(act, sizeof(act), "%d", version);
        TEST_CHECK("UPTKncclGetVersion value",
            "version", exp, act, version == expected);
    }

    /* --- UPTKncclGetVersion NULL check --- */
    {
        UPTKncclResult_t res = UPTKncclGetVersion(nullptr);
        TEST_API_STATUS("UPTKncclGetVersion(NULL)",
            "version=NULL", res, UPTKncclInvalidArgument);
    }

    /* --- UPTKncclGetErrorString --- */
    {
        const char *str = UPTKncclGetErrorString(UPTKncclSuccess);
        bool valid = (str != nullptr && strlen(str) > 0);
        TEST_CHECK("UPTKncclGetErrorString(Success)",
            "UPTKncclSuccess",
            "non-empty string",
            str ? str : "NULL",
            valid);
    }
    {
        const char *str = UPTKncclGetErrorString(UPTKncclInvalidArgument);
        bool valid = (str != nullptr && strlen(str) > 0);
        TEST_CHECK("UPTKncclGetErrorString(InvalidArg)",
            "UPTKncclInvalidArgument",
            "non-empty string",
            str ? str : "NULL",
            valid);
    }
    {
        const char *str = UPTKncclGetErrorString(UPTKncclSystemError);
        bool valid = (str != nullptr && strlen(str) > 0);
        TEST_CHECK("UPTKncclGetErrorString(SystemErr)",
            "UPTKncclSystemError",
            "non-empty string",
            str ? str : "NULL",
            valid);
    }

    /* --- UPTKncclGroupStart / UPTKncclGroupEnd --- */
    {
        UPTKncclResult_t res = UPTKncclGroupStart();
        TEST_API_STATUS("UPTKncclGroupStart",
            "UPTKncclGroupStart()", res, UPTKncclSuccess);

        res = UPTKncclGroupEnd();
        TEST_API_STATUS("UPTKncclGroupEnd",
            "UPTKncclGroupEnd()", res, UPTKncclSuccess);
    }

    TEST_SUMMARY("NCCL");
}
