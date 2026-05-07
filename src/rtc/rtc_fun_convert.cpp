#include "rtc.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

UPTKrtcResult UPTKrtcAddNameExpression(UPTKrtcProgram prog,const char * const name_expression)
{
    nvrtcResult nv_res;
    nv_res = nvrtcAddNameExpression((nvrtcProgram) prog, name_expression);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcCompileProgram(UPTKrtcProgram prog,int numOptions,const char * const * options)
{
    nvrtcResult nv_res;
    nv_res = nvrtcCompileProgram((nvrtcProgram) prog, numOptions, (const char **)options);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcCreateProgram(UPTKrtcProgram * prog,const char * src,const char * name,int numHeaders,const char * const * headers,const char * const * includeNames)
{
    nvrtcResult nv_res;
    nv_res = nvrtcCreateProgram((nvrtcProgram *) prog, src, name, numHeaders,(const char **) headers, (const char **) includeNames);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcDestroyProgram(UPTKrtcProgram * prog)
{
    nvrtcResult nv_res;
    nv_res = nvrtcDestroyProgram((nvrtcProgram *)prog);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

const char * UPTKrtcGetErrorString(UPTKrtcResult result)
{
    switch (result)
    {
    case UPTKRTC_ERROR_BUILTIN_OPERATION_FAILURE:
        return "UPTKrtc_ERROR_BUILTIN_OPERATION_FAILURE";
    case UPTKRTC_ERROR_COMPILATION:
        return "UPTKrtc_ERROR_COMPILATION";
    case UPTKRTC_ERROR_INTERNAL_ERROR:
        return "UPTKrtc_ERROR_INTERNAL_ERROR";
    case UPTKRTC_ERROR_INVALID_INPUT:
        return "UPTKrtc_ERROR_INVALID_INPUT";
    case UPTKRTC_ERROR_INVALID_OPTION:
        return "UPTKrtc_ERROR_INVALID_OPTION";
    case UPTKRTC_ERROR_INVALID_PROGRAM:
        return "UPTKrtc_ERROR_INVALID_PROGRAM";
    case UPTKRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return "UPTKrtc_ERROR_NAME_EXPRESSION_NOT_VALID";
    case UPTKRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
        return "UPTKrtc_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case UPTKRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
        return "UPTKrtc_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case UPTKRTC_ERROR_OUT_OF_MEMORY:
        return "UPTKrtc_ERROR_OUT_OF_MEMORY";
    case UPTKRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return "UPTKrtc_ERROR_PROGRAM_CREATION_FAILURE";
    case UPTKRTC_SUCCESS:
        return "UPTKrtc_SUCCESS";
    default:
        return "UPTKrtc_ERROR unknown";
    }
}

UPTKrtcResult UPTKrtcGetLoweredName(UPTKrtcProgram prog,const char * const name_expression,const char ** lowered_name)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetLoweredName((nvrtcProgram ) prog, name_expression, lowered_name);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetPTX(UPTKrtcProgram prog,char * ptx)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetPTX((nvrtcProgram ) prog, ptx);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetPTXSize(UPTKrtcProgram prog,size_t * ptxSizeRet)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetPTXSize((nvrtcProgram ) prog, ptxSizeRet);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetProgramLog(UPTKrtcProgram prog,char * log)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetProgramLog((nvrtcProgram ) prog, log);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetProgramLogSize(UPTKrtcProgram prog,size_t * logSizeRet)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetProgramLogSize((nvrtcProgram ) prog, logSizeRet);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcVersion(int * major,int * minor)
{
    nvrtcResult nv_res;
    nv_res = nvrtcVersion(major, minor);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetCUBINSize(UPTKrtcProgram prog, size_t *cubinSizeRet)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetCUBINSize((nvrtcProgram ) prog, cubinSizeRet);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetCUBIN(UPTKrtcProgram prog, char *cubin)
{
    nvrtcResult nv_res;
    nv_res = nvrtcGetCUBIN((nvrtcProgram ) prog, cubin);
    return nvrtcResultToUPTKrtcResult(nv_res);
}

UPTKrtcResult UPTKrtcGetNVVMSize(UPTKrtcProgram prog, size_t *nvvmSizeRet)
{
    Debug();
    return UPTKRTC_ERROR_INVALID_PROGRAM;
}

UPTKrtcResult UPTKrtcGetNVVM(UPTKrtcProgram prog, char *nvvm)
{
    Debug();
    return UPTKRTC_ERROR_INVALID_PROGRAM;
}

UPTKrtcResult UPTKrtcGetNumSupportedArchs(int* numArchs)
{
    Debug();
    return UPTKRTC_ERROR_INVALID_PROGRAM;
}

UPTKrtcResult UPTKrtcGetSupportedArchs(int* supportedArchs)
{
    Debug();
    return UPTKRTC_ERROR_INVALID_PROGRAM;
}


#if defined(__cplusplus)
}
#endif /* __cplusplus */
