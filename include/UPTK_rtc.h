#ifndef __UPTKRTC_H__
#define __UPTKRTC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
typedef enum {
  UPTKRTC_SUCCESS = 0,
  UPTKRTC_ERROR_OUT_OF_MEMORY = 1,
  UPTKRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  UPTKRTC_ERROR_INVALID_INPUT = 3,
  UPTKRTC_ERROR_INVALID_PROGRAM = 4,
  UPTKRTC_ERROR_INVALID_OPTION = 5,
  UPTKRTC_ERROR_COMPILATION = 6,
  UPTKRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  UPTKRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  UPTKRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  UPTKRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  UPTKRTC_ERROR_INTERNAL_ERROR = 11
} UPTKrtcResult;
const char *UPTKrtcGetErrorString(UPTKrtcResult result);
UPTKrtcResult UPTKrtcVersion(int *major, int *minor);
UPTKrtcResult UPTKrtcGetNumSupportedArchs(int* numArchs);
UPTKrtcResult UPTKrtcGetSupportedArchs(int* supportedArchs);
typedef struct _UPTKrtcProgram *UPTKrtcProgram;
UPTKrtcResult UPTKrtcCreateProgram(UPTKrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames);
UPTKrtcResult UPTKrtcDestroyProgram(UPTKrtcProgram *prog);
UPTKrtcResult UPTKrtcCompileProgram(UPTKrtcProgram prog,
                                int numOptions, const char * const *options);
UPTKrtcResult UPTKrtcGetPTXSize(UPTKrtcProgram prog, size_t *ptxSizeRet);
UPTKrtcResult UPTKrtcGetPTX(UPTKrtcProgram prog, char *ptx);
UPTKrtcResult UPTKrtcGetCUBINSize(UPTKrtcProgram prog, size_t *cubinSizeRet);
UPTKrtcResult UPTKrtcGetCUBIN(UPTKrtcProgram prog, char *cubin);
UPTKrtcResult UPTKrtcGetNVVMSize(UPTKrtcProgram prog, size_t *UPTKvmSizeRet);
UPTKrtcResult UPTKrtcGetNVVM(UPTKrtcProgram prog, char *UPTKvm);
UPTKrtcResult UPTKrtcGetProgramLogSize(UPTKrtcProgram prog, size_t *logSizeRet);
UPTKrtcResult UPTKrtcGetProgramLog(UPTKrtcProgram prog, char *log);
UPTKrtcResult UPTKrtcAddNameExpression(UPTKrtcProgram prog,
                                const char *const name_expression);
UPTKrtcResult UPTKrtcGetLoweredName(UPTKrtcProgram prog,
                                const char *const name_expression,
                                const char** lowered_name);
#ifdef __cplusplus
}
#endif
#endif
