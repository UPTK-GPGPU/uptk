#if !defined(__UPTK_LIBRARY_TYPES_H__)
#define __UPTK_LIBRARY_TYPES_H__

typedef enum UPTKDataType_t
{
    UPTK_R_16F  =  2, /* real as a half */
    UPTK_C_16F  =  6, /* complex as a pair of half numbers */
    UPTK_R_16BF = 14, /* real as a UPTK_bfloat16 */
    UPTK_C_16BF = 15, /* complex as a pair of UPTK_bfloat16 numbers */
    UPTK_R_32F  =  0, /* real as a float */
    UPTK_C_32F  =  4, /* complex as a pair of float numbers */
    UPTK_R_64F  =  1, /* real as a double */
    UPTK_C_64F  =  5, /* complex as a pair of double numbers */
    UPTK_R_4I   = 16, /* real as a signed 4-bit int */
    UPTK_C_4I   = 17, /* complex as a pair of signed 4-bit int numbers */
    UPTK_R_4U   = 18, /* real as a unsigned 4-bit int */
    UPTK_C_4U   = 19, /* complex as a pair of unsigned 4-bit int numbers */
    UPTK_R_8I   =  3, /* real as a signed 8-bit int */
    UPTK_C_8I   =  7, /* complex as a pair of signed 8-bit int numbers */
    UPTK_R_8U   =  8, /* real as a unsigned 8-bit int */
    UPTK_C_8U   =  9, /* complex as a pair of unsigned 8-bit int numbers */
    UPTK_R_16I  = 20, /* real as a signed 16-bit int */
    UPTK_C_16I  = 21, /* complex as a pair of signed 16-bit int numbers */
    UPTK_R_16U  = 22, /* real as a unsigned 16-bit int */
    UPTK_C_16U  = 23, /* complex as a pair of unsigned 16-bit int numbers */
    UPTK_R_32I  = 10, /* real as a signed 32-bit int */
    UPTK_C_32I  = 11, /* complex as a pair of signed 32-bit int numbers */
    UPTK_R_32U  = 12, /* real as a unsigned 32-bit int */
    UPTK_C_32U  = 13, /* complex as a pair of unsigned 32-bit int numbers */
    UPTK_R_64I  = 24, /* real as a signed 64-bit int */
    UPTK_C_64I  = 25, /* complex as a pair of signed 64-bit int numbers */
    UPTK_R_64U  = 26, /* real as a unsigned 64-bit int */
    UPTK_C_64U  = 27, /* complex as a pair of unsigned 64-bit int numbers */
    UPTK_R_8F_E4M3 = 28, /* real as a UPTK_fp8_e4m3 */
    UPTK_R_8F_E5M2 = 29, /* real as a UPTK_fp8_e5m2 */
} UPTKDataType;


typedef enum UPTKlibraryPropertyType_t
{
    UPTK_MAJOR_VERSION,
    UPTK_MINOR_VERSION,
    UPTK_PATCH_LEVEL
} UPTKlibraryPropertyType;


#ifndef __cplusplus
typedef enum UPTKDataType_t UPTKDataType_t;
typedef enum UPTKlibraryPropertyType_t UPTKlibraryPropertyType_t;
#endif

#endif /* !__UPTK_LIBRARY_TYPES_H__ */
