#ifndef __RTC_HPP__
#define __RTC_HPP__
#include <hip/hiprtc.h>
#include <nvrtc.h>
#include <UPTK_rtc.h>

#define RTC_INVALID_ENUM "RTC_INVALID_ENUM"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_BOLD "\x1b[1m"
#define ANSI_COLOR_HIGH_SATURATION "\x1b[91m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define Debug() do { \
    printf(ANSI_COLOR_BOLD ANSI_COLOR_HIGH_SATURATION ANSI_COLOR_MAGENTA "Warning: The %s is currently not supported.The %s is iUPTKalid.\n" ANSI_COLOR_RESET, __FUNCTION__, __FUNCTION__); \
} while (0)

nvrtcResult UPTKrtcResultTonvrtcResult(UPTKrtcResult para);
UPTKrtcResult nvrtcResultToUPTKrtcResult(nvrtcResult para);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif //  __RTC_HPP__
