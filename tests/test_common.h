#ifndef __TEST_COMMON_H__
#define __TEST_COMMON_H__

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RED    "\x1b[31m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_CYAN   "\x1b[36m"
#define ANSI_BOLD   "\x1b[1m"
#define ANSI_RESET  "\x1b[0m"

static int g_total   = 0;
static int g_passed  = 0;
static int g_failed  = 0;
static int g_skipped = 0;

#define TEST_SECTION(section_name)                                          \
    printf("\n" ANSI_BOLD ANSI_CYAN "===== %s =====" ANSI_RESET "\n\n",     \
           section_name)

#define TEST_CHECK(test_name, input_str, expected_str, actual_str, cond)    \
    do {                                                                    \
        g_total++;                                                          \
        printf("  [TEST] %-50s", test_name);                                \
        if (cond) {                                                         \
            g_passed++;                                                     \
            printf(ANSI_BOLD ANSI_GREEN "[PASS]" ANSI_RESET "\n");          \
        } else {                                                            \
            g_failed++;                                                     \
            printf(ANSI_BOLD ANSI_RED   "[FAIL]" ANSI_RESET "\n");          \
        }                                                                   \
        printf("         Input:    %s\n", input_str);                       \
        printf("         Expected: %s\n", expected_str);                    \
        printf("         Actual:   %s\n\n", actual_str);                    \
    } while (0)

#define TEST_SKIP(test_name, reason)                                        \
    do {                                                                    \
        g_total++;                                                          \
        g_skipped++;                                                        \
        printf("  [TEST] %-50s" ANSI_BOLD ANSI_YELLOW "[SKIP]" ANSI_RESET   \
               "\n         Reason: %s\n\n", test_name, reason);             \
    } while (0)

#define TEST_ENUM_CONVERT(test_name, func, input_val, expected_val)         \
    do {                                                                    \
        auto _actual_ = func(input_val);                                    \
        char _exp_[64], _act_[64];                                          \
        snprintf(_exp_, sizeof(_exp_), "%d", (int)(expected_val));          \
        snprintf(_act_, sizeof(_act_), "%d", (int)(_actual_));              \
        TEST_CHECK(test_name, #input_val, _exp_, _act_,                     \
                   (int)(_actual_) == (int)(expected_val));                  \
    } while (0)

#define TEST_ENUM_ROUNDTRIP(test_name, toB, toA, val_a, val_b)             \
    do {                                                                    \
        {                                                                   \
            auto _r_ = toB(val_a);                                          \
            char _e_[64], _a_[64];                                          \
            snprintf(_e_, sizeof(_e_), "%d", (int)(val_b));                 \
            snprintf(_a_, sizeof(_a_), "%d", (int)(_r_));                   \
            TEST_CHECK(test_name " (fwd)", #val_a, _e_, _a_,               \
                       (int)(_r_) == (int)(val_b));                         \
        }                                                                   \
        {                                                                   \
            auto _r2_ = toA(val_b);                                         \
            char _e2_[64], _a2_[64];                                        \
            snprintf(_e2_, sizeof(_e2_), "%d", (int)(val_a));               \
            snprintf(_a2_, sizeof(_a2_), "%d", (int)(_r2_));                \
            TEST_CHECK(test_name " (rev)", #val_b, _e2_, _a2_,             \
                       (int)(_r2_) == (int)(val_a));                        \
        }                                                                   \
    } while (0)

#define TEST_API_STATUS(test_name, input_str, actual, expected)             \
    do {                                                                    \
        char _e_[64], _a_[64];                                              \
        snprintf(_e_, sizeof(_e_), "%d", (int)(expected));                  \
        snprintf(_a_, sizeof(_a_), "%d", (int)(actual));                    \
        TEST_CHECK(test_name, input_str, _e_, _a_,                         \
                   (int)(actual) == (int)(expected));                        \
    } while (0)

#define TEST_SUMMARY(suite_name)                                            \
    do {                                                                    \
        printf("\n");                                                       \
        printf("=============================================\n");          \
        printf("  Test Suite : %s\n", suite_name);                          \
        printf("  Total      : %d\n", g_total);                             \
        printf("  Passed     : %d\n", g_passed);                            \
        printf("  Failed     : %d\n", g_failed);                            \
        printf("  Skipped    : %d\n", g_skipped);                           \
        if (g_failed == 0) {                                                \
            printf("  Result     : " ANSI_BOLD ANSI_GREEN                   \
                   "ALL PASSED" ANSI_RESET "\n");                           \
        } else {                                                            \
            printf("  Result     : " ANSI_BOLD ANSI_RED                     \
                   "%d FAILED" ANSI_RESET "\n", g_failed);                  \
        }                                                                   \
        printf("=============================================\n");          \
        return g_failed > 0 ? 1 : 0;                                       \
    } while (0)

#endif /* __TEST_COMMON_H__ */
