#ifndef _LOG_H_
#define _LOG_H_

#include <ctime>
#include <iostream>
#include <cstring>
#ifdef ANDROID
#include <android/log.h>
#endif

#ifndef log_debug
#ifdef ANDROID
#define log_debug(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "", __VA_ARGS__))
#else
#define log_debug(format, ...) \
    printf("\033[0;35m[DEBUG]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif
#endif

#ifndef log_error
#ifdef ANDROID
#define log_error(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "", __VA_ARGS__))
#else
#define log_error(format, ...) \
    printf("\033[0;31m[ERROR]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif
#endif

#ifndef log_info
#ifdef ANDROID
#define log_info(...) ((void)__android_log_print(ANDROID_LOG_INFO, "", __VA_ARGS__))
#else
#define log_info(format, ...) \
    printf("\033[0;32m[INFO]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif
#endif

#ifndef log_warn
#ifdef ANDROID
#define log_warn(...) ((void)__android_log_print(ANDROID_LOG_WARN, "", __VA_ARGS__))
#else
#define log_warn(format, ...) \
    printf("\033[0;33m[WARN]\033[0;0m " format "\n", ##__VA_ARGS__)
#endif
#endif

#ifndef PRINT_VAR
#ifndef NDEBUG
#define PRINT_VAR(var) std::cout << #var << " = " << var << std::endl
#else
#define PRINT_VAR(var)
#endif
#endif

#define runtime_assert(condition, message)                                                                                         \
    do {                                                                                                                           \
        if (!(condition)) {                                                                                                        \
            log_error("Assertion failed at " __FILE__ ":%d : %s\nWhen testing condition:\n    %s", __LINE__, message, #condition); \
            abort();                                                                                                               \
        }                                                                                                                          \
    } while (0)

#endif //
