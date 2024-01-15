
#pragma once

#include <cstdio>
#include <cstdarg>
#include <mutex>

static std::mutex mtx_;


// 定义ANSI转义序列控制颜色的宏
#define RED_TEXT "\033[1;31m"
#define GREEN_TEXT "\033[1;32m"
#define RESET_TEXT "\033[0m"

/// @brief 输出日志到标准输出
/// @param file 打印日志文件名
/// @param line 打印日志的行数
/// @param fmt 格式化format
/// @param  fmt的参数...
static void log_(const char* file, int line, const char* fmt, ...)
{
    std::unique_lock<std::mutex> lock(mtx_);
    // 小心溢出
    char buffer[256] = { 0 };

    int cnt = std::sprintf(buffer, "%s[+][Log %s:%d] %s", RESET_TEXT, file, line, GREEN_TEXT);
    if (cnt > 0)
        std::fprintf(stdout, "%s", buffer);


    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(stdout, fmt, args);
    va_end(args);

    std::fprintf(stdout, "%s", RESET_TEXT);
}

/// @brief 输出日志到标准错误
/// @param file 打印日志文件名
/// @param line 打印日志的行数
/// @param fmt 格式化format
/// @param  fmt的参数...
static void error_(const char* file, int line, const char* fmt, ...)
{
    std::unique_lock<std::mutex> lock(mtx_);
    char buffer[256] = { 0 };

    int cnt = std::sprintf(buffer, "%s[+][Error %s:%d] %s", RESET_TEXT, file, line, RED_TEXT);
    if (cnt > 0)
        std::fprintf(stderr, "%s", buffer);

    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "%s", RESET_TEXT);
}


/*
    用于打印日志，并且可以输出打印的文件名称、行数
*/


#ifdef NO_LOGGER // 控制是否打印输出
#define log(fmt, ...) 
#define err(fmt, ...)
#else
#define log(fmt, ...) log_(__FILE__, __LINE__, fmt, ##__VA_ARGS__)      // log实际调用处
#define err(fmt, ...) error_(__FILE__, __LINE__, fmt, ##__VA_ARGS__)    // err实际调用处
#endif