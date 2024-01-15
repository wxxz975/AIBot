
#pragma once

#include <cstdio>
#include <cstdarg>
#include <mutex>

static std::mutex mtx_;


// ����ANSIת�����п�����ɫ�ĺ�
#define RED_TEXT "\033[1;31m"
#define GREEN_TEXT "\033[1;32m"
#define RESET_TEXT "\033[0m"

/// @brief �����־����׼���
/// @param file ��ӡ��־�ļ���
/// @param line ��ӡ��־������
/// @param fmt ��ʽ��format
/// @param  fmt�Ĳ���...
static void log_(const char* file, int line, const char* fmt, ...)
{
    std::unique_lock<std::mutex> lock(mtx_);
    // С�����
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

/// @brief �����־����׼����
/// @param file ��ӡ��־�ļ���
/// @param line ��ӡ��־������
/// @param fmt ��ʽ��format
/// @param  fmt�Ĳ���...
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
    ���ڴ�ӡ��־�����ҿ��������ӡ���ļ����ơ�����
*/


#ifdef NO_LOGGER // �����Ƿ��ӡ���
#define log(fmt, ...) 
#define err(fmt, ...)
#else
#define log(fmt, ...) log_(__FILE__, __LINE__, fmt, ##__VA_ARGS__)      // logʵ�ʵ��ô�
#define err(fmt, ...) error_(__FILE__, __LINE__, fmt, ##__VA_ARGS__)    // errʵ�ʵ��ô�
#endif