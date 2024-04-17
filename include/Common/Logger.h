#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <cstdarg>

#include "Singleton.h"

enum LogLevel {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERROR
};

class Logger: public Singleton<Logger> {
public:
    Logger(const std::string& filename = "log.txt") : m_output(filename, std::ios::app) {}

    void log(LogLevel level, const std::string& message, const char* file, int line) {
        std::string level_str;
        const char* color;
        switch (level) {
            case LOG_DEBUG:
                level_str = "DEBUG";
                color = "\033[0m";  // 默认颜色
                break;
            case LOG_INFO:
                level_str = "INFO";
                color = "\033[32m"; // 绿色
                break;
            case LOG_WARNING:
                level_str = "WARNING";
                color = "\033[33m"; // 黄色
                break;
            case LOG_ERROR:
                level_str = "ERROR";
                color = "\033[31m"; // 红色
                break;
        }

        std::time_t now = std::time(nullptr);
        char timestamp[20];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        std::cout << color << "[" << timestamp << "] [" << level_str << "] [" << file << ":" << line << "] " << message << "\033[0m" << std::endl;
        m_output << "[" << timestamp << "] [" << level_str << "] [" << file << ":" << line << "] " << message << std::endl;
    }

    // void logf(LogLevel level, const char* file, int line, const char* format, ...) {
    //     va_list args;
    //     va_start(args, format);

    //     char buffer[256];
    //     std::vsnprintf(buffer, sizeof(buffer), format, args);

    //     va_end(args);

    //     log(level, buffer, file, line);
    // }

private:
    std::ofstream m_output;
};

#define log(msg) Logger::Instance()->log(LOG_INFO, msg, __FILE__, __LINE__)
#define err(msg) Logger::Instance()->log(LOG_ERROR, msg, __FILE__, __LINE__)

