/**
 * @file logger.hh
 * @brief Project logger facade.
 *
 * This header provides:
 *  - A Needletail-style logger facade.
 *  - Compile-time switches to enable/disable logging.
 *  - Optional file logging to "aker_log_<timestamp>.log" in the current working directory.
 */

#pragma once

/**
 * @brief Compile-time logging switch.
 *
 * Set to 0 to compile out all logging statements.
 */
#ifndef AKER_ENABLE_LOGGING
#define AKER_ENABLE_LOGGING 1
#endif

/**
 * @brief Compile-time file logging switch.
 *
 * Set to 1 to enable a default file sink.
 */
#ifndef AKER_LOG_TO_FILE
#define AKER_LOG_TO_FILE 0
#endif

/**
 * @brief Compile-time console logging switch.
 */
#ifndef AKER_LOG_TO_CONSOLE
#define AKER_LOG_TO_CONSOLE 1
#endif

/**
 * @brief Compile-time auto flush switch for file sink.
 */
#ifndef AKER_LOG_AUTO_FLUSH
#define AKER_LOG_AUTO_FLUSH 0
#endif

#include <atomic>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if AKER_ENABLE_LOGGING

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <iostream>

namespace aker
{

/**
 * @brief Project-wide logger facade.
 *
 * This logger follows the Needletail structure:
 *  - One-time initialization via std::call_once
 *  - Boost.Log sinks (console/file)
 *  - Convenience macros and streaming support
 *
 * It also provides a small adapter returned by getLogger() so that
 * call sites that previously used a spdlog-like "logger->info(...)" pattern
 * can keep compiling with minimal changes.
 */
class Logger final
{
public:
    /**
     * @brief Logging level.
     */
    enum class Level
    {
        k_trace,
        k_debug,
        k_info,
        k_warning,
        k_error,
        k_fatal,
    };

    /**
     * @brief Default rotation size for file sink.
     */
    static constexpr std::size_t k_default_rotation_size_bytes = 64ull * 1024ull * 1024ull;

    /**
     * @brief Logger initialization options.
     */
    struct Options
    {
        Level min_level = Level::k_info;

        bool console = true;
        std::ostream* console_stream = &std::clog;

        std::optional<std::string> file_path;
        std::size_t rotation_size_bytes = k_default_rotation_size_bytes;
        bool auto_flush = false;
    };

    /**
     * @brief A minimal spdlog-like adapter.
     */
    class Adapter final
    {
    public:
        /**
         * @brief Construct adapter with a static logger name.
         */
        explicit Adapter(std::string logger_name);

        /**
         * @brief Log a trace-level message.
         */
        template <typename... Args>
        void trace(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::trace, fmt, std::forward<Args>(args)...);
        }

        /**
         * @brief Log a debug-level message.
         */
        template <typename... Args>
        void debug(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::debug, fmt, std::forward<Args>(args)...);
        }

        /**
         * @brief Log an info-level message.
         */
        template <typename... Args>
        void info(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::info, fmt, std::forward<Args>(args)...);
        }

        /**
         * @brief Log a warning-level message.
         */
        template <typename... Args>
        void warn(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::warning, fmt, std::forward<Args>(args)...);
        }

        /**
         * @brief Log an error-level message.
         */
        template <typename... Args>
        void error(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::error, fmt, std::forward<Args>(args)...);
        }

        /**
         * @brief Log a fatal-level message.
         */
        template <typename... Args>
        void critical(std::string_view fmt, Args&&... args)
        {
            logImpl(boost::log::trivial::fatal, fmt, std::forward<Args>(args)...);
        }

    private:
        /**
         * @brief Convert a value to string using stream insertion.
         */
        template <typename T>
        static std::string stringifyValue(T&& value)
        {
            std::ostringstream oss;
            oss << std::forward<T>(value);
            return oss.str();
        }

        /**
         * @brief Format message with a simple "{}" placeholder substitution.
         */
        template <typename... Args>
        static std::string formatMessage(std::string_view fmt, Args&&... args)
        {
            std::vector<std::string> arg_strings;
            arg_strings.reserve(sizeof...(Args));

            // Convert all arguments to strings first.
            (arg_strings.emplace_back(stringifyValue(std::forward<Args>(args))), ...);

            // Replace "{}" sequentially.
            std::ostringstream out;
            std::size_t pos = 0;
            std::size_t arg_idx = 0;

            while (true)
            {
                const std::size_t brace_pos = fmt.find("{}", pos);
                if (brace_pos == std::string_view::npos)
                {
                    out << fmt.substr(pos);
                    break;
                }

                out << fmt.substr(pos, brace_pos - pos);

                if (arg_idx < arg_strings.size())
                {
                    out << arg_strings[arg_idx];
                    ++arg_idx;
                }
                else
                {
                    out << "{}";
                }

                pos = brace_pos + 2;
            }

            // If extra args remain, append them space-separated.
            if (arg_idx < arg_strings.size())
            {
                out << " ";
                for (std::size_t i = arg_idx; i < arg_strings.size(); ++i)
                {
                    if (i != arg_idx)
                    {
                        out << " ";
                    }
                    out << arg_strings[i];
                }
            }

            return out.str();
        }

        /**
         * @brief Emit a formatted message with a specific Boost.Log severity.
         */
        template <typename... Args>
        void logImpl(boost::log::trivial::severity_level severity,
                    std::string_view fmt,
                    Args&&... args)
        {
            // Ensure Boost.Log is initialized before logging.
            Logger::init();

            const std::string message = formatMessage(fmt, std::forward<Args>(args)...);

            // Attach logger name in-message to keep formatting simple.
            switch (severity)
            {
                case boost::log::trivial::trace:
                    BOOST_LOG_TRIVIAL(trace) << "[" << logger_name_ << "] " << message;
                    break;
                case boost::log::trivial::debug:
                    BOOST_LOG_TRIVIAL(debug) << "[" << logger_name_ << "] " << message;
                    break;
                case boost::log::trivial::info:
                    BOOST_LOG_TRIVIAL(info) << "[" << logger_name_ << "] " << message;
                    break;
                case boost::log::trivial::warning:
                    BOOST_LOG_TRIVIAL(warning) << "[" << logger_name_ << "] " << message;
                    break;
                case boost::log::trivial::error:
                    BOOST_LOG_TRIVIAL(error) << "[" << logger_name_ << "] " << message;
                    break;
                case boost::log::trivial::fatal:
                    BOOST_LOG_TRIVIAL(fatal) << "[" << logger_name_ << "] " << message;
                    break;
            }
        }

        std::string logger_name_;
    };

    /**
     * @brief Construct a named logger.
     */
    explicit Logger(const char* logger_name) noexcept;

    /**
     * @brief Deleted copy constructor.
     */
    Logger(const Logger&) = delete;

    /**
     * @brief Deleted copy assignment.
     */
    Logger& operator=(const Logger&) = delete;

    /**
     * @brief Get the global singleton logger.
     */
    static Logger& getInstance() noexcept;

    /**
     * @brief Initialize logging with default options (first call wins).
     */
    static void init();

    /**
     * @brief Initialize logging with custom options (first call wins).
     */
    static void init(const Options& options);

    /**
     * @brief Return whether logging has been initialized.
     */
    static bool isInitialized() noexcept;

    /**
     * @brief Set minimum severity level filter.
     */
    static void setMinLevel(Level level);

    /**
     * @brief Get a spdlog-like adapter for call sites.
     */
    Adapter* getLogger() noexcept;

private:
    /**
     * @brief Convert project level to Boost.Log level.
     */
    static boost::log::trivial::severity_level toBoostLevel(Level level);

    /**
     * @brief Build default options from compile-time switches.
     */
    static Options buildDefaultOptions();

    /**
     * @brief Build default log file path "aker_log_<timestamp>.log".
     */
    static std::string buildDefaultLogFilePath();

    /**
     * @brief Internal initialization routine.
     */
    static void initImpl(const Options& options);

    inline static std::once_flag init_once_flag_;
    inline static std::atomic<bool> initialized_{false};

    std::string logger_name_;
    Adapter adapter_;
};

} // namespace aker

// ================================
// Inline definitions (header-only)
// ================================

inline aker::Logger::Adapter::Adapter(std::string logger_name)
    : logger_name_(std::move(logger_name))
{
}

inline aker::Logger::Logger(const char* logger_name) noexcept
    : logger_name_(logger_name ? logger_name : "log"),
      adapter_(logger_name_)
{
    // Default initialization is intentionally eager to avoid missing logs.
    Logger::init();
}

inline aker::Logger& aker::Logger::getInstance() noexcept
{
    static Logger global_logger("log");
    return global_logger;
}

inline void aker::Logger::init()
{
    init(buildDefaultOptions());
}

inline void aker::Logger::init(const Options& options)
{
    std::call_once(init_once_flag_, [&]() { initImpl(options); });
}

inline bool aker::Logger::isInitialized() noexcept
{
    return initialized_.load(std::memory_order_acquire);
}

inline void aker::Logger::setMinLevel(Level level)
{
    boost::log::core::get()->set_filter(
        boost::log::trivial::severity >= toBoostLevel(level));
}

inline aker::Logger::Adapter* aker::Logger::getLogger() noexcept
{
    return &adapter_;
}

inline boost::log::trivial::severity_level aker::Logger::toBoostLevel(Level level)
{
    using boost_level = boost::log::trivial::severity_level;
    switch (level)
    {
        case Level::k_trace:
            return boost_level::trace;
        case Level::k_debug:
            return boost_level::debug;
        case Level::k_info:
            return boost_level::info;
        case Level::k_warning:
            return boost_level::warning;
        case Level::k_error:
            return boost_level::error;
        case Level::k_fatal:
            return boost_level::fatal;
    }
    return boost_level::info;
}

inline void aker::Logger::initImpl(const Options& options)
{
    namespace logging = boost::log;
    namespace expr = boost::log::expressions;

    // Setup common attributes (timestamp, thread id, etc.).
    logging::add_common_attributes();

    // Define a simple formatter consistent across sinks.
    const auto formatter =
        (expr::stream
         << "[" << expr::attr<boost::posix_time::ptime>("TimeStamp") << "] "
         << "[" << logging::trivial::severity << "] "
         << expr::smessage);

    // Console sink.
    if (options.console)
    {
        std::ostream& out = *(options.console_stream ? options.console_stream : &std::clog);
        logging::add_console_log(out, logging::keywords::format = formatter);
    }

    // File sink (optional).
    if (options.file_path && !options.file_path->empty())
    {
        logging::add_file_log(
            logging::keywords::file_name = *options.file_path,
            logging::keywords::rotation_size = options.rotation_size_bytes,
            logging::keywords::auto_flush = options.auto_flush,
            logging::keywords::format = formatter);
    }

    // Apply minimum severity level.
    logging::core::get()->set_filter(
        logging::trivial::severity >= toBoostLevel(options.min_level));

    initialized_.store(true, std::memory_order_release);
}

inline aker::Logger::Options aker::Logger::buildDefaultOptions()
{
    Options options;

    // Configure console sink.
    options.console = (AKER_LOG_TO_CONSOLE != 0);
    options.console_stream = &std::clog;

    // Configure optional file sink.
    if (AKER_LOG_TO_FILE != 0)
    {
        options.file_path = buildDefaultLogFilePath();
        options.auto_flush = (AKER_LOG_AUTO_FLUSH != 0);
    }

    return options;
}

inline std::string aker::Logger::buildDefaultLogFilePath()
{
    std::time_t now = std::time(nullptr);
    std::tm local_tm{};

#if defined(_WIN32)
    localtime_s(&local_tm, &now);
#else
    localtime_r(&now, &local_tm);
#endif

    std::ostringstream oss;
    oss << "aker_log_" << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << ".log";
    return oss.str();
}

/**
 * @brief Initialize the global logger with default options.
 */
#define AKER_LOG_INIT() ::aker::Logger::init()

/**
 * @brief Streaming-friendly logging macros.
 */
#define AKER_LOG_TRACE BOOST_LOG_TRIVIAL(trace)
#define AKER_LOG_DEBUG BOOST_LOG_TRIVIAL(debug)
#define AKER_LOG_INFO  BOOST_LOG_TRIVIAL(info)
#define AKER_LOG_WARN  BOOST_LOG_TRIVIAL(warning)
#define AKER_LOG_ERROR BOOST_LOG_TRIVIAL(error)
#define AKER_LOG_FATAL BOOST_LOG_TRIVIAL(fatal)

#else  // AKER_ENABLE_LOGGING

namespace aker
{
    /**
     * @brief No-op logger facade when logging is disabled.
     */
    class Logger final
    {
    public:
        /**
         * @brief Logging level.
         */
        enum class Level
        {
            k_trace,
            k_debug,
            k_info,
            k_warning,
            k_error,
            k_fatal,
        };

        /**
         * @brief Logger initialization options.
         */
        struct Options
        {
            Level min_level = Level::k_info;
        };

        /**
         * @brief A minimal adapter kept for source compatibility.
         */
        class Adapter final
        {
        public:
            explicit Adapter(std::string /*logger_name*/) {}

            template <typename... Args>
            void trace(std::string_view /*fmt*/, Args&&... /*args*/) {}

            template <typename... Args>
            void debug(std::string_view /*fmt*/, Args&&... /*args*/) {}

            template <typename... Args>
            void info(std::string_view /*fmt*/, Args&&... /*args*/) {}

            template <typename... Args>
            void warn(std::string_view /*fmt*/, Args&&... /*args*/) {}

            template <typename... Args>
            void error(std::string_view /*fmt*/, Args&&... /*args*/) {}

            template <typename... Args>
            void critical(std::string_view /*fmt*/, Args&&... /*args*/) {}
        };

        explicit Logger(const char* /*logger_name*/) noexcept {}
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        static Logger& getInstance() noexcept
        {
            static Logger instance("log");
            return instance;
        }

        static void init() {}
        static void init(const Options& /*options*/) {}
        static bool isInitialized() noexcept { return false; }
        static void setMinLevel(Level /*level*/) {}

        Adapter* getLogger() noexcept { return &adapter_; }

    private:
        Adapter adapter_{"log"};
    };
}

namespace aker::detail
{
    /**
     * @brief A sink-like object that discards streamed values.
     */
    class NullLogStream final
    {
    public:
        template <typename T>
        NullLogStream& operator<<(const T& /*value*/) noexcept
        {
            return *this;
        }
    };
}

#define AKER_LOG_INIT() do {} while (0)

#define AKER_LOG_TRACE ::aker::detail::NullLogStream()
#define AKER_LOG_DEBUG ::aker::detail::NullLogStream()
#define AKER_LOG_INFO  ::aker::detail::NullLogStream()
#define AKER_LOG_WARN  ::aker::detail::NullLogStream()
#define AKER_LOG_ERROR ::aker::detail::NullLogStream()
#define AKER_LOG_FATAL ::aker::detail::NullLogStream()

#endif  // AKER_ENABLE_LOGGING

