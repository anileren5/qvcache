// 
// Author: Sukjoon Oh

#ifndef AKER_OPTIONPARSER_H
#define AKER_OPTIONPARSER_H

#include "ak_anns_cache_config.hh"

#include <map>
#include <string>

namespace aker
{
    /**
     * @brief Loads ANNSCache parameters from a bootstrap config file.
     *
     * The loader accepts:
     * - INI (.ini): preferred human-friendly format
     * - JSON (.json): legacy flat key-value format
     */
    class ParameterParser final
    {
    private:
        std::string file_path;
        anns_cache_config_t parameter;

        std::map<std::string, std::string> parameter_map;

    public:
        explicit ParameterParser(std::string path);

        anns_cache_config_t getParameter() const noexcept;
    };
}

#endif
