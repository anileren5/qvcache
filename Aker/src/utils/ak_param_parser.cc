// 
// Author: Sukjoon Oh

#include "utils/ak_param_parser.hh"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <sstream>
#include <utility>

#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "ak_logger.hh"

namespace aker
{
    namespace
    {
        static inline bool isSpace(unsigned char c) { return std::isspace(c) != 0; }

        static std::string trimCopy(std::string s)
        {
            auto not_space = [](unsigned char c) { return !isSpace(c); };
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
            s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
            return s;
        }

        static std::string stripInlineComment(std::string value)
        {
            for (std::size_t i = 0; i < value.size(); ++i)
            {
                const char c = value[i];
                if (c == '#' || c == ';')
                {
                    if (i == 0 || isSpace(static_cast<unsigned char>(value[i - 1])))
                    {
                        value.resize(i);
                        break;
                    }
                }
            }
            return trimCopy(std::move(value));
        }

        static std::string toLower(std::string s)
        {
            for (auto& ch : s)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return s;
        }

        static bool endsWithNoCase(const std::string& s, const std::string& suffix)
        {
            if (s.size() < suffix.size())
                return false;

            const std::size_t offset = s.size() - suffix.size();
            for (std::size_t i = 0; i < suffix.size(); ++i)
            {
                const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[offset + i])));
                const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
                if (a != b)
                    return false;
            }
            return true;
        }

        static void flattenPtree(
            const boost::property_tree::ptree& node,
            const std::string& prefix,
            std::map<std::string, std::string>& out)
        {
            for (const auto& item : node)
            {
                const std::string& key = item.first;
                const boost::property_tree::ptree& child = item.second;

                const std::string full_key = prefix.empty() ? key : (prefix + "." + key);

                /* If the node is a leaf, store its value.
                 * If it has children (INI section / nested JSON), recurse.
                 */
                if (child.empty())
                {
                    std::string value = stripInlineComment(child.get_value<std::string>());
                    out[full_key] = value;
                }
                else
                {
                    std::string value = stripInlineComment(child.get_value<std::string>());
                    if (!value.empty())
                        out[full_key] = value;

                    flattenPtree(child, full_key, out);
                }
            }
        }
    }

    ParameterParser::ParameterParser(std::string path)
        : file_path(std::move(path))
    {
        boost::property_tree::ptree pt;

        bool parsed_as_ini = false;

        /* Prefer INI format when the extension is .ini.
         * Keep JSON support for backward compatibility.
         */
        if (endsWithNoCase(file_path, ".ini"))
        {
            boost::property_tree::read_ini(file_path, pt);
            parsed_as_ini = true;
        }
        else if (endsWithNoCase(file_path, ".json"))
        {
            boost::property_tree::read_json(file_path, pt);
            parsed_as_ini = false;
        }
        else
        {
            AKER_LOG_ERROR << "Unsupported config file format: " << file_path;
            assert(false);
        }

        /* Flatten the property tree so that both of these styles work:
         * - Flat keys:   dimension=128
         * - Sections:    [vector_format] dimension=128
         *              -> key becomes vector_format.dimension
         */
        flattenPtree(pt, "", parameter_map);

        /* Helper that accepts refined keys, sectioned keys, and legacy keys.
         */
        const auto getAny = [&](std::initializer_list<const char*> keys) -> std::string
        {
            for (const char* key : keys)
            {
                auto it = parameter_map.find(key);
                if (it != parameter_map.end() && !it->second.empty())
                    return it->second;
            }
            return std::string();
        };

        /* Required core parameters.
         */
        const auto requireValue = [&](const char* name, const std::string& value) {
            if (!value.empty())
                return;
            AKER_LOG_ERROR << "[ParameterParser] missing required parameter: " << name << " (file=" << file_path << ")";
            assert(false);
        };

        std::string dimension_str = getAny({"dimension", "vector_format.dimension", "vector_dim", "vector_format.vector_dim"});
        requireValue("dimension", dimension_str);
        parameter.vector_format.dimension = static_cast<std::uint32_t>(std::stoul(dimension_str));

        std::string pool_size_str = getAny({"pool_size", "capacity.pool_size", "slot_pool_size", "vector_pool_size"});
        requireValue("pool_size", pool_size_str);
        parameter.capacity.pool_size = static_cast<size_t>(std::stoull(pool_size_str));

        std::string vector_in_bytes_str = getAny({"vector_in_bytes", "vector_format.vector_in_bytes", "vector_data_size", "vector_format.vector_data_size"});
        requireValue("vector_in_bytes", vector_in_bytes_str);
        parameter.vector_format.vector_in_bytes = static_cast<size_t>(std::stoull(vector_in_bytes_str));

        std::string in_topk_str = getAny({"in_topk", "capacity.in_topk", "vector_in_topk", "vector_intopk"});
        requireValue("in_topk", in_topk_str);
        parameter.capacity.in_topk = static_cast<size_t>(std::stoull(in_topk_str));

        std::string top_delta_str = getAny({"top_delta", "capacity.top_delta", "vector_extras"});
        requireValue("top_delta", top_delta_str);
        parameter.capacity.top_delta = static_cast<size_t>(std::stoull(top_delta_str));

        /* Optional legacy key: slot_list_size.
         * The refined configuration derives it as: in_topk + top_delta.
         */
        std::string slot_list_size_str = getAny({"slot_list_size", "capacity.slot_list_size", "vector_list_size"});
        if (!slot_list_size_str.empty())
        {
            const size_t legacy_list_size = static_cast<size_t>(std::stoull(slot_list_size_str));
            const size_t derived_list_size = parameter.capacity.getSlotListSize();
            if (legacy_list_size != derived_list_size)
            {
                AKER_LOG_ERROR << "[ParameterParser] slot_list_size mismatch: legacy=" << legacy_list_size
                              << " derived=" << derived_list_size;
                assert(false);
            }
        }

        /* Tuning parameters.
         */
        std::string global_thresh_str = getAny({"global_thresh", "tuning.global_thresh", "fixed_thresh", "tuning.fixed_thresh"});
        if (!global_thresh_str.empty())
            parameter.tuning.global_thresh = std::stof(global_thresh_str);
        else
            parameter.tuning.global_thresh = 0.0f;

        std::string dropout_str = getAny({"dropout", "tuning.dropout"});
        if (!dropout_str.empty())
            parameter.tuning.dropout = std::stof(dropout_str);
        else
            parameter.tuning.dropout = 0.0f;

        std::string risk_thresh_str = getAny({"risk_thresh", "tuning.risk_thresh"});
        if (!risk_thresh_str.empty())
            parameter.tuning.risk_thresh = std::stof(risk_thresh_str);
        else
            parameter.tuning.risk_thresh = 0.0f;

        std::string alpha_tighten_str = getAny({"alpha_tighten", "tuning.alpha_tighten"});
        if (!alpha_tighten_str.empty())
            parameter.tuning.alpha_tighten = std::stof(alpha_tighten_str);
        else
            parameter.tuning.alpha_tighten = 0.0f;

        std::string alpha_loosen_str = getAny({"alpha_loosen", "tuning.alpha_loosen"});
        if (!alpha_loosen_str.empty())
            parameter.tuning.alpha_loosen = std::stof(alpha_loosen_str);
        else
            parameter.tuning.alpha_loosen = 0.0f;

        std::string distance_metric_str = getAny({"distance_metric", "distance.distance_metric", "distance_type"});
        requireValue("distance_metric", distance_metric_str);

        const std::string distance_metric_norm = toLower(distance_metric_str);
        if (distance_metric_norm == "l2")
            parameter.distance_metric = distance_metric_t::DISTANCE_METRIC_L2;
        else if (distance_metric_norm == "ip")
            parameter.distance_metric = distance_metric_t::DISTANCE_METRIC_IP;
        else
        {
            AKER_LOG_ERROR << "[ParameterParser] invalid distance_metric: " << distance_metric_str;
            assert(false);
        }

        /* Mode-specific requirements and ignored options.
         */
#if defined(AKER_ENABLE_PROXIMITY_MODE) && (AKER_ENABLE_PROXIMITY_MODE != 0)
        /* Proximity Mode
         * - Requires global_thresh (used for similarity checks).
         * - Ignores dropout and write-log related parameters.
         */
        requireValue("global_thresh", global_thresh_str);
        if (parameter.tuning.global_thresh <= 0.0f)
        {
            AKER_LOG_ERROR << "[ParameterParser] proximity mode requires global_thresh > 0 (global_thresh="
                          << parameter.tuning.global_thresh << ")";
            assert(false);
        }

        if (!dropout_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] proximity mode ignores dropout=" << dropout_str;
            parameter.tuning.dropout = 0.0f;
        }

        if (!risk_thresh_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] proximity mode ignores risk_thresh=" << risk_thresh_str;
            parameter.tuning.risk_thresh = 0.0f;
        }

        if (!alpha_tighten_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] proximity mode ignores alpha_tighten=" << alpha_tighten_str;
            parameter.tuning.alpha_tighten = 0.0f;
        }

        if (!alpha_loosen_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] proximity mode ignores alpha_loosen=" << alpha_loosen_str;
            parameter.tuning.alpha_loosen = 0.0f;
        }

#elif defined(AKER_ENABLE_POTLUCK_MODE) && (AKER_ENABLE_POTLUCK_MODE != 0)
        /* Potluck Mode
         * - Uses global_thresh (always starts from 0.0) and tunes it at put().
         * - Requires dropout rate in [0, 100].
         */
        if (!global_thresh_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] potluck mode ignores global_thresh=" << global_thresh_str
                          << " (global_thresh is forced to 0.0 at startup)";
        }
        parameter.tuning.global_thresh = 0.0f;

        requireValue("dropout", dropout_str);

        if (parameter.tuning.dropout < 0.0f || parameter.tuning.dropout > 100.0f)
        {
            AKER_LOG_ERROR << "[ParameterParser] potluck mode requires dropout in [0, 100] (dropout="
                          << parameter.tuning.dropout << ")";
            assert(false);
        }

        /* Potluck threshold tuning uses alpha_tighten. */
        requireValue("alpha_tighten", alpha_tighten_str);
        if (parameter.tuning.alpha_tighten >= 1.0f)
        {
            AKER_LOG_ERROR << "[ParameterParser] potluck mode expects alpha_tighten < 1 (alpha_tighten="
                          << parameter.tuning.alpha_tighten << ")";
            assert(false);
        }

        if (!risk_thresh_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] potluck mode ignores risk_thresh=" << risk_thresh_str;
            parameter.tuning.risk_thresh = 0.0f;
        }

        if (!alpha_loosen_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] potluck mode ignores alpha_loosen=" << alpha_loosen_str;
            parameter.tuning.alpha_loosen = 0.0f;
        }

#else
        /* Standard Mode
         * - Uses per-entry adaptive thresholds.
         * - Requires write-log parameters (risk_thresh, alpha_tighten, alpha_loosen).
         * - Ignores global_thresh/dropout.
         */
        requireValue("risk_thresh", risk_thresh_str);
        requireValue("alpha_tighten", alpha_tighten_str);
        requireValue("alpha_loosen", alpha_loosen_str);

        if (parameter.tuning.alpha_loosen <= 1.0f)
        {
            AKER_LOG_ERROR << "[ParameterParser] standard mode requires alpha_loosen > 1 (alpha_loosen="
                          << parameter.tuning.alpha_loosen << ")";
            assert(false);
        }

        if (parameter.tuning.alpha_tighten >= 1.0f)
        {
            AKER_LOG_ERROR << "[ParameterParser] standard mode requires alpha_tighten < 1 (alpha_tighten="
                          << parameter.tuning.alpha_tighten << ")";
            assert(false);
        }

        if (!global_thresh_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] standard mode ignores global_thresh=" << global_thresh_str;
            parameter.tuning.global_thresh = 0.0f;
        }

        if (!dropout_str.empty())
        {
            AKER_LOG_INFO << "[ParameterParser] standard mode ignores dropout=" << dropout_str;
            parameter.tuning.dropout = 0.0f;
        }
#endif

        /* Emit a configuration snapshot for debugging.
         */
        AKER_LOG_INFO << "[ParameterParser] loaded parameters from " << file_path
                      << " (format=" << (parsed_as_ini ? "ini" : "json") << ")";
        AKER_LOG_INFO << "  dimension=" << parameter.vector_format.dimension;
        AKER_LOG_INFO << "  pool_size=" << parameter.capacity.pool_size;
        AKER_LOG_INFO << "  vector_in_bytes=" << parameter.vector_format.vector_in_bytes;
        AKER_LOG_INFO << "  in_topk=" << parameter.capacity.in_topk;
        AKER_LOG_INFO << "  top_delta=" << parameter.capacity.top_delta;
        AKER_LOG_INFO << "  global_thresh=" << parameter.tuning.global_thresh;
        AKER_LOG_INFO << "  dropout=" << parameter.tuning.dropout;
        AKER_LOG_INFO << "  risk_thresh=" << parameter.tuning.risk_thresh;
        AKER_LOG_INFO << "  alpha_tighten=" << parameter.tuning.alpha_tighten;
        AKER_LOG_INFO << "  alpha_loosen=" << parameter.tuning.alpha_loosen;
        AKER_LOG_INFO << "  distance_metric=" << distance_metric_str;
    }

    anns_cache_config_t
    ParameterParser::getParameter() const noexcept
    {
        return parameter;
    }

} // namespace aker
