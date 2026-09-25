#pragma once

#include "qvcache/backend_interface.h"

#include <libpq-fe.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qvcache {

inline bool pgvector_valid_ident(const std::string& name) {
    if (name.empty() || (!std::isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_')) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    });
}

template <typename T, typename TagT = uint32_t>
class PgVectorBackend : public BackendInterface<T, TagT> {
public:
    PgVectorBackend(const std::string& table_name,
                    const std::string& data_path,
                    const std::string& db_host,
                    int db_port,
                    const std::string& db_name,
                    const std::string& db_user,
                    const std::string& db_password,
                    const std::string& metric = "l2",
                    int hnsw_ef_search = 200)
        : table_name_(table_name),
          metric_(metric),
          hnsw_ef_search_(std::max(1, std::min(hnsw_ef_search, 1000))),
          vector_data_(nullptr),
          num_vectors_(0),
          dim_(0) {
        if (hnsw_ef_search > 1000) {
            std::cerr << "Warning: hnsw.ef_search " << hnsw_ef_search
                      << " exceeds pgvector max 1000; clamping to 1000" << std::endl;
        }
        if (!pgvector_valid_ident(table_name_)) {
            throw std::runtime_error("Invalid pgvector table name: " + table_name_);
        }
        std::string metric_lower = metric_;
        std::transform(metric_lower.begin(), metric_lower.end(), metric_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (metric_lower != "l2" && metric_lower != "cosine") {
            throw std::runtime_error("Unsupported pgvector metric: " + metric);
        }
        metric_ = metric_lower;
        distance_op_ = (metric_ == "cosine") ? "<=>" : "<->";

        std::ostringstream conninfo;
        conninfo << "host=" << db_host
                 << " port=" << db_port
                 << " dbname=" << db_name
                 << " user=" << db_user
                 << " password=" << db_password;
        conninfo_ = conninfo.str();

        load_vectors_from_file(data_path);
        PGconn* c = conn();
        PGresult* res = PQexec(c, ("SELECT COUNT(*) FROM " + table_name_).c_str());
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            std::string err = PQerrorMessage(c);
            PQclear(res);
            throw std::runtime_error("Failed to read pgvector table " + table_name_ + ": " + err);
        }
        const long table_rows = std::strtol(PQgetvalue(res, 0, 0), nullptr, 10);
        PQclear(res);
        std::cout << "PgVectorBackend connected to " << db_host << ":" << db_port
                  << " table=" << table_name_ << " rows=" << table_rows
                  << " local_vectors=" << num_vectors_ << " dim=" << dim_
                  << " metric=" << metric_ << " hnsw.ef_search=" << hnsw_ef_search_ << std::endl;
    }

    ~PgVectorBackend() {
        delete[] vector_data_;
        vector_data_ = nullptr;
    }

    void search(
        const T *query,
        uint64_t K,
        TagT* result_tags,
        float* result_distances,
        void* search_parameters = nullptr,
        void* stats = nullptr) override {
        (void)search_parameters;
        (void)stats;

        const TagT invalid = std::numeric_limits<TagT>::max();
        for (uint64_t i = 0; i < K; ++i) {
            result_tags[i] = invalid;
            result_distances[i] = std::numeric_limits<float>::infinity();
        }

        PGconn* c = conn();

        const std::string vec_str = format_vector(query);
        const std::string k_str = std::to_string(K);
        const std::string sql =
            "SELECT id, vector " + distance_op_ + " $1::vector AS distance FROM " + table_name_ +
            " ORDER BY vector " + distance_op_ + " $1::vector LIMIT $2";
        const char* params[2] = {vec_str.c_str(), k_str.c_str()};
        PGresult* res = PQexecParams(c, sql.c_str(), 2, nullptr, params, nullptr, nullptr, 0);
        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            std::string err = PQerrorMessage(c);
            PQclear(res);
            throw std::runtime_error("pgvector search failed: " + err);
        }

        const int n = std::min(PQntuples(res), static_cast<int>(K));
        for (int i = 0; i < n; ++i) {
            result_tags[i] = static_cast<TagT>(std::strtoul(PQgetvalue(res, i, 0), nullptr, 10));
            const float distance = std::strtof(PQgetvalue(res, i, 1), nullptr);
            result_distances[i] = (metric_ == "cosine") ? distance : (distance * distance);
        }
        PQclear(res);
    }

    std::vector<std::vector<T>> fetch_vectors_by_ids(const std::vector<TagT>& ids) override {
        std::vector<std::vector<T>> out;
        out.reserve(ids.size());
        if (vector_data_ == nullptr) {
            throw std::runtime_error("PgVectorBackend has no local vectors for fetch_vectors_by_ids");
        }
        for (TagT id : ids) {
            std::vector<T> vec(dim_, static_cast<T>(0));
            if (static_cast<size_t>(id) < num_vectors_) {
                std::memcpy(vec.data(), vector_data_ + static_cast<size_t>(id) * dim_, dim_ * sizeof(T));
            }
            out.push_back(std::move(vec));
        }
        return out;
    }

private:
    struct ConnDeleter {
        void operator()(PGconn* c) const {
            if (c != nullptr) {
                PQfinish(c);
            }
        }
    };

    PGconn* conn() {
        thread_local std::unique_ptr<PGconn, ConnDeleter> tls;
        if (!tls) {
            tls.reset(PQconnectdb(conninfo_.c_str()));
            if (PQstatus(tls.get()) != CONNECTION_OK) {
                std::string err = PQerrorMessage(tls.get());
                tls.reset();
                throw std::runtime_error("Failed to connect to PostgreSQL: " + err);
            }
            auto exec_set = [&](const std::string& sql, bool required) {
                PGresult* set_res = PQexec(tls.get(), sql.c_str());
                if (PQresultStatus(set_res) != PGRES_COMMAND_OK) {
                    std::string err = PQerrorMessage(tls.get());
                    PQclear(set_res);
                    if (required) {
                        throw std::runtime_error("Failed to run " + sql + ": " + err);
                    }
                    std::cerr << "Warning: " << sql << " failed: " << err << std::endl;
                    return;
                }
                PQclear(set_res);
            };
            exec_set("SET hnsw.ef_search = " + std::to_string(hnsw_ef_search_), true);
            // pgvector >= 0.8: keep scanning past ef_search to raise recall on OOD/HNSW.
            exec_set("SET hnsw.iterative_scan = relaxed_order", false);
            exec_set("SET hnsw.max_scan_tuples = 200000", false);
            exec_set("SET hnsw.scan_mem_multiplier = 2", false);
        }
        return tls.get();
    }

    std::string format_vector(const T* query) const {
        std::string s;
        s.reserve(dim_ * 12 + 2);
        s.push_back('[');
        for (size_t i = 0; i < dim_; ++i) {
            if (i > 0) {
                s.push_back(',');
            }
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.6f", static_cast<double>(query[i]));
            s += buf;
        }
        s.push_back(']');
        return s;
    }

    void load_vectors_from_file(const std::string& data_path) {
        std::ifstream in(data_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open " + data_path);
        }
        uint32_t npts = 0, dim = 0;
        in.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
        if (!in) {
            throw std::runtime_error("Failed to read DiskANN header from " + data_path);
        }
        num_vectors_ = npts;
        dim_ = dim;
        vector_data_ = new T[num_vectors_ * dim_];
        in.read(reinterpret_cast<char*>(vector_data_),
                static_cast<std::streamsize>(num_vectors_ * dim_ * sizeof(T)));
        if (!in) {
            delete[] vector_data_;
            vector_data_ = nullptr;
            throw std::runtime_error("Failed to read vectors from " + data_path);
        }
    }

    std::string table_name_;
    std::string metric_;
    std::string distance_op_;
    std::string conninfo_;
    int hnsw_ef_search_;
    T* vector_data_;
    size_t num_vectors_;
    size_t dim_;
};

}  // namespace qvcache
