#pragma once

#include <cstddef>
#include <cstdint>

#include <functional>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "ak_vector_slot.hh"

namespace aker
{
    /**
     * @brief Monotonic epoch type used by the write-log.
     */
    using epoch_t = std::uint64_t;

    /**
     * @brief Opaque handle representing a cache entry registered in the write-log RR list.
     */
    using cache_entry_handle_t = void*;

    /**
     * @brief Opaque checkpoint handle into the write-log.
     *
     * This is intentionally an incomplete type in the public header.
     * Callers must not access internal fields directly.
     */
    struct WriteLogEntryNode;
    using write_log_checkpoint_t = WriteLogEntryNode*;

    /**
     * @brief Distance function type used by cache and write-log.
     */
    using distance_function_t =
        std::function<float(const vector_data_t* vector1, const vector_data_t* vector2, size_t dimension)>;

    /**
     * @brief Result conversion callback used by cache maintenance paths.
     */
    using result_transform_callback_t =
        std::function<void(vector_id_t, vector_data_t*, size_t, std::uint64_t, std::uint64_t)>;

    /**
     * @brief One close candidate returned by a write-log scan.
     */
    struct WriteLogCandidate
    {
        float distance{0.0f};
        vector_id_t vector_id{0};
        const vector_data_t* vector_data{nullptr};
        size_t vector_in_bytes{0};
        aux_data_t aux_data_1{0};
        aux_data_t aux_data_2{0};

        /**
         * @brief Internal node handle pinned for the lifetime of the scan result.
         */
        write_log_checkpoint_t node{nullptr};
    };

    /**
     * @brief Write-log health and activity metrics snapshot.
     */
    struct WriteLogMetrics
    {
        size_t log_entry_count{0};
        epoch_t latest_epoch{0};
        epoch_t head_epoch{0};
        epoch_t tail_epoch{0};

        double current_risk{0.0};
        double total_risk_factor{0.0};
        double average_risk_factor{0.0};
        double total_unseen{0.0};
        double total_cache_entries{0.0};

        std::uint64_t insert_count{0};
        std::uint64_t trim_count{0};
        std::uint64_t slow_path_checked_count{0};
        std::uint64_t refresh_count{0};
        std::uint64_t duplicate_insert_count{0};

        size_t round_robin_count{0};
    };

    /**
     * @brief RAII holder for pinned write-log nodes.
     *
     * This is used to keep candidate nodes alive while the caller consumes the scan result
     * without holding the write-log lock.
     */
    class WriteLogScanResult
    {
    public:
        /**
         * @brief Constructs an empty scan result.
         */
        WriteLogScanResult() noexcept = default;

        /**
         * @brief Destructor releases pinned nodes.
         */
        ~WriteLogScanResult() noexcept;

        WriteLogScanResult(const WriteLogScanResult&) = delete;
        WriteLogScanResult& operator=(const WriteLogScanResult&) = delete;

        /**
         * @brief Move constructor.
         */
        WriteLogScanResult(WriteLogScanResult&& other) noexcept;

        /**
         * @brief Move assignment.
         */
        WriteLogScanResult& operator=(WriteLogScanResult&& other) noexcept;

        /**
         * @brief New checkpoint (already retained by the write-log).
         */
        write_log_checkpoint_t new_checkpoint{nullptr};

        /**
         * @brief Epoch distance advanced from old checkpoint to new checkpoint.
         */
        epoch_t advanced_epoch_distance{0};

        /**
         * @brief Candidates within the distance thresh.
         */
        std::vector<WriteLogCandidate> candidates;

    private:
        friend class RiskAwareWriteLog;

        /**
         * @brief Releases pinned nodes owned by this result.
         */
        void releasePinnedNodes() noexcept;

        class RiskAwareWriteLog* owner_{nullptr};
        std::vector<write_log_checkpoint_t> pinned_nodes_;
    };

    /**
     * @brief Risk-aware write-log that tracks insert events and refreshes cached results.
     *
     * The log stores a bounded history of inserted vectors (tombstones are handled
     * externally by the approximate filter) and supports:
     * - Round-robin selection of cache entries for slow-path maintenance.
     * - Checkpoint-based scanning of a limited window of unseen log entries.
     * - Safe trimming of unreferenced head entries.
     */
    class RiskAwareWriteLog
    {
    public:
        /**
         * @brief Constructs the write-log.
         */
        RiskAwareWriteLog(size_t in_topk, size_t scan_thresh, double allowed_risk) noexcept;

        /**
         * @brief Destructor frees all log entries.
         */
        ~RiskAwareWriteLog() noexcept;

        RiskAwareWriteLog(const RiskAwareWriteLog&) = delete;
        RiskAwareWriteLog& operator=(const RiskAwareWriteLog&) = delete;

        /**
         * @brief Inserts one write-log entry by copying raw vector bytes.
         */
        void insertLogEntry(
            vector_id_t vector_id,
            const vector_data_t* vector_data,
            size_t vector_in_bytes,
            aux_data_t aux_data_1,
            aux_data_t aux_data_2) noexcept;

        /**
         * @brief Registers a cache entry handle into the round-robin list.
         */
        void addCacheEntryToRoundRobin(cache_entry_handle_t cache_entry) noexcept;

        /**
         * @brief Returns the next RR cache entry handle.
         */
        cache_entry_handle_t getNextCacheEntryFromRoundRobin() noexcept;

        /**
         * @brief Removes a cache entry handle from the RR list.
         */
        bool removeCacheEntryFromRoundRobin(cache_entry_handle_t cache_entry) noexcept;

        /**
         * @brief Retains and returns the current tail checkpoint.
         */
        write_log_checkpoint_t acquireTailCheckpoint() noexcept;

        /**
         * @brief Releases one checkpoint reference.
         */
        void releaseCheckpoint(write_log_checkpoint_t checkpoint) noexcept;

        /**
         * @brief Replaces a cache entry checkpoint (releases old, adopts already-retained new).
         */
        void replaceCheckpoint(write_log_checkpoint_t& checkpoint_slot, write_log_checkpoint_t new_checkpoint) noexcept;

        /**
         * @brief Computes unseen epoch distance from a checkpoint to the latest epoch.
         */
        epoch_t getUnseenDistance(write_log_checkpoint_t checkpoint) noexcept;

        /**
         * @brief Scans a bounded window of write-log entries and returns pinned candidates.
         */
        WriteLogScanResult scanLogWindow(
            const vector_data_t* query_vector_data,
            size_t query_vector_dim,
            float entry_max_distance,
            const distance_function_t& distance_function,
            write_log_checkpoint_t scan_start) noexcept;

        /**
         * @brief Trims unreferenced head entries safely.
         */
        void trimUnreferencedHeadEntries() noexcept;

        /**
         * @brief Clears the entire write-log state (entries, RR, maps, risk stats).
         */
        void clear() noexcept;

        /**
         * @brief Adds one cache entry risk factor into the global risk model.
         */
        void addCacheEntryRisk(double risk_factor, epoch_t unseen_distance, size_t cache_entry_count) noexcept;

        /**
         * @brief Removes one cache entry risk factor from the global risk model.
         */
        void removeCacheEntryRisk(double risk_factor, epoch_t unseen_distance, size_t cache_entry_count) noexcept;

        /**
         * @brief Consumes unseen distance after a checkpoint advances.
         */
        void consumeUnseenDistance(epoch_t unseen_distance, size_t cache_entry_count) noexcept;

        /**
         * @brief Returns whether the slow-path should run based on current risk estimate.
         */
        bool shouldRunSlowPath() noexcept;

        /**
         * @brief Records that one refresh was applied to cached results.
         */
        void recordRefresh() noexcept;

        /**
         * @brief Returns a snapshot of write-log metrics.
         */
        WriteLogMetrics getMetrics() noexcept;

        /**
         * @brief Returns a human-readable write-log status string.
         */
        std::string getStatusText() noexcept;

        /**
         * @brief Returns a concise CSV snapshot of write-log metrics.
         */
        std::string buildMetricsCsv() noexcept;

    private:
        friend class WriteLogScanResult;

        struct RiskScore
        {
            double current_risk{0.0};
            double total_risk_factor{0.0};
            double average_risk_factor{0.0};
            double total_unseen{0.0};
            double total_cache_entries{0.0};
        };

        /**
         * @brief Updates the derived risk score fields.
         */
        void recomputeRisk() noexcept;

        /**
         * @brief Estimates the number of risky unseen log entries.
         */
        static double estimateRiskyEntries(double avg_risk_factor, double total_unseen, double total_cache_entries) noexcept;

        /**
         * @brief Pins a node reference.
         */
        void retainNode(write_log_checkpoint_t node) noexcept;

        /**
         * @brief Releases a node reference.
         */
        void releaseNode(write_log_checkpoint_t node) noexcept;

        /**
         * @brief Unpins nodes in a scan result.
         */
        void releasePinnedNodes(const std::vector<write_log_checkpoint_t>& nodes) noexcept;

        const size_t in_topk_;
        const size_t scan_thresh_;
        const double allowed_risk_;

        epoch_t latest_epoch_{0};
        size_t log_entry_count_{0};

        write_log_checkpoint_t log_head_{nullptr};
        write_log_checkpoint_t log_tail_{nullptr};

        std::unordered_map<vector_id_t, write_log_checkpoint_t> log_map_;

        std::list<cache_entry_handle_t> round_robin_list_;
        std::unordered_map<cache_entry_handle_t, std::list<cache_entry_handle_t>::iterator> round_robin_location_;

        RiskScore risk_score_;

        std::uint64_t insert_count_{0};
        std::uint64_t trim_count_{0};
        std::uint64_t slow_path_checked_count_{0};
        std::uint64_t refresh_count_{0};
        std::uint64_t duplicate_insert_count_{0};
    };

    /**
     * @brief Legacy alias kept for compatibility with older internal names.
     */
    using RfWriteLog = RiskAwareWriteLog;
}
