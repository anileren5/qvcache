#include "ak_write_log.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "ak_logger.hh"

namespace aker
{
    /**
     * @brief Internal node for the write-log list.
     */
    struct WriteLogEntryNode
    {
        epoch_t epoch{0};
        vector_id_t vector_id{0};
        aux_data_t aux_data_1{0};
        aux_data_t aux_data_2{0};

        size_t vector_in_bytes{0};
        vector_data_t* vector_data{nullptr};

        std::uint32_t reference_count{0};

        WriteLogEntryNode* prev{nullptr};
        WriteLogEntryNode* next{nullptr};

        explicit WriteLogEntryNode(size_t payload_size) noexcept
            : vector_in_bytes(payload_size),
              vector_data(payload_size == 0 ? nullptr : static_cast<vector_data_t*>(std::malloc(payload_size)))
        {
            assert(payload_size == 0 || vector_data != nullptr);
        }

        ~WriteLogEntryNode() noexcept
        {
            if (vector_data != nullptr)
            {
                std::free(vector_data);
                vector_data = nullptr;
            }
        }
    };

    /* WriteLogScanResult implementation.
     */
    WriteLogScanResult::~WriteLogScanResult() noexcept
    {
        releasePinnedNodes();
    }

    WriteLogScanResult::WriteLogScanResult(WriteLogScanResult&& other) noexcept
        : new_checkpoint(other.new_checkpoint),
          advanced_epoch_distance(other.advanced_epoch_distance),
          candidates(std::move(other.candidates)),
          owner_(other.owner_),
          pinned_nodes_(std::move(other.pinned_nodes_))
    {
        other.owner_ = nullptr;
        other.new_checkpoint = nullptr;
        other.advanced_epoch_distance = 0;
    }

    WriteLogScanResult&
    WriteLogScanResult::operator=(WriteLogScanResult&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        releasePinnedNodes();

        new_checkpoint = other.new_checkpoint;
        advanced_epoch_distance = other.advanced_epoch_distance;
        candidates = std::move(other.candidates);
        owner_ = other.owner_;
        pinned_nodes_ = std::move(other.pinned_nodes_);

        other.owner_ = nullptr;
        other.new_checkpoint = nullptr;
        other.advanced_epoch_distance = 0;

        return *this;
    }

    void
    WriteLogScanResult::releasePinnedNodes() noexcept
    {
        if (owner_ == nullptr)
        {
            pinned_nodes_.clear();
            return;
        }

        owner_->releasePinnedNodes(pinned_nodes_);
        pinned_nodes_.clear();
        owner_ = nullptr;
    }

    /* RiskAwareWriteLog implementation.
     */
    RiskAwareWriteLog::RiskAwareWriteLog(size_t in_topk, size_t scan_thresh, double allowed_risk) noexcept
        : in_topk_(in_topk),
          scan_thresh_(scan_thresh),
          allowed_risk_(allowed_risk)
    {
        /* Initialize all counters and containers.
         */
        log_map_.clear();
        round_robin_list_.clear();
        round_robin_location_.clear();
    }

    RiskAwareWriteLog::~RiskAwareWriteLog() noexcept
    {
        clear();
    }

    void
    RiskAwareWriteLog::retainNode(write_log_checkpoint_t node) noexcept
    {
        if (node == nullptr)
        {
            return;
        }

        node->reference_count++;
        assert(node->reference_count < 1000000);
    }

    void
    RiskAwareWriteLog::releaseNode(write_log_checkpoint_t node) noexcept
    {
        if (node == nullptr)
        {
            return;
        }

        if (node->reference_count == 0)
        {
            return;
        }

        node->reference_count--;
    }

    double
    RiskAwareWriteLog::estimateRiskyEntries(double avg_risk_factor, double total_unseen, double total_cache_entries) noexcept
    {
        if (total_cache_entries <= 0.0)
        {
            return 0.0;
        }

        return avg_risk_factor * (total_unseen / total_cache_entries);
    }

    void
    RiskAwareWriteLog::recomputeRisk() noexcept
    {
        if (risk_score_.total_cache_entries <= 0.0)
        {
            risk_score_.average_risk_factor = 0.0;
            risk_score_.current_risk = 0.0;
            return;
        }

        risk_score_.average_risk_factor = risk_score_.total_risk_factor / risk_score_.total_cache_entries;

        const double estimated_risky = estimateRiskyEntries(
            risk_score_.average_risk_factor,
            risk_score_.total_unseen,
            risk_score_.total_cache_entries);

        if (log_entry_count_ == 0)
        {
            risk_score_.current_risk = 0.0;
            return;
        }

        risk_score_.current_risk = estimated_risky / static_cast<double>(log_entry_count_);
    }

    void
    RiskAwareWriteLog::insertLogEntry(
        vector_id_t vector_id,
        const vector_data_t* vector_data,
        size_t vector_in_bytes,
        aux_data_t aux_data_1,
        aux_data_t aux_data_2) noexcept
    {
        /* Insert a new log node (one-copy) and link it to the tail.
         */
        latest_epoch_++;

        const auto existing = log_map_.find(vector_id);
        if (existing != log_map_.end())
        {
            /* Preserve snapshot behavior: ignore duplicate vector IDs.
             */
            duplicate_insert_count_++;
            AKER_LOG_DEBUG << "[WriteLog] duplicate insert ignored: vector_id=" << vector_id;
            return;
        }

        WriteLogEntryNode* node = new WriteLogEntryNode(vector_in_bytes);
        node->epoch = latest_epoch_;
        node->vector_id = vector_id;
        node->aux_data_1 = aux_data_1;
        node->aux_data_2 = aux_data_2;
        node->reference_count = 0;

        if (vector_in_bytes > 0 && vector_data != nullptr)
        {
            std::memcpy(node->vector_data, vector_data, vector_in_bytes);
        }

        /* Link into the list.
         */
        if (log_head_ == nullptr)
        {
            log_head_ = node;
            log_tail_ = node;
        }
        else
        {
            node->prev = log_tail_;
            log_tail_->next = node;
            log_tail_ = node;
        }

        log_map_.emplace(vector_id, node);
        log_entry_count_++;
        insert_count_++;

        AKER_LOG_DEBUG << "[WriteLog] inserted log entry: epoch=" << node->epoch
                      << " vector_id=" << node->vector_id
                      << " payload_size=" << node->vector_in_bytes;
    }

    void
    RiskAwareWriteLog::addCacheEntryToRoundRobin(cache_entry_handle_t cache_entry) noexcept
    {
        /* Register cache entry handle into RR list.
         */
        if (cache_entry == nullptr)
        {
            return;
        }

        if (round_robin_location_.find(cache_entry) != round_robin_location_.end())
        {
            return;
        }

        round_robin_list_.push_back(cache_entry);
        round_robin_location_[cache_entry] = std::prev(round_robin_list_.end());

        AKER_LOG_DEBUG << "[WriteLog] added cache entry to RR: handle=" << cache_entry;
    }

    cache_entry_handle_t
    RiskAwareWriteLog::getNextCacheEntryFromRoundRobin() noexcept
    {
        /* Rotate RR list and return front element.
         */
        if (round_robin_list_.empty())
        {
            return nullptr;
        }

        cache_entry_handle_t front = round_robin_list_.front();

        if (round_robin_list_.size() > 1)
        {
            round_robin_list_.splice(round_robin_list_.end(), round_robin_list_, round_robin_list_.begin());
        }

        return front;
    }

    bool
    RiskAwareWriteLog::removeCacheEntryFromRoundRobin(cache_entry_handle_t cache_entry) noexcept
    {
        /* Remove cache entry handle from RR list.
         */
        if (cache_entry == nullptr)
        {
            return false;
        }

        const auto it = round_robin_location_.find(cache_entry);
        if (it == round_robin_location_.end())
        {
            return false;
        }

        round_robin_list_.erase(it->second);
        round_robin_location_.erase(it);
        return true;
    }

    write_log_checkpoint_t
    RiskAwareWriteLog::acquireTailCheckpoint() noexcept
    {
        /* Return the current tail with a retained reference.
         */
        if (log_tail_ == nullptr)
        {
            return nullptr;
        }

        retainNode(log_tail_);
        return log_tail_;
    }

    void
    RiskAwareWriteLog::releaseCheckpoint(write_log_checkpoint_t checkpoint) noexcept
    {
        releaseNode(checkpoint);
    }

    void
    RiskAwareWriteLog::replaceCheckpoint(write_log_checkpoint_t& checkpoint_slot, write_log_checkpoint_t new_checkpoint) noexcept
    {
        /* Replace checkpoint pointer while preserving ref-count invariants.
         */
        if (checkpoint_slot == new_checkpoint)
        {
            return;
        }

        releaseNode(checkpoint_slot);
        checkpoint_slot = new_checkpoint;
    }

    epoch_t
    RiskAwareWriteLog::getUnseenDistance(write_log_checkpoint_t checkpoint) noexcept
    {
        /* Compute distance from checkpoint epoch to latest epoch.
         */
        if (log_head_ == nullptr || log_tail_ == nullptr)
        {
            return 0;
        }

        const epoch_t latest_epoch = latest_epoch_;

        if (checkpoint == nullptr)
        {
            return (latest_epoch >= log_head_->epoch) ? (latest_epoch - log_head_->epoch) : 0;
        }

        return (latest_epoch >= checkpoint->epoch) ? (latest_epoch - checkpoint->epoch) : 0;
    }

    WriteLogScanResult
    RiskAwareWriteLog::scanLogWindow(
        const vector_data_t* query_vector_data,
        size_t query_vector_dim,
        float entry_max_distance,
        const distance_function_t& distance_function,
        write_log_checkpoint_t scan_start) noexcept
    {
        /* Scan a bounded window.
         *
         * This codebase relies on the upper ANNSCache layer holding a global lock,
         * so we do not pin every scanned node. Only the returned new checkpoint is
         * retained to keep trimming safe across calls.
         */
        WriteLogScanResult result;
        result.owner_ = this;
        result.candidates.reserve(scan_thresh_);

        write_log_checkpoint_t start_node = (scan_start != nullptr) ? scan_start : log_head_;
        if (start_node == nullptr)
        {
            result.new_checkpoint = nullptr;
            result.advanced_epoch_distance = 0;
            return result;
        }

        write_log_checkpoint_t cursor = start_node;
        size_t scanned_nodes = 0;

        for (; scanned_nodes < scan_thresh_ && cursor != nullptr; scanned_nodes++)
        {
            if (cursor->vector_data != nullptr)
            {
                const float distance = distance_function(query_vector_data, cursor->vector_data, query_vector_dim);
                if (distance < entry_max_distance)
                {
                    WriteLogCandidate candidate;
                    candidate.distance = distance;
                    candidate.vector_id = cursor->vector_id;
                    candidate.vector_data = cursor->vector_data;
                    candidate.vector_in_bytes = cursor->vector_in_bytes;
                    candidate.aux_data_1 = cursor->aux_data_1;
                    candidate.aux_data_2 = cursor->aux_data_2;
                    candidate.node = nullptr;

                    result.candidates.push_back(candidate);
                }
            }

            cursor = cursor->next;
        }

        if (cursor == nullptr)
            cursor = log_tail_;

        result.new_checkpoint = cursor;

        const epoch_t start_epoch = (scan_start != nullptr) ? scan_start->epoch : 0;
        const epoch_t end_epoch = (cursor != nullptr) ? cursor->epoch : 0;
        result.advanced_epoch_distance = (scan_start == nullptr)
                                             ? end_epoch
                                             : ((end_epoch >= start_epoch) ? (end_epoch - start_epoch) : 0);

        /* Retain only the checkpoint to preserve trimming invariants.
         * If the checkpoint does not advance, do not retain an extra reference.
         */
        if (cursor != nullptr && cursor != scan_start)
            retainNode(cursor);

        slow_path_checked_count_ += static_cast<std::uint64_t>(scanned_nodes);

        AKER_LOG_DEBUG << "[WriteLog] scan window completed: scanned=" << scanned_nodes
                      << " candidates=" << result.candidates.size()
                      << " advanced_epoch_distance=" << result.advanced_epoch_distance;

        return result;
    }

    void
    RiskAwareWriteLog::trimUnreferencedHeadEntries() noexcept
    {
        /* Trim unreferenced head entries while preserving tail.
         */
        const size_t before = log_entry_count_;

        write_log_checkpoint_t node = log_head_;
        while (node != nullptr && node != log_tail_)
        {
            if (node->reference_count != 0)
            {
                break;
            }

            write_log_checkpoint_t next_node = node->next;

            /* Remove from the id map.
             */
            log_map_.erase(node->vector_id);

            /* Unlink and delete.
             */
            delete node;
            log_entry_count_--;
            trim_count_++;

            node = next_node;
            log_head_ = node;

            if (log_head_ != nullptr)
            {
                log_head_->prev = nullptr;
            }
        }

        const size_t after = log_entry_count_;
        if (after < before)
        {
            AKER_LOG_DEBUG << "[WriteLog] trimmed head entries: before=" << before
                          << " after=" << after
                          << " trimmed=" << (before - after);
        }
    }

    void
    RiskAwareWriteLog::clear() noexcept
    {
        /* Clear entries, RR structures, and risk stats.
         */
        write_log_checkpoint_t node = log_head_;
        while (node != nullptr)
        {
            write_log_checkpoint_t next_node = node->next;
            delete node;
            node = next_node;
        }

        log_head_ = nullptr;
        log_tail_ = nullptr;
        log_entry_count_ = 0;
        latest_epoch_ = 0;

        log_map_.clear();
        round_robin_list_.clear();
        round_robin_location_.clear();

        risk_score_ = RiskScore{};

        insert_count_ = 0;
        trim_count_ = 0;
        slow_path_checked_count_ = 0;
        refresh_count_ = 0;
        duplicate_insert_count_ = 0;
    }

    void
    RiskAwareWriteLog::addCacheEntryRisk(double risk_factor, epoch_t unseen_distance, size_t cache_entry_count) noexcept
    {
        /* Update risk model for a newly inserted representative cache entry.
         */
        assert(risk_factor >= 0.0);
        assert(risk_factor <= 1.0);

        risk_score_.total_risk_factor += risk_factor;
        risk_score_.total_unseen += static_cast<double>(unseen_distance);
        risk_score_.total_cache_entries = static_cast<double>(cache_entry_count);

        recomputeRisk();
    }

    void
    RiskAwareWriteLog::removeCacheEntryRisk(double risk_factor, epoch_t unseen_distance, size_t cache_entry_count) noexcept
    {
        /* Update risk model when a representative cache entry is evicted.
         */
        assert(risk_factor >= 0.0);
        assert(risk_factor <= 1.0);

        risk_score_.total_risk_factor -= risk_factor;
        risk_score_.total_unseen -= static_cast<double>(unseen_distance);
        if (risk_score_.total_unseen < 0.0)
        {
            risk_score_.total_unseen = 0.0;
        }

        risk_score_.total_cache_entries = static_cast<double>(cache_entry_count);

        recomputeRisk();
    }

    void
    RiskAwareWriteLog::consumeUnseenDistance(epoch_t unseen_distance, size_t cache_entry_count) noexcept
    {
        /* Consume unseen distance after a checkpoint advances.
         */
        risk_score_.total_unseen -= static_cast<double>(unseen_distance);
        if (risk_score_.total_unseen < 0.0)
        {
            risk_score_.total_unseen = 0.0;
        }

        risk_score_.total_cache_entries = static_cast<double>(cache_entry_count);

        recomputeRisk();
    }

    bool
    RiskAwareWriteLog::shouldRunSlowPath() noexcept
    {
        return (allowed_risk_ < risk_score_.current_risk);
    }

    void
    RiskAwareWriteLog::recordRefresh() noexcept
    {
        refresh_count_++;
    }

    WriteLogMetrics
    RiskAwareWriteLog::getMetrics() noexcept
    {
        /* Build a stable snapshot of metrics for telemetry export.
         */
        WriteLogMetrics metrics;
        metrics.log_entry_count = log_entry_count_;
        metrics.latest_epoch = latest_epoch_;

        metrics.head_epoch = (log_head_ != nullptr) ? log_head_->epoch : 0;
        metrics.tail_epoch = (log_tail_ != nullptr) ? log_tail_->epoch : 0;

        metrics.current_risk = risk_score_.current_risk;
        metrics.total_risk_factor = risk_score_.total_risk_factor;
        metrics.average_risk_factor = risk_score_.average_risk_factor;
        metrics.total_unseen = risk_score_.total_unseen;
        metrics.total_cache_entries = risk_score_.total_cache_entries;

        metrics.insert_count = insert_count_;
        metrics.trim_count = trim_count_;
        metrics.slow_path_checked_count = slow_path_checked_count_;
        metrics.refresh_count = refresh_count_;
        metrics.duplicate_insert_count = duplicate_insert_count_;

        metrics.round_robin_count = round_robin_list_.size();

        return metrics;
    }

    std::string
    RiskAwareWriteLog::getStatusText() noexcept
    {
        /* Build a human-readable status snapshot.
         */
        std::ostringstream oss;
        oss << "WriteLog status: total elements (" << log_entry_count_ << ")\n";

        if (log_head_ != nullptr)
        {
            oss << "        Head: " << log_head_->vector_id
                << ", Head Ref: " << log_head_->reference_count << "\n";
            oss << "        Tail: " << log_tail_->vector_id
                << ", Tail Ref: " << log_tail_->reference_count << "\n";

            oss << "        Refs from tail (tail-10): ";
            write_log_checkpoint_t node = log_tail_;
            for (int i = 0; i < 10; i++)
            {
                if (node == nullptr)
                {
                    break;
                }

                oss << node->reference_count << ", ";
                node = node->prev;
            }
            oss << "...\n";

            std::uint64_t total_ref_count = 0;
            node = log_head_;
            while (node != nullptr)
            {
                total_ref_count += node->reference_count;
                node = node->next;
            }
            oss << "            Total Ref Count: " << total_ref_count << "\n";
        }

        oss << "        Risk Status:\n";
        oss << "            Current Risk: " << risk_score_.current_risk << "\n";
        oss << "            Total RiskFactor: " << risk_score_.total_risk_factor << "\n";
        oss << "            Average RiskFactor: " << risk_score_.average_risk_factor << "\n";
        oss << "            Total Unseen: " << risk_score_.total_unseen << "\n";
        oss << "            Total CacheEntry: " << risk_score_.total_cache_entries << "\n";

        oss << "            Insert count: " << insert_count_ << "\n";
        oss << "            Trim count: " << trim_count_ << "\n";
        oss << "            Slow-path checked: " << slow_path_checked_count_ << "\n";
        oss << "            Refresh count: " << refresh_count_ << "\n";
        oss << "            Duplicate insert count: " << duplicate_insert_count_ << "\n";
        oss << "            RR registered entries: " << round_robin_list_.size() << "\n";

        return oss.str();
    }

    std::string
    RiskAwareWriteLog::buildMetricsCsv() noexcept
    {
        /* Build a concise key/value CSV for external analysis.
         */
        const WriteLogMetrics metrics = getMetrics();

        std::ostringstream oss;
        oss << "metric,value\n";

        oss << "WriteLogEntryCount," << metrics.log_entry_count << "\n";
        oss << "WriteLogLatestEpoch," << metrics.latest_epoch << "\n";
        oss << "WriteLogHeadEpoch," << metrics.head_epoch << "\n";
        oss << "WriteLogTailEpoch," << metrics.tail_epoch << "\n";

        oss << std::fixed << std::setprecision(6);
        oss << "WriteLogCurrentRisk," << metrics.current_risk << "\n";
        oss << "WriteLogTotalRiskFactor," << metrics.total_risk_factor << "\n";
        oss << "WriteLogAverageRiskFactor," << metrics.average_risk_factor << "\n";
        oss << "WriteLogTotalUnseen," << metrics.total_unseen << "\n";
        oss << "WriteLogTotalCacheEntries," << metrics.total_cache_entries << "\n";

        oss << "WriteLogInsertCount," << metrics.insert_count << "\n";
        oss << "WriteLogTrimCount," << metrics.trim_count << "\n";
        oss << "WriteLogSlowPathCheckedCount," << metrics.slow_path_checked_count << "\n";
        oss << "WriteLogRefreshCount," << metrics.refresh_count << "\n";
        oss << "WriteLogDuplicateInsertCount," << metrics.duplicate_insert_count << "\n";
        oss << "WriteLogRoundRobinCount," << metrics.round_robin_count << "\n";

        return oss.str();
    }

    void
    RiskAwareWriteLog::releasePinnedNodes(const std::vector<write_log_checkpoint_t>& nodes) noexcept
    {
        /* Release pinned scan nodes.
         */
        for (write_log_checkpoint_t node : nodes)
        {
            releaseNode(node);
        }
    }
}
