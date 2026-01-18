#pragma once

#include <deque>
#include <mutex>
#include <functional>
#include <cstddef>

namespace qvcache {

/**
 * Thread-safe hit rate tracker that maintains statistics over the last N requests
 * and can trigger consolidation when hit rate drops below a threshold
 */
class HitRateTracker {
public:
    /**
     * Constructor
     * @param window_size Number of requests to track (default: 1000)
     * @param consolidation_threshold Hit rate threshold below which consolidation is triggered (default: 0.9 = 90%)
     * @param consolidation_callback Function to call when consolidation is needed
     */
    explicit HitRateTracker(
        size_t window_size = 1000,
        double consolidation_threshold = 0.9,
        std::function<void()> consolidation_callback = nullptr
    ) : window_size_(window_size),
        consolidation_threshold_(consolidation_threshold),
        consolidation_callback_(consolidation_callback),
        total_requests_(0),
        total_hits_(0) {
    }

    /**
     * Record a request result
     * @param is_hit Whether the request was a hit (true) or miss (false)
     * @return true if consolidation was triggered, false otherwise
     */
    bool record_request(bool is_hit) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Add the result to the window
        request_window_.push_back(is_hit);
        
        // Remove oldest result if window is full
        if (request_window_.size() > window_size_) {
            bool oldest_result = request_window_.front();
            request_window_.pop_front();
            
            // Update running totals
            if (oldest_result) {
                total_hits_--;
            }
            total_requests_--;
        }
        
        // Update running totals for new result
        if (is_hit) {
            total_hits_++;
        }
        total_requests_++;
        
        // Calculate current hit rate
        double current_hit_rate = (total_requests_ > 0) ? 
            static_cast<double>(total_hits_) / static_cast<double>(total_requests_) : 0.0;
        
        // Check if consolidation is needed
        bool should_consolidate = (total_requests_ >= window_size_) && 
                                 (current_hit_rate < consolidation_threshold_);
        
        // Trigger consolidation if needed
        if (should_consolidate && consolidation_callback_) {
            consolidation_callback_();
        }
        
        return should_consolidate;
    }

    /**
     * Get current hit rate
     * @return Hit rate as a value between 0.0 and 1.0
     */
    double get_hit_rate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (total_requests_ > 0) ? 
            static_cast<double>(total_hits_) / static_cast<double>(total_requests_) : 0.0;
    }

    /**
     * Get number of requests in current window
     * @return Number of requests tracked
     */
    size_t get_request_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_requests_;
    }

    /**
     * Get number of hits in current window
     * @return Number of hits tracked
     */
    size_t get_hit_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_hits_;
    }

    /**
     * Get window size
     * @return Maximum number of requests to track
     */
    size_t get_window_size() const {
        return window_size_;
    }

    /**
     * Get consolidation threshold
     * @return Hit rate threshold for consolidation
     */
    double get_consolidation_threshold() const {
        return consolidation_threshold_;
    }

    /**
     * Set consolidation callback
     * @param callback Function to call when consolidation is needed
     */
    void set_consolidation_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        consolidation_callback_ = callback;
    }

    /**
     * Reset statistics
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        request_window_.clear();
        total_requests_ = 0;
        total_hits_ = 0;
    }

    /**
     * Get detailed statistics
     * @return Struct containing hit rate, request count, and hit count
     */
    struct Statistics {
        double hit_rate;
        size_t request_count;
        size_t hit_count;
        size_t window_size;
        double consolidation_threshold;
    };

    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            (total_requests_ > 0) ? static_cast<double>(total_hits_) / static_cast<double>(total_requests_) : 0.0,
            total_requests_,
            total_hits_,
            window_size_,
            consolidation_threshold_
        };
    }

private:
    const size_t window_size_;
    const double consolidation_threshold_;
    std::function<void()> consolidation_callback_;
    
    mutable std::mutex mutex_;
    std::deque<bool> request_window_;  // Circular buffer of request results
    size_t total_requests_;            // Total requests in current window
    size_t total_hits_;                // Total hits in current window
};

} // namespace qvcache
