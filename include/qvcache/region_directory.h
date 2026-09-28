#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace qvcache {

// Maps a coarse query-space cell to the mini-indexes that own it.
// The write shard is where the next miss or data insert in that cell lands.
struct RegionRecord {
    size_t write_shard = std::numeric_limits<size_t>::max();
    std::vector<size_t> shards;
    uint64_t last_access = 0;
};

class RegionDirectory {
public:
    explicit RegionDirectory(size_t max_overflow = 2) : max_overflow_(max_overflow) {}

    std::vector<size_t> shards_of(uint32_t key) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) {
            return {};
        }
        return it->second.shards;
    }

    size_t write_shard_of(uint32_t key) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) {
            return std::numeric_limits<size_t>::max();
        }
        return it->second.write_shard;
    }

    void bind_write(uint32_t key, size_t shard, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& rec = map_[key];
        rec.write_shard = shard;
        rec.last_access = now;
        auto found = std::find(rec.shards.begin(), rec.shards.end(), shard);
        if (found == rec.shards.end()) {
            rec.shards.insert(rec.shards.begin(), shard);
        } else if (found != rec.shards.begin()) {
            std::iter_swap(rec.shards.begin(), found);
        }
    }

    bool add_overflow(uint32_t key, size_t shard, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& rec = map_[key];
        if (rec.shards.size() >= 1 + max_overflow_) {
            return false;
        }
        if (std::find(rec.shards.begin(), rec.shards.end(), shard) == rec.shards.end()) {
            rec.shards.push_back(shard);
        }
        rec.write_shard = shard;
        rec.last_access = now;
        return true;
    }

    void touch(uint32_t key, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second.last_access = now;
        }
    }

    std::vector<uint32_t> keys_of_shard(size_t shard) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<uint32_t> keys;
        for (const auto& kv : map_) {
            const auto& rec = kv.second;
            if (std::find(rec.shards.begin(), rec.shards.end(), shard) != rec.shards.end()) {
                keys.push_back(kv.first);
            }
        }
        return keys;
    }

    void drop_shard(size_t shard) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = map_.begin(); it != map_.end();) {
            auto& rec = it->second;
            rec.shards.erase(std::remove(rec.shards.begin(), rec.shards.end(), shard), rec.shards.end());
            if (rec.write_shard == shard) {
                rec.write_shard = rec.shards.empty() ? std::numeric_limits<size_t>::max()
                                                     : rec.shards.front();
            }
            if (rec.shards.empty()) {
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t region_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint32_t, RegionRecord> map_;
    size_t max_overflow_;
};

}  // namespace qvcache
