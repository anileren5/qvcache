#pragma once

#include <unordered_map>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <cstddef>
#include <vector>

namespace qvcache {

/**
 * Thread-safe LRU (Least Recently Used) cache implementation
 * 
 * This class provides a thread-safe LRU cache that can store and manage
 * elements of type TagT. It supports:
 * - Accessing elements (moves to front if exists)
 * - Inserting new elements (adds to front)
 * - Evicting the N least recently used elements
 * - Thread-safe operations using read-write locks
 * 
 * @tparam TagT The type of tags to store in the cache
 */
template<typename TagT>
class LRUCache {
private:
    // Doubly-linked list to maintain order (most recently used at front)
    std::list<TagT> order_list;
    
    // Hash map for O(1) lookup
    std::unordered_map<TagT, typename std::list<TagT>::iterator> tag_map;
    
    // Read-write mutex for thread safety
    mutable std::shared_mutex mutex_;
    
    // Maximum capacity of the cache
    size_t max_capacity_;

public:
    /**
     * Constructor
     * @param max_capacity Maximum number of elements the cache can hold
     */
    explicit LRUCache(size_t max_capacity) : max_capacity_(max_capacity) {}
    
    /**
     * Destructor
     */
    ~LRUCache() = default;
    
    // Disable copy constructor and assignment
    LRUCache(const LRUCache&) = delete;
    LRUCache& operator=(const LRUCache&) = delete;
    
    /**
     * Access an element in the cache
     * If the element exists, it's moved to the front (most recently used)
     * If it doesn't exist, it's inserted and moved to the front
     * If the cache is full, the least recently used element is evicted
     * 
     * @param tag The tag to access
     * @return true if the tag was newly inserted, false if it already existed
     */
    bool access(const TagT& tag) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto it = tag_map.find(tag);
        if (it != tag_map.end()) {
            // Tag already exists, move to front
            order_list.splice(order_list.begin(), order_list, it->second);
            return false;
        }
        
        // Check if we need to evict before inserting
        if (order_list.size() >= max_capacity_) {
            // Evict the least recently used element (last in list)
            TagT lru_tag = order_list.back();
            order_list.pop_back();
            tag_map.erase(lru_tag);
        }
        
        // Insert new tag at front
        order_list.push_front(tag);
        tag_map[tag] = order_list.begin();
        return true;
    }
    
    /**
     * Evict the N least recently used elements
     * 
     * @param n Number of elements to evict
     * @return Vector of evicted tags
     */
    std::vector<TagT> evict(size_t n) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        std::vector<TagT> evicted_tags;
        evicted_tags.reserve(n);
        
        size_t to_evict = std::min(n, order_list.size());
        
        for (size_t i = 0; i < to_evict; ++i) {
            TagT lru_tag = order_list.back();
            evicted_tags.push_back(lru_tag);
            order_list.pop_back();
            tag_map.erase(lru_tag);
        }
        
        return evicted_tags;
    }
    
    /**
     * Check if a tag exists in the cache (read-only operation)
     * 
     * @param tag The tag to check
     * @return true if the tag exists, false otherwise
     */
    bool contains(const TagT& tag) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return tag_map.find(tag) != tag_map.end();
    }
    
    /**
     * Get the current size of the cache (read-only operation)
     * 
     * @return Number of elements currently in the cache
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return order_list.size();
    }
    
    /**
     * Check if the cache is empty (read-only operation)
     * 
     * @return true if the cache is empty, false otherwise
     */
    bool empty() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return order_list.empty();
    }
    
    /**
     * Get the maximum capacity of the cache
     * 
     * @return Maximum number of elements the cache can hold
     */
    size_t max_capacity() const {
        return max_capacity_;
    }
    
    /**
     * Clear all elements from the cache
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        order_list.clear();
        tag_map.clear();
    }
    
    /**
     * Get all tags in order from most recently used to least recently used
     * 
     * @return Vector of all tags in LRU order
     */
    std::vector<TagT> get_all_tags() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return std::vector<TagT>(order_list.begin(), order_list.end());
    }
    
    /**
     * Get the N most recently used tags
     * 
     * @param n Number of tags to retrieve
     * @return Vector of the N most recently used tags
     */
    std::vector<TagT> get_mru_tags(size_t n) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        std::vector<TagT> mru_tags;
        mru_tags.reserve(std::min(n, order_list.size()));
        
        auto it = order_list.begin();
        for (size_t i = 0; i < n && it != order_list.end(); ++i, ++it) {
            mru_tags.push_back(*it);
        }
        
        return mru_tags;
    }
    
    /**
     * Get the N least recently used tags
     * 
     * @param n Number of tags to retrieve
     * @return Vector of the N least recently used tags
     */
    std::vector<TagT> get_lru_tags(size_t n) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        std::vector<TagT> lru_tags;
        lru_tags.reserve(std::min(n, order_list.size()));
        
        auto it = order_list.rbegin();
        for (size_t i = 0; i < n && it != order_list.rend(); ++i, ++it) {
            lru_tags.push_back(*it);
        }
        
        return lru_tags;
    }
};

} // namespace qvcache 