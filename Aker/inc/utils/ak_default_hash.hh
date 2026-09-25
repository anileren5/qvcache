// 
// Author: Sukjoon Oh

#ifndef AKER_HASH_H
#define AKER_HASH_H

#include <cstdint>
#include <cstddef>

namespace aker
{
    std::uint64_t defaultHash(const void* data, size_t size) noexcept;
}

#endif