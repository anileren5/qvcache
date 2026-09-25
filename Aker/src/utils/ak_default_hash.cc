#include <cstdint>
#include <cstddef>

#include "utils/ak_default_hash.hh"
#include "xxHash/xxh3.h"

namespace aker
{
    std::uint64_t
    defaultHash(const void* data, size_t size) noexcept
    {
        std::uint64_t hash = XXH3_64bits(data, size);
        return hash;
    }
}