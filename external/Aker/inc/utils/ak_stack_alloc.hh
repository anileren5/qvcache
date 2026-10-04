#pragma once

#include <cstddef>
#include <memory>
#include <new>

/* Converted float buffer for FAISS add/search.
 *
 * Must NOT use alloca in this constructor: gcc emits an out-of-line ctor, so
 * alloca is freed on return and add/search would index a dangling pointer.
 * The pointer is valid for the lifetime of this object.
 */

namespace aker
{
    class StackFloatBuffer
    {
    public:
        explicit StackFloatBuffer(std::size_t length) noexcept
            : length_(length),
              storage_(length == 0 ? nullptr : new (std::nothrow) float[length])
        {
        }

        StackFloatBuffer(const StackFloatBuffer&) = delete;
        StackFloatBuffer& operator=(const StackFloatBuffer&) = delete;

        float* data() noexcept { return storage_.get(); }
        const float* data() const noexcept { return storage_.get(); }
        std::size_t size() const noexcept { return length_; }

    private:
        std::size_t length_;
        std::unique_ptr<float[]> storage_;
    };
}
