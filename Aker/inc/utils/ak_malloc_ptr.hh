// 
// Author: Sukjoon Oh
//
// Unique-pointer helpers for malloc/free-managed buffers.
// This is useful for keeping legacy allocation behaviour (malloc) while
// benefiting from RAII to avoid leaks on early returns.

#ifndef AKER_MALLOC_PTR_HH
#define AKER_MALLOC_PTR_HH

#include <cstdlib>
#include <memory>

namespace aker
{
    template <typename T>
    struct FreeDeleter
    {
        void operator()(T* ptr) const noexcept
        {
            std::free(static_cast<void*>(ptr));
        }
    };

    template <typename T>
    using MallocPtr = std::unique_ptr<T, FreeDeleter<T>>;

    template <typename T>
    inline MallocPtr<T>
    makeMallocPtr(size_t element_count) noexcept
    {
        /* Allocates an uninitialized malloc buffer.
         */
        return MallocPtr<T>(static_cast<T*>(std::malloc(sizeof(T) * element_count)));
    }
}

#endif
