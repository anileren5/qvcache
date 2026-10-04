// 
// Author: Sukjoon Oh
//
// Refactor note:
// - Preserve the original public API while making the container safer and easier to read.
// - Provide RAII for the raw vector buffer (free on destruction) with an idempotent freeVectorData().
// - Use a standard-library based spin mutex (SpinMutex) to keep low-latency locking.
//

#ifndef AKER_VECTOR_SLOT_H
#define AKER_VECTOR_SLOT_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>


namespace aker
{
    /* Public type aliases used across the project.
     */
    typedef std::uint64_t                   vector_id_t;
    typedef std::uint32_t                   vector_version_t;
    typedef std::uint8_t                    vector_state_t;
    typedef std::uint8_t                    vector_data_t;

    typedef std::uint64_t                   aux_data_t;

    enum
    {
        VECTOR_STATE_VALID                  = 0x00,
        VECTOR_STATE_INVALID                = 0x01
    };

    typedef std::uint32_t                   vector_type_t;

    enum
    {
        VECTOR_TYPE_FREE                    = 0x00,    // Free vector
        VECTOR_TYPE_FLOAT                   = 0x01,
        VECTOR_TYPE_INT8                    = 0x02,
        VECTOR_TYPE_UINT8                   = 0x03
    };

    /* Compact header metadata for a vector slot.
     */
    typedef struct
    {
        vector_id_t                         vector_id;
        vector_version_t                    vector_version;
        std::uint32_t                       entry_reference_count;

    } vector_header_t;

    class VectorSlot
    {
    private:
        /* Core payload and metadata.
         */
        vector_header_t                     vector_header_;
        vector_data_t*                      vector_data_;
        aux_data_t                          aux_data_1_;
        aux_data_t                          aux_data_2_;

        /* Per-vector state.
         *
         * This project is configured to use the global ANNSCache lock.
         * Per-vector locks are intentionally removed.
         */
        vector_state_t                      vector_state_;

        /* Distance metadata used by the cache logic.
         */
        float                               distance_;

    public:
        VectorSlot() noexcept;
        VectorSlot(size_t vector_in_bytes) noexcept;
        virtual ~VectorSlot() noexcept;

        /* Locking interface.
         */
        void                                lock() noexcept;
        bool                                tryLock() noexcept;
        void                                unlock() noexcept;

        /* Validity flag helpers.
         */
        void                                makeValid() noexcept;
        void                                makeInvalid() noexcept;
        bool                                isValid() const noexcept;

        /* Header metadata accessors.
         */
        vector_id_t                         getVectorId() const noexcept;
        void                                setVectorId(vector_id_t vector_id) noexcept;

        vector_version_t                    getVectorVersion() const noexcept;
        void                                setVectorVersion(vector_version_t vector_version) noexcept;
        void                                increaseVectorVersion() noexcept;

        std::uint32_t                       getRefCount() const noexcept;
        void                                resetRefCount() noexcept;
        void                                increaseRefCount() noexcept;
        void                                decreaseRefCount() noexcept;

        /* Raw payload accessors.
         */
        vector_data_t*                      getVectorData() noexcept;
        const vector_data_t*                getVectorData() const noexcept;
        void                                setVectorData(vector_data_t* vector_data) noexcept;

        /* Aux metadata accessors.
         */
        aux_data_t                          getAuxData1() const noexcept;
        void                                setAuxData1(aux_data_t aux_data) noexcept;

        aux_data_t                          getAuxData2() const noexcept;
        void                                setAuxData2(aux_data_t aux_data) noexcept;

        /* Distance helpers.
         */
        float                               getDistance() const noexcept;
        void                                setDistance(float distance) noexcept;

        /* Raw buffer reclamation.
         * This function is idempotent to avoid double-free across legacy call sites.
         */
        void                                freeVectorData() noexcept;

        /**
         * @brief Returns a debug status string.
         */
        std::string                         getStatusText() noexcept;
    };
}

#endif
