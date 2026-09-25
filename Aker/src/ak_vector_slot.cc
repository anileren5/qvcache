#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <limits>

#include "ak_vector_slot.hh"

namespace aker
{
    /* Constructors / destructor.
     * The raw payload is now released automatically at destruction time.
     */
    VectorSlot::VectorSlot() noexcept
        : vector_header_{0, 0, 1},
          vector_data_(nullptr),
          aux_data_1_(0),
          aux_data_2_(0),
          vector_state_(VECTOR_STATE_VALID),
          distance_(std::numeric_limits<float>::max())
    {
    }

    VectorSlot::VectorSlot(size_t vector_in_bytes) noexcept
        : vector_header_{0, 0, 1},
          vector_data_(static_cast<vector_data_t*>(std::malloc(vector_in_bytes))),
          aux_data_1_(0),
          aux_data_2_(0),
          vector_state_(VECTOR_STATE_VALID),
          distance_(std::numeric_limits<float>::max())
    {
        assert(vector_in_bytes == 0 || vector_data_ != nullptr);
    }

    VectorSlot::~VectorSlot() noexcept
    {
        freeVectorData();
    }

    /* Locking primitives.
     *
     * Aker's current configuration uses the global ANNSCache lock.
     * Per-slot locks are intentionally removed, but the methods are
     * kept as no-ops for backward compatibility.
     */
    void
    VectorSlot::lock() noexcept
    {
    }

    bool
    VectorSlot::tryLock() noexcept
    {
        return true;
    }

    void
    VectorSlot::unlock() noexcept
    {
    }

    /* Validity helpers.
     */
    void
    VectorSlot::makeValid() noexcept
    {
        vector_state_ = VECTOR_STATE_VALID;
    }

    void
    VectorSlot::makeInvalid() noexcept
    {
        vector_state_ = VECTOR_STATE_INVALID;
    }

    bool
    VectorSlot::isValid() const noexcept
    {
        return (vector_state_ == VECTOR_STATE_VALID);
    }

    /* Header metadata accessors.
     */
    vector_id_t 
    VectorSlot::getVectorId() const noexcept
    {
        return vector_header_.vector_id;
    }

    void
    VectorSlot::setVectorId(vector_id_t vector_id) noexcept
    {
        vector_header_.vector_id = vector_id;
    }

    vector_version_t
    VectorSlot::getVectorVersion() const noexcept
    {
        return vector_header_.vector_version;
    }

    void
    VectorSlot::setVectorVersion(vector_version_t vector_version) noexcept
    {
        vector_header_.vector_version = vector_version;
    }

    void
    VectorSlot::increaseVectorVersion() noexcept
    {
        vector_header_.vector_version++;
    }

    std::uint32_t
    VectorSlot::getRefCount() const noexcept
    {
        return vector_header_.entry_reference_count;
    }

    void
    VectorSlot::resetRefCount() noexcept
    {
        vector_header_.entry_reference_count = 1;
    }

    void
    VectorSlot::increaseRefCount() noexcept
    {
        vector_header_.entry_reference_count++;
    }

    void
    VectorSlot::decreaseRefCount() noexcept
    {
        vector_header_.entry_reference_count--;
        assert(vector_header_.entry_reference_count < 1000000);
    }

    /* Payload accessors.
     */
    vector_data_t*
    VectorSlot::getVectorData() noexcept
    {
        return vector_data_;
    }

    const vector_data_t*
    VectorSlot::getVectorData() const noexcept
    {
        return vector_data_;
    }

    void
    VectorSlot::setVectorData(vector_data_t* vector_data) noexcept
    {
        vector_data_ = vector_data;
    }

    /* Aux metadata.
     */
    aux_data_t
    VectorSlot::getAuxData1() const noexcept
    {
        return aux_data_1_;
    }

    void
    VectorSlot::setAuxData1(aux_data_t aux_data) noexcept
    {
        aux_data_1_ = aux_data;
    }

    aux_data_t
    VectorSlot::getAuxData2() const noexcept
    {
        return aux_data_2_;
    }

    void
    VectorSlot::setAuxData2(aux_data_t aux_data) noexcept
    {
        aux_data_2_ = aux_data;
    }

    /* Distance helpers.
     */
    float
    VectorSlot::getDistance() const noexcept
    {
        return distance_;
    }

    void
    VectorSlot::setDistance(float distance) noexcept
    {
        distance_ = distance;
    }

    /* Buffer reclamation.
     * Idempotent to keep legacy code that calls freeVectorData() before delete safe.
     */
    void
    VectorSlot::freeVectorData() noexcept
    {
        if (vector_data_ != nullptr)
        {
            std::free(vector_data_);
            vector_data_ = nullptr;
        }
    }

    /* Debugging helper.
     */
    std::string
    VectorSlot::getStatusText() noexcept
    {
        char buffer[128];

        std::snprintf(
            buffer, sizeof(buffer),
            "VectorSlot[ID: %" PRIu64 ", Version: %u, Ref: %u, Data: %p]",
                static_cast<std::uint64_t>(vector_header_.vector_id),
                static_cast<unsigned>(vector_header_.vector_version),
                static_cast<unsigned>(vector_header_.entry_reference_count),
                static_cast<void*>(vector_data_)
        );

        return std::string(buffer);
    }
}
