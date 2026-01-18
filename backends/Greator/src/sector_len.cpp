// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "pq_flash_index.h"
#include <cassert>

// Global SECTOR_LEN variable with default value 4096
uint32_t SECTOR_LEN = 4096;

// Function to set SECTOR_LEN at runtime
void set_sector_len(uint32_t sector_len) {
    // Assert that SECTOR_LEN is a multiple of 4096 for optimal performance and alignment
    // This runtime check ensures proper memory alignment and prevents segfaults
    assert(sector_len % 4096 == 0 && "SECTOR_LEN must be a multiple of 4096 for proper alignment");
    SECTOR_LEN = sector_len;
} 