#pragma once

/*
 * Internal module locks were removed.
 *
 * The current Aker codebase relies on the upper ANNSCache layer holding a single global
 * cache lock that serializes all cache mutations. Per-module locks (and the previous
 * NullMutex-based configuration switch) were intentionally removed to reduce complexity.
 */
