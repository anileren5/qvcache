// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <stdio.h>

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpuHasAvxSupport() {
  bool avxSupported = false;

  // Checking for AVX requires 3 things:
  // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
  //     instructions (allowing saving YMM registers on context
  //     switch)
  // 2) CPUID indicates support for AVX
  // 3) XGETBV indicates the AVX registers will be saved and
  //     restored on context switch
  //
  // Note that XGETBV is only available on 686 or later CPUs, so
  // the instruction needs to be conditionally run.
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);

  bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
  bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avxSupported = (xcrFeatureMask & 0x6) || false;
  }

  return avxSupported;
}

bool cpuHasAvx2Support() {
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int n = cpuInfo[0];
  if (n >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    static int avx2Mask = 0x20;
    return (cpuInfo[1] & avx2Mask) > 0;
  }
  return false;
}

#endif

#ifdef _WINDOWS
bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();
#else
bool greator::Avx2SupportedCPU = true;
bool greator::AvxSupportedCPU = false;
#endif

namespace greator {
// Get the right distance function for the given metric.
template <> greator::Distance<float> *get_distance_function(greator::Metric m) {
  if (m == greator::Metric::L2) {
    if (Avx2SupportedCPU) {
      greator::cout << "L2: Using AVX2 distance computation" << std::endl;
      return new greator::DistanceL2();
    } else if (AvxSupportedCPU) {
      greator::cout << "L2: AVX2 not supported. Using AVX distance computation"
                    << std::endl;
      return new greator::AVXDistanceL2Float();
    } else {
      greator::cout << "L2: Older CPU. Using slow distance computation"
                    << std::endl;
      return new greator::SlowDistanceL2Float();
    }
  } else if (m == greator::Metric::COSINE) {
    greator::cout << "Cosine: Using either AVX or AVX2 implementation"
                  << std::endl;
    return new greator::DistanceCosineFloat();
  } else if (m == greator::Metric::INNER_PRODUCT) {
    if (Avx2SupportedCPU) {
      greator::cout << "Inner product: Using AVX2 distance computation"
                    << std::endl;
      return new greator::AVXDistanceInnerProductFloat();
    } else {
      greator::cout << "Inner product: AVX2 not supported. Using slow distance computation"
                    << std::endl;
      return new greator::SlowDistanceInnerProductFloat();
    }
  } else {
    std::stringstream stream;
    stream << "Only L2, cosine, and inner product metric supported as of now. Email "
              "gopalsr@microsoft.com if you need support for any other metric"
           << std::endl;
    std::cerr << stream.str() << std::endl;
    throw greator::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
}

template <>
greator::Distance<int8_t> *get_distance_function(greator::Metric m) {
  if (m == greator::Metric::L2) {
    if (Avx2SupportedCPU) {
      greator::cout << "Using AVX2 distance computation" << std::endl;
      return new greator::DistanceL2Int8();
    } else if (AvxSupportedCPU) {
      greator::cout << "AVX2 not supported. Using AVX distance computation"
                    << std::endl;
      return new greator::AVXDistanceL2Int8();
    } else {
      greator::cout << "Older CPU. Using slow distance computation"
                    << std::endl;
      return new greator::SlowDistanceL2Int<int8_t>();
    }
  } else if (m == greator::Metric::COSINE) {
    greator::cout << "Using either AVX or AVX2 for Cosine similarity"
                  << std::endl;
    return new greator::DistanceCosineInt8();
  } else if (m == greator::Metric::INNER_PRODUCT) {
    std::stringstream stream;
    stream << "Inner product metric is only supported for float vectors, not int8_t. "
              "Email gopalsr@microsoft.com if you need support for int8_t inner product."
           << std::endl;
    std::cerr << stream.str() << std::endl;
    throw greator::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  } else {
    std::stringstream stream;
    stream << "Only L2, cosine, and inner product metric supported as of now. Email "
              "gopalsr@microsoft.com if you need support for any other metric"
           << std::endl;
    std::cerr << stream.str() << std::endl;
    throw greator::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
}

template <>
greator::Distance<uint8_t> *get_distance_function(greator::Metric m) {
  if (m == greator::Metric::L2) {
    greator::cout
        << "AVX/AVX2 distance function not defined for Uint8. Using "
           "slow version. "
           "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
        << std::endl;
    return new greator::DistanceL2UInt8();
  } else if (m == greator::Metric::COSINE) {
    greator::cout
        << "AVX/AVX2 distance function not defined for Uint8. Using "
           "slow version. "
           "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
        << std::endl;
    return new greator::SlowDistanceCosineUInt8();
  } else if (m == greator::Metric::INNER_PRODUCT) {
    std::stringstream stream;
    stream << "Inner product metric is only supported for float vectors, not uint8_t. "
              "Email gopalsr@microsoft.com if you need support for uint8_t inner product."
           << std::endl;
    std::cerr << stream.str() << std::endl;
    throw greator::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  } else {
    std::stringstream stream;
    stream << "Only L2, cosine, and inner product metric supported as of now. Email "
              "gopalsr@microsoft.com if you need any support for any other "
              "metric"
           << std::endl;
    std::cerr << stream.str() << std::endl;
    throw greator::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }
}

void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf,
                   _u64 npts, _u64 ndims) {
  readr.read((char *)read_buf, npts * ndims * sizeof(float));
  _u32 ndims_u32 = (_u32)ndims;
#pragma omp parallel for
  for (_s64 i = 0; i < (_s64)npts; i++) {
    float norm_pt = std::numeric_limits<float>::epsilon();
    for (_u32 dim = 0; dim < ndims_u32; dim++) {
      norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
    }
    norm_pt = std::sqrt(norm_pt);
    for (_u32 dim = 0; dim < ndims_u32; dim++) {
      *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
    }
  }
  writr.write((char *)read_buf, npts * ndims * sizeof(float));
}

void normalize_data_file(const std::string &inFileName,
                         const std::string &outFileName) {
  std::ifstream readr(inFileName, std::ios::binary);
  std::ofstream writr(outFileName, std::ios::binary);

  int npts_s32, ndims_s32;
  readr.read((char *)&npts_s32, sizeof(_s32));
  readr.read((char *)&ndims_s32, sizeof(_s32));

  writr.write((char *)&npts_s32, sizeof(_s32));
  writr.write((char *)&ndims_s32, sizeof(_s32));

  _u64 npts = (_u64)npts_s32, ndims = (_u64)ndims_s32;
  greator::cout << "Normalizing FLOAT vectors in file: " << inFileName
                << std::endl;
  greator::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims
                << std::endl;

  _u64 blk_size = 131072;
  _u64 nblks = ROUND_UP(npts, blk_size) / blk_size;
  greator::cout << "# blks: " << nblks << std::endl;

  float *read_buf = new float[npts * ndims];
  for (_u64 i = 0; i < nblks; i++) {
    _u64 cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(writr, readr, read_buf, cblk_size, ndims);
  }
  delete[] read_buf;

  greator::cout << "Wrote normalized points to file: " << outFileName
                << std::endl;
}
} // namespace diskann
