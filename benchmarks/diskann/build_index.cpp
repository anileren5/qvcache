// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "greator/aux_utils.h"
#include "greator/utils.h"
#include "greator/pq_flash_index.h"
#include <boost/program_options.hpp>

template <typename T>
bool build_index(const char *dataFilePath, const char *indexFilePath,
                 const char *indexBuildParameters, greator::Metric m,
                 bool singleFile) {
  return greator::build_disk_index<T>(dataFilePath, indexFilePath,
                                      indexBuildParameters, m, singleFile);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    greator::cout << "Usage: " << argv[0]
                  << " <data_type (float/int8/uint8)> --data_file <file.bin>"
                     " --index_prefix_path <path> --R <num> --L <num> --B <num> --M <num> --T <num>"
                     " --dist_metric <l2/cosine/inner_product> --single_file_index <0/1> --sector_len <bytes>"
                     " See README for more information on parameters."
                  << std::endl;
    return -1;
  }

  // Parse command line arguments using boost::program_options
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  
  std::string data_file, index_prefix_path, dist_metric;
  uint32_t R, L, B, M, T;
  bool single_file_index;
  uint32_t sector_len = 4096; // Default value
  
  desc.add_options()
    ("data_file", po::value<std::string>(&data_file)->required(), "Data file")
    ("index_prefix_path", po::value<std::string>(&index_prefix_path)->required(), "Index prefix path")
    ("R", po::value<uint32_t>(&R)->required(), "R parameter")
    ("L", po::value<uint32_t>(&L)->required(), "L parameter")
    ("B", po::value<uint32_t>(&B)->required(), "B parameter")
    ("M", po::value<uint32_t>(&M)->required(), "M parameter")
    ("T", po::value<uint32_t>(&T)->required(), "T parameter")
    ("dist_metric", po::value<std::string>(&dist_metric)->required(), "Distance metric")
    ("single_file_index", po::value<bool>(&single_file_index)->required(), "Single file index (0/1)")
    ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Set the global SECTOR_LEN variable
  set_sector_len(sector_len);

  std::string params = std::to_string(R) + " " + std::to_string(L) + " " +
                       std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(T);

  greator::Metric m;
  if (dist_metric == "cosine" || dist_metric == "COSINE") {
    m = greator::Metric::COSINE;
  } else if (dist_metric == "inner_product" || dist_metric == "INNER_PRODUCT" || dist_metric == "innerproduct") {
    m = greator::Metric::INNER_PRODUCT;
  } else if (dist_metric == "l2" || dist_metric == "L2") {
    m = greator::Metric::L2;
  } else {
    greator::cout << "Metric " << dist_metric << " is not supported. Using L2"
                  << std::endl;
    m = greator::Metric::L2;
  }

  if (std::string(argv[1]) == std::string("float"))
    build_index<float>(data_file.c_str(), index_prefix_path.c_str(), params.c_str(), m,
                       single_file_index);
  else if (std::string(argv[1]) == std::string("int8"))
    build_index<int8_t>(data_file.c_str(), index_prefix_path.c_str(), params.c_str(), m,
                        single_file_index);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_index<uint8_t>(data_file.c_str(), index_prefix_path.c_str(), params.c_str(), m,
                         single_file_index);
  else
    greator::cout << "Error. wrong file type" << std::endl;

  return 0;
}
