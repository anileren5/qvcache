// Simplified SPTAG Client Backend using Python client via subprocess
// This avoids all the TBB/Boost/SPTAG C++ linking issues
// We just call the working Python client that's already tested

#include "sptag_backend.h"
#include "diskann/utils.h"

#include <cstdio>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <array>
#include <atomic>
#include <unistd.h>
#include <mutex>
#include <thread>
#include <sys/wait.h>
#include <fcntl.h>

// PIMPL implementation for simple client with persistent Python subprocess
class SPTAGClientImpl {
public:
    std::string server_addr;
    std::string server_port;
    std::string python_client_path;
    size_t vector_dim;
    FILE* python_process;
    int stdout_fd;
    pid_t child_pid;
    std::mutex process_mutex;
    bool process_initialized;
    
    SPTAGClientImpl(const std::string& addr, const std::string& port, size_t dim)
        : server_addr(addr), server_port(port), vector_dim(dim), 
          python_process(nullptr), stdout_fd(-1), child_pid(-1), process_initialized(false) {
        python_client_path = "/app/backends/SPTAG/Release/SPTAGClient.py";
        std::cerr << "SPTAGClientBackend: Initializing Python process for " << addr << ":" << port << std::endl;
        init_python_process();
        if (!process_initialized) {
            std::cerr << "SPTAGClientBackend: WARNING - Python process initialization failed!" << std::endl;
        } else {
            std::cerr << "SPTAGClientBackend: Python process initialized successfully" << std::endl;
        }
    }
    
    ~SPTAGClientImpl() {
        std::lock_guard<std::mutex> lock(process_mutex);
        if (python_process) {
            // Send quit command (ignore errors during shutdown)
            try {
                fprintf(python_process, "QUIT\n");
                fflush(python_process);
            } catch (...) {
                // Ignore errors during shutdown
            }
            fclose(python_process);
            python_process = nullptr;
        }
        if (stdout_fd >= 0) {
            close(stdout_fd);
            stdout_fd = -1;
        }
        // Wait for child process to exit (non-blocking)
        if (child_pid > 0) {
            int status;
            waitpid(child_pid, &status, WNOHANG);
            child_pid = -1;
        }
    }
    
    void init_python_process() {
        std::lock_guard<std::mutex> lock(process_mutex);
        if (process_initialized) return;
        
        // Start persistent Python process
        std::string python_script = 
            "import sys\n"
            "import json\n"
            "sys.path.insert(0, '/app/backends/SPTAG/Release')\n"
            "import SPTAGClient\n"
            "import numpy as np\n"
            "import struct\n"
            "\n"
            "# Connect to SPTAG server once\n"
            "client = None\n"
            "for _ in range(50):\n"
            "    try:\n"
            "        client = SPTAGClient.AnnClient('" + server_addr + "', '" + server_port + "')\n"
            "        import time\n"
            "        for _ in range(50):\n"
            "            if client.IsConnected():\n"
            "                break\n"
            "            time.sleep(0.1)\n"
            "        if client.IsConnected():\n"
            "            break\n"
            "    except:\n"
            "        import time\n"
            "        time.sleep(0.1)\n"
            "\n"
            "if not client or not client.IsConnected():\n"
            "    sys.exit(1)\n"
            "\n"
            "# Process queries from stdin\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if not line or line.strip() == 'QUIT':\n"
            "        break\n"
            "    \n"
            "    try:\n"
            "        # Parse query: K:query_file:result_file\n"
            "        parts = line.strip().split(':', 2)\n"
            "        if len(parts) != 3:\n"
            "            print('ERROR: Invalid format', flush=True)\n"
            "            continue\n"
            "        \n"
            "        K = int(parts[0])\n"
            "        query_file = parts[1]\n"
            "        result_file = parts[2]\n"
            "        \n"
            "        # Load query from file\n"
            "        try:\n"
            "            with open(query_file, 'rb') as f:\n"
            "                num = struct.unpack('<I', f.read(4))[0]\n"
            "                dim = struct.unpack('<I', f.read(4))[0]\n"
            "                query_vec = np.frombuffer(f.read(), dtype=np.float32).reshape(num, dim)[0]\n"
            "        except Exception as e:\n"
            "            print(f'ERROR: Failed to read query file: {str(e)}', flush=True)\n"
            "            continue\n"
            "        \n"
            "        # Search\n"
            "        try:\n"
            "            result = client.Search(query_vec, K, 'Float', False)\n"
            "        except Exception as e:\n"
            "            print(f'ERROR: Search failed: {str(e)}', flush=True)\n"
            "            continue\n"
            "        \n"
            "        # Write results to file\n"
            "        try:\n"
            "            with open(result_file, 'w') as f:\n"
            "                if result and len(result) >= 2:\n"
            "                    ids, dists = result[0], result[1]\n"
            "                    for id_val, dist_val in zip(ids, dists):\n"
            "                        f.write(f'{id_val} {dist_val}\\n')\n"
            "            print('DONE', flush=True)\n"
            "        except Exception as e:\n"
            "            print(f'ERROR: Failed to write result file: {str(e)}', flush=True)\n"
            "            continue\n"
            "    except Exception as e:\n"
            "        print(f'ERROR: {str(e)}', flush=True)\n";
        
        // Write script to temp file
        std::string script_file = "/tmp/sptag_persistent_" + std::to_string(getpid()) + ".py";
        {
            std::ofstream out(script_file);
            out << python_script;
            out.close();
        }
        
        // Start Python process with bidirectional pipes using pipe() and fork()
        int stdin_pipe[2], stdout_pipe[2];
        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            std::cerr << "SPTAGClientBackend: Failed to create pipes" << std::endl;
            remove(script_file.c_str());
            return;
        }
        
        pid_t pid = fork();
        if (pid == 0) {
            // Child process: redirect stdin/stdout and exec Python
            close(stdin_pipe[1]);  // Close write end of stdin pipe
            close(stdout_pipe[0]); // Close read end of stdout pipe
            
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stdout_pipe[1], STDERR_FILENO);
            
            close(stdin_pipe[0]);
            close(stdout_pipe[1]);
            
            execl("/usr/bin/python3", "python3", script_file.c_str(), (char*)nullptr);
            exit(1);
        } else if (pid > 0) {
            // Parent process: close unused ends and store file descriptors
            close(stdin_pipe[0]);  // Close read end of stdin pipe
            close(stdout_pipe[1]); // Close write end of stdout pipe
            
            python_process = fdopen(stdin_pipe[1], "w");
            if (!python_process) {
                std::cerr << "SPTAGClientBackend: Failed to open stdin pipe" << std::endl;
                close(stdin_pipe[1]);
                close(stdout_pipe[0]);
                remove(script_file.c_str());
                return;
            }
            
            // Store stdout pipe file descriptor for reading and child PID
            stdout_fd = stdout_pipe[0];
            child_pid = pid;
            
            // Don't remove script file yet - child process needs it
            // It will be cleaned up on exit or we can leave it in /tmp
            
            // Give Python process time to start and connect to SPTAG server
            // Wait longer (up to 6 seconds) for the connection to establish in cluster environments
            // The Python script will exit with error code 1 if it can't connect,
            // so we should check if the process is still alive multiple times
            int max_wait_attempts = 30; // 30 * 200ms = 6 seconds
            bool process_alive = true;
            for (int attempt = 0; attempt < max_wait_attempts; ++attempt) {
                usleep(200000); // 200ms delay
                int status;
                if (waitpid(child_pid, &status, WNOHANG) != 0) {
                    // Process has exited
                    process_alive = false;
                    int exit_status = WEXITSTATUS(status);
                    std::cerr << "SPTAGClientBackend: Python process exited after " 
                              << (attempt + 1) * 200 << "ms (failed to connect to " 
                              << server_addr << ":" << server_port << ")" << std::endl;
                    std::cerr << "SPTAGClientBackend: Process exit status: " << exit_status << std::endl;
                    
                    // Try to read error output from the process before it died
                    // Set stdout_fd to non-blocking to read any buffered output
                    int flags = fcntl(stdout_fd, F_GETFL, 0);
                    fcntl(stdout_fd, F_SETFL, flags | O_NONBLOCK);
                    char error_buffer[4096];
                    ssize_t error_read = read(stdout_fd, error_buffer, sizeof(error_buffer) - 1);
                    if (error_read > 0) {
                        error_buffer[error_read] = '\0';
                        std::cerr << "SPTAGClientBackend: Python process error output: " 
                                  << std::string(error_buffer) << std::endl;
                    }
                    fcntl(stdout_fd, F_SETFL, flags);
                    
                    fclose(python_process);
                    python_process = nullptr;
                    close(stdout_fd);
                    stdout_fd = -1;
                    child_pid = -1;
                    remove(script_file.c_str());
                    process_initialized = false;
                    return;
                }
            }
            
            if (!process_alive) {
                process_initialized = false;
                return;
            }
            
            // Process is still alive after waiting - assume it connected successfully
            process_initialized = true;
            std::cerr << "SPTAGClientBackend: Python process initialized and connected to " 
                      << server_addr << ":" << server_port << " after " 
                      << (max_wait_attempts * 200) << "ms" << std::endl;
        } else {
            std::cerr << "SPTAGClientBackend: Failed to fork" << std::endl;
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            remove(script_file.c_str());
            return;
        }
    }
};

namespace qvcache {

template <typename T, typename TagT>
SPTAGClientBackend<T, TagT>::SPTAGClientBackend(
    const std::string& server_addr,
    const std::string& server_port,
    size_t vector_dim,
    const std::string& data_path) {
    
    if (vector_dim == 0 && !data_path.empty()) {
        size_t file_num_vectors, file_dim;
        diskann::get_bin_metadata(data_path, file_num_vectors, file_dim);
        vector_dim = file_dim;
    } else if (vector_dim == 0) {
        throw std::runtime_error("Vector dimension must be provided if data_path is empty.");
    }
    
    // Load vector data for fetch_vectors_by_ids if data_path is provided
    if (!data_path.empty()) {
        size_t file_num_vectors, file_dim;
        diskann::get_bin_metadata(data_path, file_num_vectors, file_dim);
        if (vector_dim != file_dim) {
            throw std::runtime_error("Dimension mismatch: provided dim=" + std::to_string(vector_dim) 
                                   + " but data file has dim=" + std::to_string(file_dim));
        }
        num_vectors = file_num_vectors;
        
        vector_data = new T[num_vectors * vector_dim];
        std::ifstream reader(data_path, std::ios::binary);
        if (!reader.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_path);
        }
        reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
        reader.read(reinterpret_cast<char*>(vector_data), num_vectors * vector_dim * sizeof(T));
        reader.close();
        std::cout << "SPTAGClientBackend: Loaded " << num_vectors << " vectors of dimension " 
                  << vector_dim << " from " << data_path << std::endl;
    } else {
        vector_data = nullptr;
        num_vectors = 0;
    }
    
    dim = vector_dim;
    pimpl_ = std::unique_ptr<SPTAGClientImpl>(new SPTAGClientImpl(server_addr, server_port, vector_dim));
    std::cout << "SPTAGClientBackend (simple) initialized. Will connect to " 
              << server_addr << ":" << server_port << " via Python client" << std::endl;
}

template <typename T, typename TagT>
SPTAGClientBackend<T, TagT>::~SPTAGClientBackend() {
    if (vector_data) {
        delete[] vector_data;
    }
}

template <typename T, typename TagT>
void SPTAGClientBackend<T, TagT>::search(
    const T* query,
    uint64_t K,
    TagT* result_tags,
    float* result_distances,
    void* search_parameters,
    void* stats) {
    
    // Use persistent Python process (serialized access)
    std::lock_guard<std::mutex> lock(pimpl_->process_mutex);
    
    if (!pimpl_->python_process || pimpl_->stdout_fd < 0 || !pimpl_->process_initialized) {
        std::cerr << "SPTAGClientBackend: Python process not initialized (python_process=" 
                  << (pimpl_->python_process ? "valid" : "null") 
                  << ", stdout_fd=" << pimpl_->stdout_fd 
                  << ", process_initialized=" << pimpl_->process_initialized << ")" << std::endl;
        // Try to reinitialize if not already attempted
        if (!pimpl_->process_initialized) {
            std::cerr << "SPTAGClientBackend: Attempting to reinitialize Python process..." << std::endl;
            pimpl_->init_python_process();
            if (!pimpl_->process_initialized || !pimpl_->python_process || pimpl_->stdout_fd < 0) {
                std::cerr << "SPTAGClientBackend: Reinitialization failed, returning invalid results" << std::endl;
                for (uint64_t k = 0; k < K; ++k) {
                    result_tags[k] = std::numeric_limits<TagT>::max();
                    result_distances[k] = std::numeric_limits<float>::infinity();
                }
                return;
            }
            std::cerr << "SPTAGClientBackend: Reinitialization successful!" << std::endl;
        } else {
            for (uint64_t k = 0; k < K; ++k) {
                result_tags[k] = std::numeric_limits<TagT>::max();
                result_distances[k] = std::numeric_limits<float>::infinity();
            }
            return;
        }
    }
    
    // Create unique query and result files
    static std::atomic<uint64_t> counter{0};
    uint64_t file_id = counter.fetch_add(1, std::memory_order_relaxed);
    std::string query_file = "/tmp/sptag_query_" + std::to_string(getpid()) + "_" + std::to_string(file_id) + ".bin";
    std::string result_file = "/tmp/sptag_result_" + std::to_string(getpid()) + "_" + std::to_string(file_id) + ".txt";
    
    // Write query to file
    {
        std::ofstream out(query_file, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "SPTAGClientBackend: Failed to create query file" << std::endl;
            for (uint64_t k = 0; k < K; ++k) {
                result_tags[k] = std::numeric_limits<TagT>::max();
                result_distances[k] = std::numeric_limits<float>::infinity();
            }
            return;
        }
        uint32_t num = 1;
        uint32_t dim_u32 = static_cast<uint32_t>(dim);
        out.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&dim_u32), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(query), dim * sizeof(T));
        out.close();
    }
    
    // Send query file path and result file path to Python process: "K:query_file:result_file\n"
    fprintf(pimpl_->python_process, "%lu:%s:%s\n", K, query_file.c_str(), result_file.c_str());
    fflush(pimpl_->python_process);
    
    // Read response from stdout (skip initialization messages, wait for "DONE" or "ERROR")
    std::string response;
    char buffer[512];
    int max_lines = 100; // Maximum lines to read before giving up
    int lines_read = 0;
    
    // Read lines until we get "DONE" or "ERROR"
    while (lines_read < max_lines) {
        ssize_t bytes_read = read(pimpl_->stdout_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) {
            std::cerr << "SPTAGClientBackend: Failed to read from Python process (stdout_fd=" << pimpl_->stdout_fd << ", lines_read=" << lines_read << ")" << std::endl;
            remove(query_file.c_str());
            for (uint64_t k = 0; k < K; ++k) {
                result_tags[k] = std::numeric_limits<TagT>::max();
                result_distances[k] = std::numeric_limits<float>::infinity();
            }
            return;
        }
        buffer[bytes_read] = '\0';
        
        // Process each line
        std::string chunk(buffer);
        std::istringstream iss(chunk);
        std::string line;
        bool found_response = false;
        while (std::getline(iss, line)) {
            lines_read++;
            // Skip initialization messages (like "[1] Using AVX2 InstructionSet!")
            if (line.find("Using") != std::string::npos || 
                line.find("InstructionSet") != std::string::npos ||
                line.empty()) {
                continue;
            }
            // Check if this is our response
            if (line.find("DONE") != std::string::npos || line.find("ERROR") != std::string::npos) {
                response = line;
                found_response = true;
                break;
            }
        }
        if (found_response) {
            break;
        }
    }
    
    // Check for errors
    if (response.find("ERROR") != std::string::npos) {
        std::cerr << "SPTAGClientBackend: Python process error: " << response << std::endl;
        remove(query_file.c_str());
        remove(result_file.c_str());
        for (uint64_t k = 0; k < K; ++k) {
            result_tags[k] = std::numeric_limits<TagT>::max();
            result_distances[k] = std::numeric_limits<float>::infinity();
        }
        return;
    }
    
    if (response.find("DONE") == std::string::npos) {
        std::cerr << "SPTAGClientBackend: Did not receive DONE or ERROR response, got: " << response << std::endl;
        remove(query_file.c_str());
        remove(result_file.c_str());
        for (uint64_t k = 0; k < K; ++k) {
            result_tags[k] = std::numeric_limits<TagT>::max();
            result_distances[k] = std::numeric_limits<float>::infinity();
        }
        return;
    }
    
    // Remove query file after Python has read it
    remove(query_file.c_str());
    
    // Wait a bit for file to be written (Python prints "DONE" after writing the file, but let's be safe)
    usleep(10000); // 10ms to ensure file is flushed
    
    // Try multiple times to open the result file (in case of race condition)
    std::ifstream in;
    int retries = 20; // Increased retries
    while (retries > 0 && !in.is_open()) {
        in.open(result_file);
        if (!in.is_open()) {
            usleep(10000); // 10ms between retries
            retries--;
        }
    }
    
    if (!in.is_open()) {
        std::cerr << "SPTAGClientBackend: Failed to open result file after retries: " << result_file 
                  << " (response was: '" << response << "', file exists: " 
                  << (access(result_file.c_str(), F_OK) == 0 ? "yes" : "no") << ")" << std::endl;
        for (uint64_t k = 0; k < K; ++k) {
            result_tags[k] = std::numeric_limits<TagT>::max();
            result_distances[k] = std::numeric_limits<float>::infinity();
        }
        return;
    }
    
    {
        
        uint64_t k = 0;
        std::string line;
        while (k < K && std::getline(in, line)) {
            std::istringstream iss(line);
            uint64_t id;
            float dist;
            if (iss >> id >> dist) {
                result_tags[k] = static_cast<TagT>(id);
                result_distances[k] = dist;
                k++;
            }
        }
        in.close();
        remove(result_file.c_str());
        
        // Fill remaining slots with invalid values
        for (; k < K; ++k) {
            result_tags[k] = std::numeric_limits<TagT>::max();
            result_distances[k] = std::numeric_limits<float>::infinity();
        }
    }
}

template <typename T, typename TagT>
std::vector<std::vector<T>> SPTAGClientBackend<T, TagT>::fetch_vectors_by_ids(
    const std::vector<TagT>& ids) {
    
    std::vector<std::vector<T>> result;
    result.reserve(ids.size());
    
    if (vector_data == nullptr) {
        for (const TagT& id : ids) {
            result.emplace_back(dim, static_cast<T>(0));
        }
        return result;
    }
    
    for (const TagT& id : ids) {
        std::vector<T> vec(dim);
        if (static_cast<size_t>(id) < num_vectors) {
            std::memcpy(vec.data(), &vector_data[static_cast<size_t>(id) * dim], dim * sizeof(T));
        } else {
            std::fill(vec.begin(), vec.end(), static_cast<T>(0));
        }
        result.push_back(std::move(vec));
    }
    
    return result;
}

// Explicit template instantiations
template class SPTAGClientBackend<float, uint32_t>;
template class SPTAGClientBackend<int8_t, uint32_t>;
template class SPTAGClientBackend<uint8_t, uint32_t>;

} // namespace qvcache

