#pragma once

#ifdef EXEC_ENV_OLS
namespace greator {
struct FileContent {
public:
  FileContent(void *content, size_t size) : _content(content), _size(size) {}

  void *_content;
  size_t _size;
};
} // namespace greator
#endif
