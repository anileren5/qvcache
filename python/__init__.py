"""
QVCache Python bindings

A tiered memory/disk approximate nearest neighbor search library.

The main module is 'qvcache' which is a compiled C++ extension.
"""

__version__ = "0.1.0"

# The qvcache module is imported directly - it's a compiled extension (.so/.pyd)
# When installed via pip, setuptools will handle the import path
# For development, it should be in the same directory as this file
try:
    # Try importing the compiled extension directly
    import qvcache
    # Import all public symbols
    from qvcache import *
except ImportError as e:
    raise ImportError(
        f"Could not import qvcache extension module: {e}\n"
        "Please ensure the package is properly installed:\n"
        "  cd python && pip install .\n"
        "Or if building manually:\n"
        "  cd build && cmake .. && make qvcache_python_module"
    ) from e

