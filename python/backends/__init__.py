"""
Backend implementations for QVCache.

This package contains implementations of the BackendInterface for various
vector databases and search methods.

Backends are imported lazily to avoid importing unnecessary dependencies.
Import directly from the backend module, e.g.:
    from backends.pgvector_backend import PgVectorBackend
"""

__all__ = [
    'BruteforceBackend',
    'FaissBackend',
    'PgVectorBackend',
    'PineconeBackend',
    'QdrantBackend',
]

# Lazy imports to avoid loading all dependencies at once
def __getattr__(name):
    if name == 'BruteforceBackend':
        from .bruteforce_backend import BruteforceBackend
        return BruteforceBackend
    elif name == 'FaissBackend':
        from .faiss_backend import FaissBackend
        return FaissBackend
    elif name == 'PgVectorBackend':
        from .pgvector_backend import PgVectorBackend
        return PgVectorBackend
    elif name == 'PineconeBackend':
        from .pinecone_backend import PineconeBackend
        return PineconeBackend
    elif name == 'QdrantBackend':
        from .qdrant_backend import QdrantBackend
        return QdrantBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

