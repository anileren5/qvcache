"""
Backend implementations for QVCache.

Import directly from the backend module, e.g.:
    from backends.pgvector_backend import PgVectorBackend
"""

__all__ = [
    'PgVectorBackend',
]

def __getattr__(name):
    if name == 'PgVectorBackend':
        from .pgvector_backend import PgVectorBackend
        return PgVectorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
