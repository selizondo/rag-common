from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rag-common")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from rag_common.models import Chunk, RetrievalResult
from rag_common import metrics, chunkers, vector_store, retrievers, base, parsers

__all__ = [
    "Chunk", "RetrievalResult",
    "metrics", "chunkers", "vector_store", "retrievers",
    "base", "parsers",
]
