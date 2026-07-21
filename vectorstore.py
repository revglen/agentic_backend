from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend import config

class VectorStore:
  def __init__(self):
    self._embeddings = None
    self._INDEX_DIR = Path(config.VECTOR_INDEX_DIR)

    # In-memory handles for the two indices, loaded lazily.
    self._stores: dict = {"docs": None, "web_cache": None}
    self._INDEX_NAMES = {"docs": "faiss_index", "web_cache": "web_cache_index"}

  def get_embeddings(self):    
      if self._embeddings is None:
          self._embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
      return self._embeddings

  def _split(self, docs: List[Document]) -> List[Document]:
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=config.CHUNK_SIZE,
          chunk_overlap=config.CHUNK_OVERLAP,
      )
      return splitter.split_documents(docs)

  def _index_files_exist(self, store_key: str) -> bool:
      name = self._INDEX_NAMES[store_key]
      return (self._INDEX_DIR / f"{name}.faiss").exists() and (self._INDEX_DIR / f"{name}.pkl").exists()

  def _get_store(self, store_key: str) -> Optional[FAISS]:
      """Returns the loaded FAISS store for 'docs' or 'web_cache', or None if
      nothing has been ingested into it yet (FAISS can't be instantiated empty)."""
      if self._stores[store_key] is None and self._index_files_exist(store_key):
          self._stores[store_key] = FAISS.load_local(
              str(self._INDEX_DIR),
              self.get_embeddings(),
              index_name=self._INDEX_NAMES[store_key],
              allow_dangerous_deserialization=True,  # safe: it's our own local index
          )
      return self._stores[store_key]

  def _add_and_save(self, store_key: str, chunks: List[Document]) -> None:
      existing = self._get_store(store_key)
      if existing is None:
          self._stores[store_key] = FAISS.from_documents(chunks, self.get_embeddings())
      else:
          existing.add_documents(chunks)
          self._stores[store_key] = existing

      self._INDEX_DIR.mkdir(parents=True, exist_ok=True)
      self._stores[store_key].save_local(str(self._INDEX_DIR), index_name=self._INDEX_NAMES[store_key])

  def _retriever_for(self, store_key: str, k: int, label_filter: Optional[str] = None):
      vs = self._get_store(store_key)
      if vs is None:
          return None
      search_kwargs = {"k": k}
      if label_filter:
          search_kwargs["filter"] = {"label": label_filter}  # FAISS supports simple dict equality filters
      return vs.as_retriever(search_kwargs=search_kwargs)

  # ---------------------------------------------------------------------------
  # Society documents index
  # ---------------------------------------------------------------------------

  def ingest_file(self, file_path: str, doc_label: str = "") -> int:
      """
      Load a PDF or text file, split it, tag it with a label (e.g.
      'developer_proposal_A' or 'society_bylaws'), and add it to the society
      documents vector store. Returns the number of chunks added.
      """
      path = Path(file_path)
      if path.suffix.lower() == ".pdf":
          loader = PyPDFLoader(str(path))
      else:
          loader = TextLoader(str(path), encoding="utf-8")

      raw_docs = loader.load()
      for d in raw_docs:
          d.metadata["source_file"] = path.name
          if doc_label:
              d.metadata["label"] = doc_label

      chunks = self._split(raw_docs)
      self._add_and_save("docs", chunks)
      return len(chunks)

  def get_retriever(self, k: int = None, label_filter: str = None):
      """Returns a retriever over society documents, or None if none uploaded yet."""
      return self._retriever_for("docs", k or config.RETRIEVER_K, label_filter)

  def format_docs(self, docs: List[Document]) -> str:
      parts = []
      for d in docs:
          src = d.metadata.get("source_file", "unknown")
          label = d.metadata.get("label", "")
          tag = f"{src}" + (f" [{label}]" if label else "")
          parts.append(f"--- {tag} ---\n{d.page_content}")
      return "\n\n".join(parts)

  # ---------------------------------------------------------------------------
  # Web search results cache index
  # ---------------------------------------------------------------------------

  def ingest_web_cache(self, question: str, content: str) -> int:
      """
      Stores a web search result, tagged with the question that produced it, so
      a future similar question can reuse it without hitting the web again.
      Returns the number of chunks added.
      """
      doc = Document(
          page_content=content,
          metadata={"source_file": "web_search_cache", "cached_question": question},
      )
      chunks = self._split([doc])
      for c in chunks:
          c.metadata["cached_question"] = question
      self._add_and_save("web_cache", chunks)
      return len(chunks)

  def get_web_cache_retriever(self, k: int = None):
      """Returns a retriever over cached web results, or None if the cache is empty."""
      return self._retriever_for("web_cache", k or config.CACHE_RETRIEVER_K)