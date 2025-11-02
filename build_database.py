import os
import shutil
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

# --- KONFIGURATION ---
JUDGMENTS_FILE = "judgments.jsonl"
LAWS_FILE = "laws.jsonl"

# NEU: Getrennte DB-Pfade
PERSIST_DIRECTORY_LAWS = "db_laws"
PERSIST_DIRECTORY_JUDGMENTS = "db_judgments"

# (Bleibt gleich)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """Lädt Dokumente aus einer JSONL-Datei."""
    logger.info(f"Lade Dokumente aus {file_path}...")
    try:
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            content_key="full_text",
            json_lines=True,
            metadata_func=lambda record, metadata: {
                "source": record.get("source_file", "Unbekannt"),
                "case_number": record.get("case_number", "N/A")
            }
        )
        documents = loader.load()
        logger.info(f"{len(documents)} Dokumente erfolgreich aus {file_path} geladen.")
        return documents
    except Exception as e:
        logger.error(f"Fehler beim Laden von {file_path}: {e}")
        return []

def split_documents(documents: List[Document]) -> List[Document]:
    """Teilt die Dokumente in handhabbare Chunks."""
    logger.info(f"Starte Chunking für {len(documents)} Dokumente...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Dokumente erfolgreich in {len(chunks)} Chunks aufgeteilt.")
    return chunks

def create_vector_db(documents: List[Document], embeddings: HuggingFaceEmbeddings, persist_directory: str):
    """Erstellt eine Vektor-DB im angegebenen Verzeichnis."""
    
    if os.path.exists(persist_directory):
        logger.warning(f"Lösche existierendes DB-Verzeichnis: {persist_directory}")
        shutil.rmtree(persist_directory)
        
    logger.info(f"Starte Chunking für Verzeichnis: {persist_directory}...")
    chunks = split_documents(documents)
    
    if not chunks:
        logger.error(f"Keine Chunks zum Verarbeiten für {persist_directory} gefunden.")
        return

    logger.info(f"Erstelle Vektor-DB in {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logger.info(f"Vektor-DB mit {len(chunks)} Chunks wurde in '{persist_directory}' gespeichert.")

def main():
    """
    Haupt-Ingestion-Skript.
    1. Lädt Embedding-Modell.
    2. Lädt Gesetze, chunked und speichert in 'db_laws'.
    3. Lädt Urteile, chunked und speichert in 'db_judgments'.
    """
    
    # 1. Embedding-Modell laden
    logger.info(f"Lade Embedding-Modell: {EMBEDDING_MODEL_NAME}")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 2. Gesetzes-DB erstellen
    laws = load_documents_from_jsonl(LAWS_FILE)
    if laws:
        create_vector_db(laws, embeddings, PERSIST_DIRECTORY_LAWS)
    else:
        logger.warning("Keine Gesetze (laws.jsonl) gefunden. Überspringe 'db_laws'.")

    # 3. Urteils-DB erstellen
    judgments = load_documents_from_jsonl(JUDGMENTS_FILE)
    if judgments:
        create_vector_db(judgments, embeddings, PERSIST_DIRECTORY_JUDGMENTS)
    else:
        logger.warning("Keine Urteile (judgments.jsonl) gefunden. Überspringe 'db_judgments'.")

    logger.info("--- Ingestion-Prozess (Zwei-Phasen) erfolgreich abgeschlossen! ---")

if __name__ == "__main__":
    main()