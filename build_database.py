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
# (Diese Werte müssen mit rag_system.py übereinstimmen)
JUDGMENTS_FILE = "judgments.jsonl"
LAWS_FILE = "laws.jsonl"
PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Besser für Deutsch!
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
    
    # Dieser Splitter ist optimiert für Code und Text.
    # Er versucht, bei Absätzen und dann bei Sätzen zu trennen.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Dokumente erfolgreich in {len(chunks)} Chunks aufgeteilt.")
    return chunks

def main():
    """
    Haupt-Ingestion-Skript.
    1. Löscht alte DB.
    2. Lädt Urteile UND Gesetze.
    3. Teilt sie in Chunks.
    4. Erstellt und speichert die neue Vektor-DB.
    """
    
    # 1. Alte DB löschen für einen sauberen Neuaufbau
    if os.path.exists(PERSIST_DIRECTORY):
        logger.warning(f"Lösche existierendes DB-Verzeichnis: {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)
        
    # 2. Alle Dokumente laden
    judgments = load_documents_from_jsonl(JUDGMENTS_FILE)
    laws = load_documents_from_jsonl(LAWS_FILE)
    all_documents = judgments + laws
    
    if not all_documents:
        logger.error("Keine Dokumente zum Verarbeiten gefunden. Breche ab.")
        return
        
    logger.info(f"Insgesamt {len(all_documents)} Dokumente (Urteile + Gesetze) geladen.")

    # 3. Dokumente in Chunks teilen (DAS IST DER KRITISCHE SCHRITT)
    all_chunks = split_documents(all_documents)
    
    # 4. Embedding-Modell laden (Neues, besseres Modell)
    logger.info(f"Lade Embedding-Modell: {EMBEDDING_MODEL_NAME}")
    # Wir nutzen 'device='cpu'' explizit, um Konsistenz zu gewährleisten.
    # Für CUDA (Nvidia GPU) können Sie 'device='cuda'' verwenden.
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 5. Vektor-DB erstellen und persistent speichern
    logger.info(f"Erstelle Vektor-DB in {PERSIST_DIRECTORY}...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    logger.info("--- Ingestion-Prozess erfolgreich abgeschlossen! ---")
    logger.info(f"Vektor-DB mit {len(all_chunks)} Chunks wurde in '{PERSIST_DIRECTORY}' gespeichert.")

if __name__ == "__main__":
    main()