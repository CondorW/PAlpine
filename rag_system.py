import json
import os
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
# KORRIGIERTE IMPORTE:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURATION ---

# 1. Quelldatei aus Phase 1
JSONL_FILE_PATH = "judgments.jsonl"

# 2. Name des Ordners für die lokale Vektor-Datenbank
PERSIST_DIRECTORY = "db"

# 3. Name des Embedding-Modells (wandelt Text in Vektoren um)
# 'all-MiniLM-L6-v2' ist klein, schnell und sehr gut für Deutsch/Englisch
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 4. Name des LLM-Modells, das du mit Ollama geladen hast
OLLAMA_MODEL_NAME = "llama3" 

# --- FUNKTIONEN ---

def load_documents_from_jsonl(file_path):
    """Lädt die Dokumente aus unserer JSONL-Datei."""
    print(f"Lade Dokumente aus {file_path}...")
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.',       # Jedes JSON-Objekt als ein Dokument
        content_key="full_text", # Der Textinhalt ist unter dem Key "full_text"
        json_lines=True,     # Wichtig, da es eine JSONL-Datei ist
        # Metadaten mitladen, damit wir wissen, woher die Antwort kommt
        metadata_func=lambda record, metadata: {
            "source": record.get("source_file"),
            "case_number": record.get("case_number")
        }
    )
    documents = loader.load()
    print(f"{len(documents)} Dokumente erfolgreich geladen.")
    return documents

def create_vector_db(documents, embedding_model, persist_directory):
    """
    Erstellt eine Vektor-Datenbank aus den Dokumenten.
    Splittet die Dokumente in kleinere "Chunks" und indexiert sie.
    """
    print("Starte Text-Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Größe der einzelnen Textabschnitte (in Zeichen)
        chunk_overlap=200   # Überlappung zwischen Chunks, um Kontext zu wahren
    )
    splits = text_splitter.split_documents(documents)
    print(f"Dokumente in {len(splits)} Abschnitte (Chunks) aufgeteilt.")
    
    print(f"Erstelle Vektor-Datenbank in '{persist_directory}'...")
    # Erstellt die ChromaDB und speichert sie im Ordner "persist_directory"
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("Vektor-Datenbank erfolgreich erstellt und gespeichert.")
    return vectorstore

def get_rag_chain(vectorstore, llm):
    """
    Baut die RAG-Kette (Retrieval-Augmented Generation) zusammen.
    """
    # Der "Retriever" ist das Suchwerkzeug für unsere Vektor-DB
    # search_kwargs={"k": 5} -> "Finde die 5 relevantesten Chunks"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Das Prompt-Template: Unsere Anweisung an das LLM
    template = """
Du bist ein hochpräziser juristischer Assistent für das Recht im Fürstentum Liechtenstein. 
Antworte auf die Frage des Benutzers präzise und ausschließlich basierend auf dem folgenden Kontext (den relevanten Urteilsauszügen).
Antworte auf Deutsch.
Wenn die Antwort nicht im Kontext enthalten ist, sage: "Ich konnte dazu keine Informationen in den vorliegenden Urteilen finden."
Zitiere deine Quellen *immer* direkt am Ende deiner Aussage, indem du die 'source' (den Dateinamen des Urteils) angibst.

KONTEXT:
{context}

FRAGE:
{question}

PRÄZISE ANTWORT:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Formatierungsfunktion, um die Quellen schön darzustellen
    def format_docs(docs):
        return "\n\n".join(
            f"--- Quelle: {doc.metadata.get('source', 'Unbekannt')} ---\n{doc.page_content}"
            for doc in docs
        )

    # Die Kette selbst:
    # 1. ({"context": retriever | format_docs, ...}): Parallel die Frage an den Retriever senden
    #    und die gefundenen Dokumente formatieren.
    # 2. | prompt: Die formatierten Docs und die Frage in das Prompt-Template einsetzen.
    # 3. | llm: Das fertige Prompt an das LLM (Ollama) senden.
    # 4. | StrOutputParser: Die Antwort des LLM als reinen Text (String) ausgeben.
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- HAUPTSKRIPT ---

def main():
    # Lade das Embedding-Modell
    print("Lade Embedding-Modell...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Prüfen, ob die Vektor-DB bereits existiert
    if not os.path.exists(PERSIST_DIRECTORY):
        print("Keine Vektor-DB gefunden. Erstelle eine neue.")
        # 1. Daten laden
        documents = load_documents_from_jsonl(JSONL_FILE_PATH)
        # 2. Vektor-DB erstellen und speichern
        vectorstore = create_vector_db(documents, embeddings, PERSIST_DIRECTORY)
    else:
        print(f"Lade existierende Vektor-DB aus '{PERSIST_DIRECTORY}'...")
        # Lade die bereits gespeicherte DB
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    # Lade das LLM (Ollama)
    print(f"Lade LLM '{OLLAMA_MODEL_NAME}' via Ollama...")
    # Stelle sicher, dass Ollama im Hintergrund läuft!
    try:
        llm = Ollama(model=OLLAMA_MODEL_NAME)
        llm.invoke("Hallo") # Testaufruf
    except Exception as e:
        print("\n\n!! FEHLER: Konnte keine Verbindung zu Ollama herstellen.")
        print("Bitte stelle sicher, dass Ollama läuft (https://ollama.com).")
        print(f"Fehlermeldung: {e}")
        return

    # Erstelle die RAG-Kette
    rag_chain = get_rag_chain(vectorstore, llm)
    
    print("\n--- PAlpine RAG-System (Prototyp) ---")
    print(f"Datenbank: {len(vectorstore.get()['ids'])} Text-Abschnitte")
    print("Stelle deine Fragen an die Urteile. (Beenden mit 'exit')")
    
    # Interaktive Frageschleife
    while True:
        try:
            query = input("\nDEINE FRAGE: ")
            if query.lower().strip() == 'exit':
                print("Wird beendet...")
                break
            
            if not query:
                continue

            print("Suche und generiere Antwort...")
            # Starte die Kette
            response = rag_chain.invoke(query)
            
            print("\nANTWORT:")
            print(response)

        except KeyboardInterrupt:
            print("\nBeendet durch Benutzer.")
            break

if __name__ == "__main__":
    main()