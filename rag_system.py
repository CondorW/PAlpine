import json
import os
import logging
from typing import List, Dict

# --- NEUE v1.0 IMPORTE ---
# Core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# LLM (ChatOllama statt Ollama)
from langchain_ollama import ChatOllama
# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
# Vektor-Datenbank
from langchain_chroma import Chroma
# Dokumenten-Lader & Splitter
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- KONFIGURATION ---
JSONL_FILE_PATH = "judgments.jsonl"
PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3" 

# --- Logging einschalten, um die generierten Queries zu sehen ---
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- FUNKTIONEN (Bleiben gleich) ---

def load_documents_from_jsonl(file_path):
    print(f"Lade Dokumente aus {file_path}...")
    loader = JSONLoader(
        file_path=file_path, jq_schema='.', content_key="full_text", json_lines=True,
        metadata_func=lambda record, metadata: {
            "source": record.get("source_file"),
            "case_number": record.get("case_number")
        }
    )
    documents = loader.load()
    print(f"{len(documents)} Dokumente erfolgreich geladen.")
    return documents

def create_vector_db(documents, embedding_model, persist_directory):
    # Diese Funktion wird jetzt nicht mehr gebraucht, da die DB existiert,
    # aber wir lassen sie für zukünftige Updates drin.
    print("Starte Text-Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Dokumente in {len(splits)} Abschnitte (Chunks) aufgeteilt.")
    print(f"Erstelle Vektor-Datenbank in '{PERSIST_DIRECTORY}'...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embedding_model, persist_directory=PERSIST_DIRECTORY
    )
    print("Vektor-Datenbank erfolgreich erstellt und gespeichert.")
    return vectorstore

# --- v1.0 RAG-KETTE (DEIN SYNTHESIZER) ---

def get_smart_rag_chain(vectorstore, llm):
    """
    Baut die RAG-Kette mit DEINER "Query-Morphing"-Idee,
    aber mit reiner v1.0-Architektur (LCEL).
    """
    
    # --- 1. DER QUERY-MORPHING-PROMPT (Problem 2 Fix) ---
    query_expansion_prompt_template = """
Du bist ein juristischer Assistent. Deine Aufgabe ist es, eine Benutzerfrage
in 3 alternative, spezifische Suchanfragen für eine Vektordatenbank
umzuwandeln. Die Suchen sollten relevante juristische Keywords
und Synonyme (z.B. für "Haftung" auch "Verantwortlichkeit", "Sorgfaltspflicht")
verwenden, um implizite Informationen zu finden.

Liefere NUR die 3 Suchanfragen, getrennt durch einen Zeilenumbruch.
KEINE Einleitung, KEINE Nummerierung.

FRAGE:
{question}

SUCHANFRAGEN:
"""
    query_expansion_prompt = PromptTemplate(
        template=query_expansion_prompt_template, input_variables=["question"]
    )

    # --- 2. DER "MULTI-QUERY"-PROZESS (Der v1.0-Weg) ---
    
    # Basis-Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Diese Kette generiert die 3 Suchanfragen
    query_generation_chain = (
        query_expansion_prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n")) # Wandelt den Text-Output in eine Liste von Strings um
    )

    def retrieve_docs_for_queries(queries: List[str]) -> List[Dict]:
        """Nimmt eine Liste von Anfragen, führt Suchen für jede aus und gibt einzigartige Docs zurück."""
        print(f"INFO: Generierte Suchanfragen: {queries}")
        all_docs = []
        for query in queries:
            if query.strip(): # Nur wenn die Zeile nicht leer ist
                all_docs.extend(retriever.invoke(query))
        
        # De-duplizieren (basierend auf dem Inhalt)
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        return list(unique_docs)

    # Das ist jetzt der "Retriever"-Teil unserer Kette:
    # Es nimmt die "question", generiert Queries, holt Dokumente
    smart_retriever_chain = query_generation_chain | RunnableLambda(retrieve_docs_for_queries)

    # --- 3. DER FINALE ANTWORT-PROMPT (Problem 1 Fix) ---
    final_response_template = """
Du bist ein hochpräziser juristischer Assistent für das Recht im Fürstentum Liechtenstein.
Antworte auf die Frage des Benutzers präzise und AUSSCHLIESSLICH basierend auf dem folgenden Kontext.
Antworte auf Deutsch.

**WICHTIGE REGELN:**
1.  **Quellenpflicht:** Jede Antwort MUSS mit einer Quellenangabe enden.
    (Quelle: DATEINAME.pdf) oder (Quellen: DATEI1.pdf, DATEI2.pdf)
2.  **Synthese-Hinweis (DEIN VORSCHLAG):** Wenn die Antwort eine Synthese aus
    verschiedenen Quellen ist, die das Thema nur implizit oder am Rande behandeln,
    leite die Antwort ein mit: 
    "Basierend auf einer Synthese der vorliegenden Urteile, die sich implizit mit dem Thema befassen, gilt folgendes:"
3.  **Keine Antwort:** Wenn der Kontext absolut KEINE Relevanz zur Frage hat,
    antworte mit:
    "Ich konnte in den vorliegenden 217 Urteilen keine spezifischen oder impliziten Informationen zu dieser Frage finden."

KONTEXT:
{context}

FRAGE:
{question}

PRÄZISE ANTWORT (inkl. Zitat und ggf. Synthese-Hinweis):
"""
    final_response_prompt = PromptTemplate(template=final_response_template, input_variables=["context", "question"])

    def format_docs(docs):
        chunks = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unbekannt')
            chunks.append(f"--- Quelle: {source} ---\n{doc.page_content}")
        return "\n\n".join(chunks)

    # --- 4. DIE FINALE v1.0 KETTE ---
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x['question']) | smart_retriever_chain | format_docs,
            "question": RunnableLambda(lambda x: x['question'])
        }
        | final_response_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- HAUPTSKRIPT (Jetzt mit v1.0-Importen) ---

def main():
    print("Lade Embedding-Modell (v1.0)...")
    # NEUER v1.0 IMPORT
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"FEHLER: Datenbank-Ordner '{PERSIST_DIRECTORY}' nicht gefunden.")
        return
    else:
        print(f"Lade existierende Vektor-DB aus '{PERSIST_DIRECTORY}' (v1.0)...")
        # NEUER v1.0 IMPORT
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    print(f"Lade LLM '{OLLAMA_MODEL_NAME}' via Ollama (v1.0)...")
    try:
        # --- HIER IST DER FIX ---
        # NEUER v1.0 IMPORT: Es heißt ChatOllama, nicht Ollama
        llm = ChatOllama(model=OLLAMA_MODEL_NAME) 
        llm.invoke("Hallo") 
    except Exception as e:
        print("\n\n!! FEHLER: Konnte keine Verbindung zu Ollama herstellen.")
        print("Bitte stelle sicher, dass Ollama läuft (z.B. mit 'sudo systemctl start ollama' oder 'ollama serve').")
        print(f"Python-Fehler: {e}")
        return

    rag_chain = get_smart_rag_chain(vectorstore, llm)
    
    print("\n--- PAlpine RAG-System (Prototyp v4.1 - 'Der 1.0-Fix') ---")
    print("Stelle deine Fragen an die Urteile. (Beenden mit 'exit')")
    
    while True:
        try:
            query = input("\nDEINE FRAGE: ")
            if query.lower().strip() == 'exit':
                print("Wird beendet...")
                break
            if not query:
                continue

            print("Suche (Stufe 1: v1.0 Query-Expansion)...")
            response = rag_chain.invoke({"question": query}) 
            
            print("\nANTWORT:")
            print(response)

        except KeyboardInterrupt:
            print("\nBeendet durch Benutzer.")
            break

if __name__ == "__main__":
    main()