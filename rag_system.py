import json
import os
import logging
from typing import List, Dict
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- KONFIGURATION (v9.2 - Parallel-Abruf) ---
PERSIST_DIRECTORY_LAWS = "db_laws"
PERSIST_DIRECTORY_JUDGMENTS = "db_judgments"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL_NAME = "command-r7b" 

# Anzahl der abgerufenen Dokumente
LAW_RETRIEVER_K = 10
JUDGMENT_RETRIEVER_K = 10

# --- Logging ---
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

def format_docs_with_sources(docs: List[Document]) -> str:
    """Formatiert Dokumente und fügt Quellenangaben hinzu."""
    chunks = {}
    unique_sources = set()
    context_str = ""
    
    for doc in docs:
        source = doc.metadata.get('source', 'Unbekannt')
        if source not in chunks:
             chunks[source] = []
        chunks[source].append(doc.page_content)
        unique_sources.add(source)

    for source in sorted(list(unique_sources)):
        context_str += f"\n--- Quelle: {source} ---\n"
        context_str += "\n...\n".join(chunks[source])
        
    return context_str

def get_pgr_rag_chain(law_retriever, judgment_retriever, llm):
    """
    Baut die RAG-Kette (v9.2) nach der "Anwalts-Logik":
    1. Finde Gesetze (parallel).
    2. Finde Urteile (parallel).
    3. Synthetisiere die Antwort (getrennt).
    """

    # --- 1. PROMPT: FINALE ANTWORT-SYNTHESE (Unverändert) ---
    final_response_template = """
Du bist ein hochpräziser juristischer Assistent für das Recht im Fürstentum Liechtenstein.
Du antwortest IMMER auf Deutsch und befolgst die "TERMINOLOGIE-PFLICHT".

**DEINE AUFGABE:**
Beantworte die "FRAGE" des Benutzers, indem du die bereitgestellten Kontexte nutzt.

**STRIKTE REGELN ZUR ANTWORT-STRUKTUR:**
1.  **STRUKTUR:** Du MUSST die Antwort klar strukturieren:
    * Beginne mit der Überschrift: "I. Gesetzliche Regelung"
    * Erkläre die Gesetzeslage AUSSCHLIESSLICH basierend auf dem "GESETZES-KONTEXT".
    * Füge dann die Überschrift hinzu: "II. Relevante Judikatur"
    * Ergänze die Antwort mit relevanten Urteilen AUSSCHLIESSLICH aus dem "JUDIKATUR-KONTEXT".
2.  **AUSSCHLIESSLICHKEIT:** Deine Antwort darf AUSSCHLIESSLICH auf den Informationen 
    in den "GESETZES-KONTEXT" und "JUDIKATUR-KONTEXT" basieren.
3.  **KEINE HALLUZINATION:** Wenn *beide* Kontexte die Frage nicht beantworten, antworte *exakt* und *nur* mit dem Satz:
    "Ich konnte in den vorliegenden Dokumenten (Urteile und Gesetze) keine Informationen zu dieser Frage finden."
4.  **QUELLENPFLICHT:** JEDE Information MUSS mit der genauen Quelle (z.B. (Quelle: LILEX_PGR.pdf) 
    oder (Quelle: 05CG2021179_8346_251101034656.pdf)) belegt werden.
5.  **TERMINOLOGIE-PFLICHT:** Verwende AUSSCHLIESSLICH die exakten juristischen Begriffe 
    (z.B. "Stiftungsrat" statt "Vorstand").

---
**GESETZES-KONTEXT:**
{law_context}
---
**JUDIKATUR-KONTEXT:**
{judgment_context}
---

**FRAGE:**
{question}

**PRÄZISE, STRUKTURIERTE ANTWORT (auf Deutsch, basierend auf Regeln 1-5):**
"""
    final_response_prompt = ChatPromptTemplate.from_template(final_response_template)

    # --- 2. DIE RETRIEVER-KETTEN (Parallel) ---
    
    # Kette 1: Holt Gesetze und formatiert sie
    law_chain = (
        RunnableLambda(lambda x: x['question']) 
        | law_retriever 
        | format_docs_with_sources
    )
    
    # Kette 2: Holt Urteile und formatiert sie
    judgment_chain = (
        RunnableLambda(lambda x: x['question']) 
        | judgment_retriever 
        | format_docs_with_sources
    )

    # --- 3. DIE FINALE v9.2 KETTE (PARALLEL) ---
    
    # Wir erstellen ein "Dictionary", das parallel ausgeführt wird.
    # 'law_context' wird mit Kette 1 gefüllt.
    # 'judgment_context' wird mit Kette 2 gefüllt.
    parallel_retrieval = RunnableParallel(
        {
            "law_context": law_chain,
            "judgment_context": judgment_chain,
            "question": RunnablePassthrough(lambda x: x['question'])
        }
    )

    # Die Hauptkette
    rag_chain = (
        parallel_retrieval
        | final_response_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- HAUPTSKRIPT (Unverändert) ---
# (Lädt bereits beide DBs, was perfekt für die neue Kette ist)

def main():
    print("Lade Embedding-Modell (v9.2)...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Lade Gesetzes-DB
    if not os.path.exists(PERSIST_DIRECTORY_LAWS):
        print(f"!! FEHLER: Gesetzes-DB '{PERSIST_DIRECTORY_LAWS}' nicht gefunden.")
        print("!! Bitte führe zuerst 'build_database.py' (v9.0+) aus.")
        return
    else:
        print(f"Lade Gesetzes-DB aus '{PERSIST_DIRECTORY_LAWS}'...")
        law_vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY_LAWS,
            embedding_function=embeddings
        )
        law_retriever = law_vectorstore.as_retriever(search_kwargs={"k": LAW_RETRIEVER_K})

    # Lade Urteils-DB
    if not os.path.exists(PERSIST_DIRECTORY_JUDGMENTS):
        print(f"!! FEHLER: Urteils-DB '{PERSIST_DIRECTORY_JUDGMENTS}' nicht gefunden.")
        print("!! Bitte führe zuerst 'build_database.py' (v9.0+) aus.")
        return
    else:
        print(f"Lade Urteils-DB aus '{PERSIST_DIRECTORY_JUDGMENTS}'...")
        judgment_vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY_JUDGMENTS,
            embedding_function=embeddings
        )
        judgment_retriever = judgment_vectorstore.as_retriever(search_kwargs={"k": JUDGMENT_RETRIEVER_K})

    # Lade LLM
    print(f"Lade LLM '{OLLAMA_MODEL_NAME}' via Ollama...")
    try:
        llm = ChatOllama(model=OLLAMA_MODEL_NAME) 
        llm.invoke("Hallo") 
    except Exception as e:
        print(f"\n\n!! FEHLER: Konnte keine Verbindung zu Ollama herstellen: {e}")
        return

    # Erstelle die neue RAG-Kette
    rag_chain = get_pgr_rag_chain(law_retriever, judgment_retriever, llm)
    
    print(f"\n--- PAlpine RAG-System (v9.2 - 'Parallel-Assistent') ---")
    print("Stelle deine Fragen. (Beenden mit 'exit')")
    
    while True:
        try:
            query = input("\nDEINE FRAGE: ")
            if query.lower().strip() == 'exit':
                print("Wird beendet...")
                break
            if not query:
                continue

            print("Suche (Parallel: Gesetze & Judikatur)...")
            response = rag_chain.invoke({"question": query}) 
            
            print("\nANTWORT:")
            print(response)

        except KeyboardInterrupt:
            print("\nBeendet durch Benutzer.")
            break

if __name__ == "__main__":
    main()