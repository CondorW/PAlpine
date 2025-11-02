import json
import os
import logging
from typing import List, Dict

# --- v1.0 IMPORTE (ALLE KORRIGIERT) ---
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- KONFIGURATION ---
JSONL_FILE_PATH = "judgments.jsonl"
PERSIST_DIRECTORY = "db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3" 

# --- Logging einschalten, um die generierten Queries zu sehen ---
logging.basicConfig()
logger = logging.getLogger(__name__)
# Setze das Level auf INFO, um die generierten Suchen zu sehen
logger.setLevel(logging.INFO) 

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
    # Wird nicht mehr gebraucht, aber bleibt für Vollständigkeit
    pass 

# --- v6.0 RAG-KETTE (DEIN SYNTHESIZER mit Deutsch-Zwang) ---

def get_smart_rag_chain(vectorstore, llm):
    """
    Baut die RAG-Kette mit DEINER "Query-Morphing"-Idee (v1.0).
    """
    
    # --- 1. DER QUERY-MORPHING-PROMPT (Problem 2 Fix) ---
    query_expansion_prompt_template = """
Du bist ein juristischer Assistent für deutsches/Liechtensteiner Recht.
Deine Aufgabe ist es, eine Benutzerfrage in 3 alternative, spezifische
Suchanfragen für eine Vektordatenbank umzuwandeln.

**WICHTIGE REGELN:**
1.  **SPRACHE:** Die Suchanfragen MÜSSEN zu 100% auf **DEUTSCH** sein.
2.  **KEYWORDS:** Verwende relevante juristische Keywords und Synonyme
    (z.B. für "Haftung" auch "Verantwortlichkeit", "Sorgfaltspflicht").
3.  **FORMAT:** Liefere NUR die 3 Suchanfragen, getrennt durch einen Zeilenumbruch.
4.  **VERBOTEN:** KEINE Einleitung, KEINE Nummerierung, KEINE englischen Wörter.

**BEISPIEL:**
FRAGE:
Haftung Liquidatoren
SUCHANFRAGEN (nur Deutsch):
Verantwortlichkeit der Liquidatoren bei einer AG
Sorgfaltspflichten der Liquidationsorgane
Haftungsfolgen für Liquidatoren bei Liquidation

FRAGE:
{question}

SUCHANFRAGEN (nur Deutsch):
"""
    query_expansion_prompt = PromptTemplate(
        template=query_expansion_prompt_template, input_variables=["question"]
    )

    # --- 2. DER "MULTI-QUERY"-PROZESS (Der v1.0-Weg) ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) # Holen wir 7 Chunks

    query_generation_chain = (
        query_expansion_prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n")) 
    )

    def retrieve_docs_for_queries(queries: List[str]) -> List[Dict]:
        """Nimmt eine Liste von Anfragen, führt Suchen für jede aus und gibt einzigartige Docs zurück."""
        logger.info(f"Generierte Suchanfragen (v6.0 German-Fix):\n{queries}")
        
        all_docs = []
        for query in queries:
            if query.strip(): 
                all_docs.extend(retriever.invoke(query))
        
        # De-duplizieren (basierend auf dem Inhalt)
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        return list(unique_docs)

    smart_retriever_chain = query_generation_chain | RunnableLambda(retrieve_docs_for_queries)

    # --- 3. DER FINALE ANTWORT-PROMPT (Problem 1 Fix + Glossar) ---
    final_response_template = """
Du bist ein hochpräziser juristischer Assistent für das Recht im Fürstentum Liechtenstein.
Du antwortest IMMER auf Deutsch.

**JURISTISCHES GLOSSAR:**
* **PGR:** Personen- und Gesellschaftsrecht
* **EO:** Exekutionsordnung
* **OGH:** Oberster Gerichtshof
* **VGH:** Verwaltungsgerichtshof
* **STGH:** Staatsgerichtshof

**DEINE AUFGABE:**
Beantworte die "FRAGE" des Benutzers.

**STRIKTE REGELN:**
1.  **AUSSCHLIESSLICHKEIT:** Deine Antwort darf AUSSCHLIESSLICH auf den Informationen im "KONTEXT" basieren.
2.  **KEINE HALLUZINATION:** Wenn der "KONTEXT" die Frage nicht beantwortet, antworte *exakt* und *nur* mit dem Satz:
    "Ich konnte in den vorliegenden 217 Urteilen keine Informationen zu dieser Frage finden."
3.  **SYNTHESE (DEINE IDEE):** Wenn die Antwort eine Synthese aus mehreren Quellen ist (weil die Frage nur implizit beantwortet wird), leite die Antwort ein mit:
    "Basierend auf einer Synthese der vorliegenden Urteile, die sich implizit mit dem Thema befassen, gilt folgendes:"
4.  **QUELLENPFLICHT:** JEDE Antwort (außer der "Keine Antwort"-Satz) MUSS mit den genauen Quellenangaben enden.
    (Quelle: DATEINAME.pdf) oder (Quellen: DATEI1.pdf, DATEI2.pdf)

KONTEXT:
{context}

FRAGE:
{question}

PRÄZISE ANTWORT (auf Deutsch, basierend auf Regeln 1-4):
"""
    final_response_prompt = PromptTemplate(template=final_response_template, input_variables=["context", "question"])

    def format_docs(docs):
        # Wir filtern Duplikate und fügen Quellen hinzu
        chunks = {}
        unique_sources = set()
        context_str = ""
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unbekannt')
            if source not in chunks:
                 chunks[source] = []
            chunks[source].append(doc.page_content)
            unique_sources.add(source)

        for source in unique_sources:
            context_str += f"\n--- Quelle: {source} ---\n"
            context_str += "\n...\n".join(chunks[source])
            
        return context_str

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

# --- HAUPTSKRIPT (v1.0-Importe) ---

def main():
    print("Lade Embedding-Modell (v1.0)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"FEHLER: Datenbank-Ordner '{PERSIST_DIRECTORY}' nicht gefunden.")
        return
    else:
        print(f"Lade existierende Vektor-DB aus '{PERSIST_DIRECTORY}' (v1.0)...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    print(f"Lade LLM '{OLLAMA_MODEL_NAME}' via Ollama (v1.0)...")
    try:
        llm = ChatOllama(model=OLLAMA_MODEL_NAME) 
        llm.invoke("Hallo") 
    except Exception as e:
        print("\n\n!! FEHLER: Konnte keine Verbindung zu Ollama herstellen.")
        print("Bitte stelle sicher, dass Ollama läuft (z.B. mit 'sudo systemctl start ollama' oder 'ollama serve').")
        print(f"Python-Fehler: {e}")
        return

    rag_chain = get_smart_rag_chain(vectorstore, llm)
    
    print("\n--- PAlpine RAG-System (Prototyp v6.0 - 'Der Deutsche Synthesizer') ---")
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