import streamlit as st
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from rag_system import get_smart_rag_chain, PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME
import os

# --- Hilfsfunktionen & Caching ---

@st.cache_resource
def load_embeddings():
    """Lädt das Embedding-Modell und speichert es im Cache."""
    st.info(f"Lade Embedding-Modell: {EMBEDDING_MODEL_NAME}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

@st.cache_resource
def load_vectorstore(_embeddings):
    """Lädt die Vektordatenbank und speichert sie im Cache."""
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Datenbank-Verzeichnis '{PERSIST_DIRECTORY}' nicht gefunden. Bitte 'build_database.py' ausführen.")
        return None
    st.info(f"Lade Vektor-DB aus {PERSIST_DIRECTORY}...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=_embeddings
    )
    return vectorstore

@st.cache_resource
def load_llm():
    """Lädt das LLM (Ollama) und speichert es im Cache."""
    try:
        st.info(f"Verbinde mit Ollama-Modell: {OLLAMA_MODEL_NAME}...")
        llm = ChatOllama(model=OLLAMA_MODEL_NAME)
        llm.invoke("Hallo") # Testaufruf
        return llm
    except Exception as e:
        st.error(f"Fehler bei der Verbindung mit Ollama: {e}")
        st.error("Stellen Sie sicher, dass Ollama läuft ('ollama serve')")
        return None

@st.cache_resource
def get_cached_rag_chain() -> Runnable:
    """Baut die RAG-Kette und speichert sie im Cache."""
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm = load_llm()
    
    if not vectorstore or not llm:
        st.stop()
        
    st.success("RAG-System ist bereit.")
    return get_smart_rag_chain(vectorstore, llm)

# --- Streamlit UI ---

st.set_page_config(page_title="PAlpine RAG", layout="wide")
st.title("PAlpine RAG 🏔️ - Juristisches Assistenzsystem")
st.markdown("Prototyp v7.0 - Stellt Fragen an die Liechtensteiner Urteile und Gesetze (PGR, ABGB, EO, ZPO).")

# Initialisiere die RAG-Kette
try:
    rag_chain = get_cached_rag_chain()
except Exception as e:
    st.error(f"Ein schwerwiegender Fehler ist beim Initialisieren aufgetreten: {e}")
    st.stop()


# Chat-Initialisierung
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeige Chat-Verlauf an
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Eingabefeld für neue Fragen
if prompt := st.chat_input("Stellen Sie Ihre juristische Frage..."):
    # Zeige User-Frage
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generiere und zeige Bot-Antwort
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Suche in Urteilen und Gesetzen... (kann bis zu 30 Sek. dauern)"):
            try:
                # Hier rufen wir die Kette auf
                response = rag_chain.invoke({"question": prompt})
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Ein Fehler ist bei der Abfrage aufgetreten: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})