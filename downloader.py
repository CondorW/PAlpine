import requests
import os
import time
from urllib.parse import unquote, urlparse

# --- KONFIGURATION ---
LINK_FILE = "pdf_links.txt"
DOWNLOAD_FOLDER = "Ressources" 
POLITENESS_DELAY_SECONDS = 0.5 

# --- Downloader Code ---

def download_pdf(url, session, download_folder):
    """Lädt ein einzelnes PDF von einer URL herunter und speichert es."""
    try:
        os.makedirs(download_folder, exist_ok=True)
        
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))
        
        file_path = os.path.join(download_folder, filename)
        
        # Das Skript ist "resumable"
        if os.path.exists(file_path):
            print(f"  Übersprungen: {filename} (existiert bereits)")
            return True 

        print(f"  Lade herunter: {filename}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = session.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status() 
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True

    except requests.exceptions.RequestException as e:
        print(f"!! FEHLER beim Download von {url}: {e}")
        return False

def main_downloader():
    """Liest die Link-Datei und startet die Downloads."""
    print("Starte PAlpine Downloader (Phase 1c: PDF Fetcher)...")

    if not os.path.exists(LINK_FILE):
        print(f"!! FEHLER: Link-Datei '{LINK_FILE}' nicht gefunden.")
        return

    with open(LINK_FILE, 'r', encoding='utf-8') as f:
        urls_to_download = [line.strip() for line in f if line.strip()]
        
    total_links = len(urls_to_download)
    print(f"{total_links} PDF-Links zum Herunterladen gefunden.")
    
    session = requests.Session()
    success_count = 0
    fail_count = 0

    for i, url in enumerate(urls_to_download):
        print(f"--- Verarbeite Link {i+1} von {total_links} ---")
        
        if download_pdf(url, session, DOWNLOAD_FOLDER):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Download abgeschlossen ---")
    print(f"Erfolgreich heruntergeladen/übersprungen: {success_count}")
    print(f"Fehlgeschlagen: {fail_count}")

if __name__ == "__main__":
    main_downloader()