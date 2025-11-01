import requests
import os
import time
from urllib.parse import unquote, urlparse

# --- KONFIGURATION ---

# Die Datei, die unsere 2121 gesammelten Links enthält
LINK_FILE = "pdf_links.txt"

# Der Ordner, in den wir alle PDFs speichern
DOWNLOAD_FOLDER = "Ressources" 

# Höflichkeitspause zwischen Downloads
POLITENESS_DELAY_SECONDS = 0.5 # Kann schneller sein als beim Scrapen

# --- Downloader Code ---

def download_pdf(url, session, download_folder):
    """
    Lädt ein einzelnes PDF von einer URL herunter und speichert es.
    """
    try:
        # Erstelle den vollen Pfad zum Download-Ordner, falls er nicht existiert
        os.makedirs(download_folder, exist_ok=True)
        
        # Extrahiere den Dateinamen aus der URL
        # z.B. .../PDF/03CG2014199_6868.pdf -> 03CG2014199_6868.pdf
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))
        
        # Setze den vollständigen Speicherpfad zusammen
        file_path = os.path.join(download_folder, filename)
        
        # --- WICHTIG: Überspringen, wenn Datei bereits existiert ---
        # Das macht das Skript "resumable" (wiederaufnehmbar)
        if os.path.exists(file_path):
            print(f"  Übersprungen: {filename} (existiert bereits)")
            return True # Zählt als Erfolg

        print(f"  Lade herunter: {filename}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 'stream=True' ist wichtig für große Dateien (lädt nicht alles auf einmal in den RAM)
        response = session.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status() # Stellt sicher, dass der Download OK war (Status 200)
        
        # Speichere die Datei Stück für Stück auf der Festplatte
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True

    except requests.exceptions.RequestException as e:
        print(f"!! FEHLER beim Download von {url}: {e}")
        return False

def main_downloader():
    """
    Haupt-Download-Logik: Liest die Link-Datei und startet die Downloads.
    """
    print("Starte PAlpine Downloader (Phase 1c: PDF Fetcher)...")

    # Prüfen, ob die Link-Datei existiert
    if not os.path.exists(LINK_FILE):
        print(f"!! FEHLER: Link-Datei '{LINK_FILE}' nicht gefunden.")
        print("Bitte führe zuerst 'scraper.py' aus, um die Links zu sammeln.")
        return

    # Lese alle Links aus der Datei
    try:
        with open(LINK_FILE, 'r', encoding='utf-8') as f:
            urls_to_download = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"!! FEHLER beim Lesen der Datei '{LINK_FILE}': {e}")
        return
        
    total_links = len(urls_to_download)
    if total_links == 0:
        print(f"Keine Links in '{LINK_FILE}' gefunden.")
        return
        
    print(f"{total_links} PDF-Links zum Herunterladen gefunden.")
    
    session = requests.Session()
    success_count = 0
    fail_count = 0

    # --- HAUPTSCHLEIFE ---
    for i, url in enumerate(urls_to_download):
        print(f"--- Verarbeite Link {i+1} von {total_links} ---")
        
        if download_pdf(url, session, DOWNLOAD_FOLDER):
            success_count += 1
        else:
            fail_count += 1
            
        # Höflichkeitspause
        time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Download abgeschlossen ---")
    print(f"Erfolgreich heruntergeladen/übersprungen: {success_count}")
    print(f"Fehlgeschlagen: {fail_count}")
    print(f"Alle Dateien liegen in: '{DOWNLOAD_FOLDER}'")

if __name__ == "__main__":
    main_downloader()