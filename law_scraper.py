import requests
import time
import os
import fitz  # PyMuPDF
import re
import json
from urllib.parse import urljoin

# --- KONFIGURATION (Mit deinen korrekten Links) ---

BASE_URL = "https://www.gesetze.li"

# Deine korrigierten Links, die direkt zum PDF führen
LAWS_TO_SCRAPE = {
    "PGR": "https://www.gesetze.li/konso/pdf/1926004000?version=132", # Personen- und Gesellschaftsrecht
    "ABGB": "https://www.gesetze.li/konso/pdf/1003001000?version=80", # Allgemeines bürgerliches Gesetzbuch
    "EO": "https://www.gesetze.li/konso/pdf/1972032002?version=28",
    "ZPO": "https://www.gesetze.li/konso/pdf/1912009001?version=53", # (Du hattest es ZPF genannt, aber das ist die ZPO)
}

OUTPUT_JSONL_FILE = "laws.jsonl"
POLITENESS_DELAY_SECONDS = 0.5

# --- 1. Text-Extraktion (von main.py kopiert) ---

def extract_text_from_pdf_content(pdf_content):
    """Extrahiert Text direkt aus PDF-Byte-Inhalt."""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"!! PyMuPDF-Fehler: {e}")
        return None

def clean_law_text(text):
    """Bereinigt den extrahierten Gesetzestext."""
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()

# --- 2. Vereinfachte Download-Logik ---

def download_and_process_law(law_name, pdf_url, session):
    """Lädt das PDF direkt von der URL herunter und gibt den reinen Text zurück."""
    try:
        print(f"  -> Lade PDF für {law_name} von: {pdf_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": BASE_URL
        }
        response = session.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        pdf_content = response.content 
        
        print("  -> Extrahiere Text aus PDF...")
        raw_text = extract_text_from_pdf_content(pdf_content)
        
        if raw_text:
            return clean_law_text(raw_text)
        
        return None

    except requests.exceptions.RequestException as e:
        print(f"!! FEHLER beim Download von {pdf_url}: {e}")
        return None

# --- 3. Hauptskript (Vereinfacht) ---

def main():
    print(f"Starte Gesetzes-Scraper (v2.0)... speichere in '{OUTPUT_JSONL_FILE}'")
    session = requests.Session()
    
    # Überschreibt die Datei (startet neu)
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f_out:
        for law_name, pdf_url in LAWS_TO_SCRAPE.items():
            print(f"\n--- Verarbeite: {law_name} ---")
            
            # WIR RUFEN DEN DOWNLOADER JETZT DIREKT AUF
            law_text = download_and_process_law(law_name, pdf_url, session)
            
            # Speichere es im JSONL-Format (genau wie unsere Urteile)
            if law_text:
                law_data = {
                    "source_file": f"LILEX_{law_name}.pdf", # Eigener Name
                    "case_number": law_name, # z.B. "PGR"
                    "full_text": law_text
                }
                f_out.write(json.dumps(law_data, ensure_ascii=False) + '\n')
                print(f"  ==> {law_name} erfolgreich verarbeitet und gespeichert.")
            else:
                print(f"  !! Verarbeitung für {law_name} fehlgeschlagen.")
            
            time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Gesetzes-Scraping abgeschlossen ---")
    print(f"Alle Gesetze wurden in '{OUTPUT_JSONL_FILE}' gespeichert.")

if __name__ == "__main__":
    main()