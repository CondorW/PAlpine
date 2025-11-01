import requests
from bs4 import BeautifulSoup
import time
import os

# --- KONFIGURATION ---

# Die Basis-URL der Website
BASE_URL = "https://www.ogh.li"

# Die Jahre, die wir scrapen wollen (basierend auf dem Dropdown)
# Nimm für den ersten Test nur 2-3 Jahre.
# Später kannst du die Liste erweitern: list(range(2025, 1998, -1))
YEARS_TO_SCRAPE = [2024, 2023, 2022] 

# Name der Datei, in der wir die gefundenen PDF-Links speichern
OUTPUT_FILE = "pdf_links.txt"

# WICHTIG: Höflichkeitspause, um den Server nicht zu überlasten.
POLITENESS_DELAY_SECONDS = 1

# --- Web Scraper Code ---

def get_html_from_url(url, session):
    """
    Holt den HTML-Inhalt von einer URL mit robustem Error-Handling.
    """
    try:
        # Tarnen uns als normaler Browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Fehler bei 404, 500 etc.
        
        # Wichtig, um Umlaute (ä, ö, ü) korrekt zu verarbeiten
        response.encoding = 'utf-8' 
        
        return response.text
        
    except requests.exceptions.RequestException as e:
        print(f"!! Fehler beim Abrufen von {url}: {e}")
        return None

def find_pdf_links(html_content):
    """
    Parst das HTML und extrahiert alle direkten PDF-Links.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links_found = []
    
    # Basierend auf deiner Analyse:
    # Wir suchen nach <a>-Tags mit der Klasse "item__docslink"
    link_tags = soup.find_all('a', class_="item__docslink")
    
    for a_tag in link_tags:
        href = a_tag.get('href')
        if href and href.endswith('.pdf'):
            # Die Links sind bereits absolut (starten mit "https://..."),
            # wir müssen sie nicht zusammensetzen.
            links_found.append(href)
            
    return links_found

# --- Hauptskript ---
def main_scraper():
    """
    Die Haupt-Scraping-Logik: Geht durch die Jahre und sammelt PDF-Links.
    """
    print("Starte PAlpine Scraper (Phase 1b: Link Finder)...")
    
    session = requests.Session()
    
    # Ein 'set' stellt sicher, dass wir jeden Link nur einmal speichern
    all_pdf_links = set()
    
    for year in YEARS_TO_SCRAPE:
        
        # Baue die URL für das jeweilige Jahr (z.B. .../entscheidungen/2024)
        url_to_scrape = f"{BASE_URL}/entscheidungen/{year}"
        
        print(f"-> Durchsuche Jahr {year}: {url_to_scrape}")
        
        html = get_html_from_url(url_to_scrape, session)
        
        if html:
            links_on_this_page = find_pdf_links(html)
            
            if not links_on_this_page:
                print(f"  Keine PDF-Links für {year} gefunden.")
            else:
                print(f"  {len(links_on_this_page)} PDF-Links für {year} gefunden.")
                # Füge die neuen Links zum Set hinzu
                all_pdf_links.update(links_on_this_page)
        
        # Sei höflich!
        print(f"  Warte {POLITENESS_DELAY_SECONDS} Sekunde...")
        time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Scraping abgeschlossen ---")
    
    # Speichere die Ergebnisse
    if all_pdf_links:
        print(f"Insgesamt {len(all_pdf_links)} einzigartige PDF-Links gefunden.")
        
        # Sortiere die Liste für bessere Lesbarkeit
        sorted_links = sorted(list(all_pdf_links))
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for link in sorted_links:
                f.write(link + '\n')
        
        print(f"Alle Links wurden in '{OUTPUT_FILE}' gespeichert.")
    else:
        print("Keine Links gefunden.")

if __name__ == "__main__":
    main_scraper()