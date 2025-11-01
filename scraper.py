import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin, urlparse, parse_qs
import re

# --- KONFIGURATION ---

OGH_BASE_URL = "https://www.ogh.li" 
LEGACY_BASE_URL = "https://www.gerichtsentscheidungen.li"

# Wir nehmen jetzt alle Jahre (dein Cutoff-Plan)
YEARS_TO_SCRAPE = list(range(2025, 1998, -1)) # 2025 rückwärts bis 1999

OUTPUT_FILE = "pdf_links.txt"
POLITENESS_DELAY_SECONDS = 1.0

# --- Web Scraper Code ---

def get_html_from_url(url, session):
    """Holt den HTML-Inhalt von einer URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"!! Fehler beim Abrufen von {url}: {e}")
        return None

def find_direct_pdf_links(html_content):
    """
    STRATEGIE A (NEUES FORMAT): Findet direkte PDF-Links
    <a class="item__docslink" ... href=".../files/attachments/....pdf">
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links_found = []
    
    link_tags = soup.find_all('a', class_="item__docslink")
    
    for a_tag in link_tags:
        href = a_tag.get('href')
        if href and href.endswith('.pdf'):
            full_url = urljoin(OGH_BASE_URL, href)
            links_found.append(full_url)
            
    return links_found

def find_and_construct_legacy_links(html_content):
    """
    STRATEGIE B (ALTES FORMAT): Findet Links zum alten System und
    konstruiert den PDF-Link direkt, ohne die Seite zu besuchen.
    
    Sucht nach: <a class="item" href="...&id=6868...">
                 <h2>03 CG.2014.199</h2>
    Konstruiert: https://www.gerichtsentscheidungen.li/PDF/03CG2014199_6868.pdf
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    constructed_links = []
    
    # Finde alle <a>-Tags mit der Klasse "item" (dein Fund)
    item_tags = soup.find_all('a', class_="item")
    
    for a_tag in item_tags:
        # 1. Prüfen, ob es wirklich ein "altes" Item ist (kein "item__docslink" drin)
        if a_tag.find('a', class_="item__docslink"):
            continue # Überspringen, wird von Strategie A behandelt

        href = a_tag.get('href')
        
        # 2. 'id' aus dem href extrahieren (...&id=6868...)
        if not href or "gerichtsentscheidungen.li" not in href:
            continue
            
        parsed_url = urlparse(href)
        query_params = parse_qs(parsed_url.query)
        
        if 'id' not in query_params:
            continue # Kein ID-Parameter, Link ist unbrauchbar
            
        entscheid_id = query_params['id'][0] # z.B. "6868"
        
        # 3. Aktennummer aus dem <h2>-Tag extrahieren
        h2_tag = a_tag.find('h2')
        if not h2_tag:
            continue # Keine Aktennummer, unbrauchbar
            
        # Aktennummer bereinigen: " 03 CG.2014.199 " -> "03CG2014199"
        case_number = h2_tag.text.strip()
        case_number_clean = re.sub(r'[\s\.]', '', case_number) # Entfernt alle Leerzeichen und Punkte
        
        # 4. Finalen PDF-Link konstruieren (ohne Ticks)
        final_pdf_link = f"{LEGACY_BASE_URL}/PDF/{case_number_clean}_{entscheid_id}.pdf"
        constructed_links.append(final_pdf_link)
        
    return constructed_links


# --- Hauptskript ---
def main_scraper():
    """
    Die Haupt-Scraping-Logik: Geht durch die Jahre und sammelt BEIDE Link-Typen.
    """
    print("Starte PAlpine Scraper (Phase 1b: Link Finder - v3 'Konstrukteur')")
    
    session = requests.Session()
    all_pdf_links = set()
    
    # Lade bereits gefundene Links, um Arbeit zu sparen
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            all_pdf_links.update([line.strip() for line in f if line.strip()])
        print(f"{len(all_pdf_links)} bereits existierende Links aus '{OUTPUT_FILE}' geladen.")

    # Gehe alle Jahre durch
    for year in YEARS_TO_SCRAPE:
        url_to_scrape = f"{OGH_BASE_URL}/entscheidungen/{year}"
        print(f"\n-> Durchsuche Jahr {year}: {url_to_scrape}")
        
        html = get_html_from_url(url_to_scrape, session)
        
        if html:
            # STRATEGIE A (NEU)
            direct_links = find_direct_pdf_links(html)
            if direct_links:
                print(f"  [Neues Format] {len(direct_links)} direkte PDF-Links gefunden.")
                all_pdf_links.update(direct_links)
            
            # STRATEGIE B (ALT, KONSTRUIERT)
            constructed_links = find_and_construct_legacy_links(html)
            if constructed_links:
                print(f"  [Altes Format] {len(constructed_links)} PDF-Links *konstruiert*.")
                all_pdf_links.update(constructed_links)
            
            if not direct_links and not constructed_links:
                print("  Keine Links (weder neu noch alt) für dieses Jahr gefunden.")
        
        # Höflichkeitspause
        print(f"  Warte {POLITENESS_DELAY_SECONDS} Sekunde...")
        time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Scraping abgeschlossen ---")
    
    if all_pdf_links:
        print(f"Insgesamt {len(all_pdf_links)} einzigartige PDF-Links gefunden.")
        
        sorted_links = sorted(list(all_pdf_links))
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for link in sorted_links:
                f.write(link + '\n')
        
        print(f"Alle Links wurden in '{OUTPUT_FILE}' gespeichert.")
    else:
        print("Keine Links gefunden.")

if __name__ == "__main__":
    main_scraper()