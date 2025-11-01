import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin

# --- KONFIGURATION ---

OUTPUT_FILE = "pdf_links.txt"
POLITENESS_DELAY_SECONDS = 0.5

# QUELLE 1: OBERSTER GERICHTSHOF (OGH)
OGH_BASE_URL = "https://www.ogh.li"
# Wir nehmen nur die Jahre, die nachweislich funktionieren (Neues Format)
OGH_YEARS_TO_SCRAPE = list(range(2025, 2020, -1)) # 2025 bis 2021

# QUELLE 2: VERWALTUNGSGERICHTSHOF (VGH)
VGH_BASE_URL = "https://www.vgh.li"
VGH_SEARCH_URL = "https://www.vgh.li/entscheidungen.html"
# (Wir müssen hier die Jahre aus der VGH-Seite extrahieren)

# --- Web Scraper Code ---

def get_html_from_url(url, session):
    """Holt den HTML-Inhalt von einer URL (schnelle requests-Methode)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"!! Fehler beim Abrufen von {url}: {e}")
        return None

def find_ogh_links(html_content, all_pdf_links_set):
    """Sucht nach OGH-Links (Neues Format)."""
    soup = BeautifulSoup(html_content, 'html.parser')
    links_found_count = 0
    link_tags = soup.find_all('a', class_="item__docslink")
    
    for a_tag in link_tags:
        href = a_tag.get('href')
        if href and href.endswith('.pdf'):
            full_url = urljoin(OGH_BASE_URL, href)
            if full_url not in all_pdf_links_set:
                all_pdf_links_set.add(full_url)
                links_found_count += 1
    return links_found_count

def find_vgh_links(html_content, all_pdf_links_set):
    """
    Sucht nach VGH-Links.
    (Annahme: Links sind <a>-Tags, die auf .pdf enden und im Pfad '/files/' haben)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links_found_count = 0
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href')
        if href and href.endswith('.pdf') and ('/files/' in href):
            full_url = urljoin(VGH_BASE_URL, href)
            if full_url not in all_pdf_links_set:
                all_pdf_links_set.add(full_url)
                links_found_count += 1
    return links_found_count

# --- Hauptskript ---
def main_scraper():
    print("Starte PAlpine Scraper (Phase 1b: Link Finder - v15 'Der Pivot')")
    
    session = requests.Session()
    all_pdf_links = set()
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            all_pdf_links.update([line.strip() for line in f if line.strip()])
        print(f"{len(all_pdf_links)} bereits existierende Links aus '{OUTPUT_FILE}' geladen.")

    print("\n--- Scrape Quelle 1: OGH (Oberster Gerichtshof) ---")
    for year in OGH_YEARS_TO_SCRAPE:
        url_to_scrape = f"{OGH_BASE_URL}/entscheidungen/{year}"
        print(f"-> Durchsuche OGH Jahr {year}: {url_to_scrape}")
        
        html = get_html_from_url(url_to_scrape, session)
        if html:
            count = find_ogh_links(html, all_pdf_links)
            print(f"  {count} NEUE OGH PDF-Links gefunden.")
        time.sleep(POLITENESS_DELAY_SECONDS)

    print("\n--- Scrape Quelle 2: VGH (Verwaltungsgerichtshof) ---")
    # Der VGH hat (zum Glück) alle Links auf einer Seite.
    print(f"-> Durchsuche VGH Hauptseite: {VGH_SEARCH_URL}")
    html = get_html_from_url(VGH_SEARCH_URL, session)
    if html:
        count = find_vgh_links(html, all_pdf_links)
        print(f"  {count} NEUE VGH PDF-Links gefunden.")
    
    print("\n--- Scraping abgeschlossen ---")
    
    if all_pdf_links:
        print(f"Insgesamt {len(all_pdf_links)} einzigartige PDF-Links in der Liste.")
        
        sorted_links = sorted(list(all_pdf_links))
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for link in sorted_links:
                f.write(link + '\n')
        print(f"Alle Links wurden in '{OUTPUT_FILE}' gespeichert.")

if __name__ == "__main__":
    main_scraper()