import fitz  # PyMuPDF
import os
import json
import re

# --- KONFIGURATION ---

# ACHTUNG: Auf Groß-/Kleinschreibung achten!
# Dein Ordner heißt "Ressources" (großes R)
PDF_SOURCE_FOLDER = "Ressources" 

# Die Datei, in die wir das saubere Ergebnis schreiben
OUTPUT_JSONL_FILE = "judgments.jsonl"

# --- 1. PDF-Extraktion ---

def extract_text_from_pdf(pdf_path):
    """
    Öffnet ein PDF und extrahiert den gesamten Text von allen Seiten.
    
    Args:
        pdf_path (str): Der Dateipfad zum PDF.
        
    Returns:
        str: Der extrahierte Rohtext oder None bei einem Fehler.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"!! Fehler beim Lesen von {pdf_path}: {e}")
        return None

# --- 2. Textbereinigung ---

def clean_judgment_text(text):
    """
    Führt eine grundlegende Bereinigung des extrahierten Textes durch.
    
    Args:
        text (str): Der Rohtext aus dem PDF.
        
    Returns:
        str: Der bereinigte Text.
    """
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'Seite \d+ von \d+', '', text, flags=re.IGNORECASE)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text.strip()

# --- 3. Hauptverarbeitung (ETL-Pipeline) ---

def process_pdfs_in_folder(source_folder, output_file):
    """
    Hauptfunktion: Geht durch alle PDFs im Quellordner,
    extrahiert, bereinigt und speichert sie in einer JSONL-Datei.
    """
    print(f"Starte ETL-Prozess für Ordner: '{source_folder}'...")
    processed_count = 0
    
    # Prüfen, ob der Ordner überhaupt existiert
    if not os.path.isdir(source_folder):
        print(f"!! FEHLER: Der Ordner '{source_folder}' wurde nicht gefunden.")
        print("Stelle sicher, dass der Ordner existiert und der Name (inkl. Groß/Klein) korrekt ist.")
        return

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for filename in os.listdir(source_folder):
            
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(source_folder, filename)
                print(f"-> Verarbeite: {filename}...")
                
                # 1. EXTRACT
                raw_text = extract_text_from_pdf(pdf_path)
                
                if raw_text and len(raw_text) > 5: # Nur speichern, wenn Text vorhanden ist
                    # 2. TRANSFORM
                    cleaned_text = clean_judgment_text(raw_text)
                    
                    # Erstelle ein Datenobjekt
                    judgment_data = {
                        "source_file": filename,
                        "case_number": filename.replace(".pdf", ""), 
                        "full_text": cleaned_text
                    }
                    
                    # 3. LOAD (in die JSONL-Datei)
                    json_line = json.dumps(judgment_data, ensure_ascii=False)
                    f_out.write(json_line + '\n')
                    processed_count += 1
                
                else:
                    print(f"!! Konnte keinen Text aus {filename} extrahieren. Übersprungen.")

    print("\n--- Verarbeitung abgeschlossen ---")
    if processed_count == 0:
        print("Keine PDFs im Ordner gefunden oder verarbeitet.")
    else:
        print(f"Erfolgreich verarbeitet: {processed_count} Dateien.")
        print(f"Ergebnis gespeichert in: {output_file}")


# --- Skript ausführen (DAS HAT GEFEHLT) ---

if __name__ == "__main__":
    process_pdfs_in_folder(
        source_folder=PDF_SOURCE_FOLDER, 
        output_file=OUTPUT_JSONL_FILE
    )