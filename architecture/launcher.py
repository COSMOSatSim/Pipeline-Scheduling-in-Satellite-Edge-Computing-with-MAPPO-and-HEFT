import os
import glob
import subprocess
import sys
import argparse
from multiprocessing import Pool, cpu_count

# Percorso della cartella contenente i file generati
CONFIG_DIR = os.path.join(os.getcwd(), "SIMS_SETS")

# Lista dei tipi di simulazione noti (basati sui nomi file generati)
SIM_TYPES = ["ALL", "ILP", "OrbitAware", "DTS-base", "DTS-APopt"]

def run_simulation(config_path):
    file_name = os.path.basename(config_path)
    print(f"--> [START] Avvio simulazione per: {file_name}")

    try:
        img_file = "config/img_resolution.json5"  # oppure usa il percorso assoluto se non è nella cwd
        subprocess.run([sys.executable, "main.py", config_path, img_file], check=True)
        print(f"<-- [END] Completata: {file_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! [ERROR] Errore nella simulazione {file_name}: {e}")
        return False


def get_user_choice():
    """
    Chiede all'utente quale tipo di simulazione eseguire tramite menu.
    """
    print("\n--- SELEZIONE SIMULAZIONI ---")
    print("Quali file di configurazione vuoi eseguire?")
    for idx, mode in enumerate(SIM_TYPES):
        print(f" {idx}) {mode}")

    while True:
        try:
            choice = input("\nInserisci il numero corrispondente: ")
            idx = int(choice)
            if 0 <= idx < len(SIM_TYPES):
                return SIM_TYPES[idx]
            else:
                print("Numero non valido, riprova.")
        except ValueError:
            print("Inserisci un numero intero.")


def main():
    # 1. Gestione argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Launcher Simulazioni Edge")
    parser.add_argument("--filter", choices=SIM_TYPES, help="Filtra il tipo di simulazione da eseguire", default=None)
    args = parser.parse_args()

    # Se l'argomento non è passato, usa il menu interattivo
    selected_mode = args.filter
    if selected_mode is None:
        selected_mode = get_user_choice()

    print(f"\nModo selezionato: {selected_mode}")

    # 2. Trova TUTTI i file .json5
    all_json_files = glob.glob(os.path.join(CONFIG_DIR, "*.json5"))

    if not all_json_files:
        print(f"Nessun file .json5 trovato in: {CONFIG_DIR}")
        return

    # 3. Filtra i file in base alla scelta
    files_to_run = []
    if selected_mode == "ALL":
        files_to_run = all_json_files
    else:
        # Controlla se il nome del modo (es. "ILP") è presente nel nome del file
        # Aggiungiamo un controllo per evitare falsi positivi (es. DTS matcherebbe DTS-base e DTS-APopt)
        for f in all_json_files:
            filename = os.path.basename(f)
            # Controllo semplice: se la stringa (es "ILP") è nel nome file
            if selected_mode in filename:
                files_to_run.append(f)

    print(f"Trovati {len(files_to_run)} file corrispondenti al filtro '{selected_mode}'.")

    if len(files_to_run) == 0:
        print("Nessun file da eseguire. Esco.")
        return

    # 4. Imposta i processi
    num_processes = max(1, cpu_count() - 2)
    print(f"Avvio esecuzione parallela su {num_processes} processi...\n")

    # 5. Avvia il pool solo sui file filtrati
    with Pool(processes=num_processes) as pool:
        pool.map(run_simulation, files_to_run)

    print("\n--- TUTTE LE SIMULAZIONI SELEZIONATE SONO COMPLETATE ---")


if __name__ == "__main__":
    main()