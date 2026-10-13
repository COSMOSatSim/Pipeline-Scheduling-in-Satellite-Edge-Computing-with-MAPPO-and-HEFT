import subprocess
import json5

def main():
    seeds = [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113] 

    for seed in seeds:
        
        # Leggi il file di configurazione JSON
        with open('config/config.json5') as config_file:
            config = json5.load(config_file)

        config["seed"] = seed

        with open("config/config.json5", "w") as f:
            json5.dump(config, f, indent=4)

        print(f"Eseguo main.py con seed {seed}")
        subprocess.run(["python3", "main.py"])

if __name__ == "__main__":
    main()