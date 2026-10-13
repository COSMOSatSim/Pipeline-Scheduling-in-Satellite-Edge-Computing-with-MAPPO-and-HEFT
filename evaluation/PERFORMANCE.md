# Misure delle ottimizzazioni EFT e training

EFT resta applicato a ogni passo e ricalcola lo scheduling globale. Sono stati
ridotti i calcoli ripetuti senza modificare reward, iperparametri o ordine di
priorità HEFT.

Le modifiche sono:

- Calcolo di `C_T` e `C_E` una sola volta per passo, condiviso fra reward e `infos`.
- Cache delle durate di trasmissione durante ciascuna chiamata EFT e ciascuna
  valutazione energetica. Le finestre offline vengono sempre ricalcolate sul
  satellite effettivo e sul nuovo istante di completamento del predecessore.
- Fusione degli intervalli occupati esattamente adiacenti nella ricerca degli
  slot: si evita di scandire separatamente tutti i task di un blocco continuo.
  I gap restano disponibili e i singoli task restano distinti nelle code.
  Se sono presenti task con durata nulla si usa la lista originale degli slot.

Misure locali del 6 settembre 2026: Windows, Python 3.10.6, CPU, seed 2026.
Il benchmark misura reset e tutti gli step di un singolo ambiente; esclude la
costruzione dello scenario, le reti e il calcolo dei digest. Mediana di due
ripetizioni dopo un episodio di verifica. Le assegnazioni SAS/GND rispettano
comunque i task obbligatoriamente locali.

| Scenario | Assegnazioni | Prima | Dopo | Accelerazione |
|---|---|---:|---:|---:|
| 4096, scenario 0, 3 agenti | Casuali | 0,701 s | 0,393 s | 1,78x |
| 4096, scenario 0, 3 agenti | SAS | 0,949 s | 0,331 s | 2,87x |
| 4096, scenario 0, 3 agenti | GND | 0,722 s | 0,328 s | 2,20x |
| 8192, scenario 0, 3 agenti | Casuali | 2,771 s | 1,425 s | 1,94x |
| 8192, scenario 0, 3 agenti | SAS | 5,592 s | 1,241 s | 4,51x |
| 8192, scenario 0, 3 agenti | GND | 3,864 s | 1,290 s | 3,00x |

I digest coincidono a ogni passo dei sei episodi (1.482 transizioni), includendo
ordine delle code, tempi di esecuzione e trasmissione, osservazioni locali e
globali, maschere, costi, reward e terminazioni. Verificato anche il reset.
Altri controlli: 10.000 inserimenti casuali con identico earliest start e tre
prefissi di uno scenario a cinque agenti con durate nulle.

Training completo breve: due iterazioni, otto ambienti sequenziali, scenario
8192/0, seed e parametri canonici del progetto, CPU. Tempo medio per iterazione
da **26,48 s a 15,75 s** (1,68x; circa 40,5% in meno). La fase ambiente passa
da 22,85 s a 12,14 s; PPO rimane circa 2,4 s. I tensori finali di actor, critic,
optimizer e normalizzatore sono esattamente identici. Le run di verifica sono
separate dai modelli esistenti, in `data/performance/train_reference` e
`data/performance/train_optimized`.

Queste misure non stimano la durata sul cluster: Linux usa normalmente otto
processi per gli ambienti e il training usa la GPU. Non sono stati eseguiti job
sul cluster né modificati i suoi file. Gli script SLURM non sono presenti nel
progetto esaminato, quindi l'allocazione CPU effettiva non è stata verificata.

Per misurare il training sul cluster, dopo aver trasferito i file modificati,
aggiungere prima del consueto comando di avvio:

```bash
export SECORS_PROFILE_TIMING=1
```

Nella cartella della run compare `training_timing.csv`: tempi per iterazione
di raccolta delle azioni, ambiente (inclusi comunicazione e attesa dei processi),
buffer, GAE, PPO, checkpoint e grafici, più backend e device effettivi.
Il profiling sincronizza CUDA ai confini delle fasi e introduce un piccolo
overhead; disattivarlo togliendo la variabile dopo la misurazione. Il totale
esclude l'inizializzazione della run e il grafico finale dopo il ciclo.
Funziona anche alla ripresa, senza modificare `run_info.json`.

Per ripetere il benchmark dell'ambiente dalla root del progetto:

```bash
python -m evaluation.benchmark_eft --output data/performance/eft_current.json
```

`--compare data/performance/eft_reference.json` confronta i digest e i tempi
con il riferimento salvato localmente; `--reference-dir DIRECTORY` permette
di eseguire i cinque moduli originali da una copia separata del loro albero.
I dati completi della verifica sono in `data/performance/eft_reference.json`
e `data/performance/eft_optimized.json`.
