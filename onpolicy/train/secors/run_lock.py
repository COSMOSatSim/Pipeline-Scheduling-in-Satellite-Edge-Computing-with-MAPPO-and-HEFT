"""Presidio della cartella di una run di addestramento."""

import os
import socket
import subprocess
import time

NOME = ".run_lock"
# Oltre questa soglia il presidio e' considerato abbandonato.
SCADENZA_S = float(os.environ.get("SECORS_LOCK_TIMEOUT", "300"))

# Al passaggio di testimone fra due anelli della catena il successore parte non appena il
# predecessore termina, ma lo scheduler puo' darlo ancora in esecuzione per qualche istante:
# trovandolo vivo il successore si fermerebbe proprio quando deve subentrare.
TENTATIVI = int(os.environ.get("SECORS_LOCK_TENTATIVI", "7"))
PAUSA_S = float(os.environ.get("SECORS_LOCK_PAUSA", "20"))


def _percorso(run_dir) -> str:
    return os.path.join(str(run_dir), NOME)


def _descrizione() -> str:
    job = os.environ.get("SLURM_JOB_ID", "-")
    return "host=%s pid=%d job=%s" % (socket.gethostname(), os.getpid(), job)


def leggi(run_dir):
    """(contenuto, secondi dall'ultimo aggiornamento) o None se non c'e'."""
    p = _percorso(run_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            testo = f.read().strip()
        return testo, time.time() - os.path.getmtime(p)
    except OSError:
        return None


def _job_vivo(testo: str):
    """Il job che ha lasciato il presidio risulta ancora in esecuzione?"""
    for pezzo in testo.split():
        if pezzo.startswith("job=") and pezzo[4:] not in ("", "-"):
            job = pezzo[4:]
            break
    else:
        return None
    try:
        esito = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"],
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if esito.returncode != 0:      # job sconosciuto allo scheduler: terminato
        return False
    stato = esito.stdout.strip().upper()
    if not stato:
        return False
    # COMPLETING e i suoi simili compaiono ancora in coda ma non scrivono piu': trattarli come vivi
    # bloccherebbe il successore proprio al cambio.
    return stato.split()[0] in ("PENDING", "RUNNING", "CONFIGURING",
                                "SUSPENDED", "RESIZING", "REQUEUED")


def _crea_in_esclusiva(run_dir) -> bool:
    """Crea il presidio solo se non esiste."""
    try:
        fd = os.open(_percorso(run_dir), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_descrizione())
    return True


def acquisisci(run_dir) -> None:
    """Prende il presidio, o solleva se un altro processo lo tiene aggiornato."""
    if _crea_in_esclusiva(run_dir):
        return                       # cartella libera: presa senza contese

    stato = leggi(run_dir)
    if stato is None:                # tolto mentre lo leggevamo: si ritenta
        if _crea_in_esclusiva(run_dir):
            return
        stato = leggi(run_dir) or ("sconosciuto", 0.0)

    testo, eta = stato
    vivo = _job_vivo(testo)
    # Se il proprietario risulta vivo si insiste: al cambio di anello lo scheduler impiega qualche
    # istante ad aggiornare lo stato, e arrendersi al primo controllo interromperebbe la catena
    # senza motivo.
    for tentativo in range(TENTATIVI - 1):
        if not (vivo or (vivo is None and eta < SCADENZA_S)):
            break
        print("[run_lock] la cartella risulta presidiata (%s): attendo %.0f s "
              "e riprovo (%d di %d)" % (testo, PAUSA_S, tentativo + 1, TENTATIVI - 1),
              flush=True)
        time.sleep(PAUSA_S)
        if _crea_in_esclusiva(run_dir):
            return                   # rilasciato nel frattempo: presa senza contese
        stato = leggi(run_dir)
        if stato is None:
            if _crea_in_esclusiva(run_dir):
                return
            stato = leggi(run_dir) or ("sconosciuto", 0.0)
        testo, eta = stato
        vivo = _job_vivo(testo)

    if vivo or (vivo is None and eta < SCADENZA_S):
        raise RuntimeError(
            "La cartella %s e' gia' presidiata da un'altra esecuzione "
            "(%s, aggiornata %.0f s fa). Due processi sulla stessa run ne "
            "corrompono i checkpoint. Attendere la fine, oppure scegliere "
            "un'altra cartella di uscita. Se l'esecuzione precedente e' "
            "morta, rimuovere %s."
            % (run_dir, testo, eta, _percorso(run_dir)))

    if vivo is False:
        print("[run_lock] il job che presidiava la cartella (%s) non e' piu' "
              "in esecuzione: la run viene ripresa" % testo, flush=True)
    else:
        print("[run_lock] presidio scaduto (%.0f s, %s): la run risulta "
              "interrotta e viene ripresa" % (eta, testo), flush=True)

    # Subentro a un presidio abbandonato: si toglie e si ricrea in esclusiva, cosi' se due processi
    # arrivano insieme uno solo vince.
    try:
        os.remove(_percorso(run_dir))
    except OSError:
        pass
    if not _crea_in_esclusiva(run_dir):
        raise RuntimeError(
            "La cartella %s e' stata presa da un'altra esecuzione mentre si "
            "subentrava al presidio abbandonato." % run_dir)


def aggiorna(run_dir) -> None:
    """Segnala che la run e' viva."""
    try:
        with open(_percorso(run_dir), "w", encoding="utf-8") as f:
            f.write(_descrizione())
    except OSError:
        pass


def rilascia(run_dir) -> None:
    """Toglie il presidio a fine run."""
    try:
        os.remove(_percorso(run_dir))
    except OSError:
        pass
