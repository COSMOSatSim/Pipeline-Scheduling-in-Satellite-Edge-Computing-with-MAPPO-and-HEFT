"""
core/ - infrastruttura condivisa dai metodi di generalizzazione.

- abstract_eval.py : costo (C_T, C_E combinati come nella reward) di un actor
                     su una lista di scenari, via MDP astratto. È la "bilancia"
                     di validazione dei metodi regolati (greedy_soup,
                     lambda_scaling); una valutazione costa secondi.
"""
