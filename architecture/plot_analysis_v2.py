import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------

BASE_DIR = "MERGED_DATA"
OUTPUT_DIR = "ANALYSIS_PLOTS_V7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PATHS = {
    "results": os.path.join(BASE_DIR, "MERGED_all_results.csv"),
    "energy": os.path.join(BASE_DIR, "MERGED_all_energy_stats.csv")
}

# Filtri globali da applicare a TUTTI i grafici
GLOBAL_FILTERS = {
    'deadline': 10,
    # 'energy_budget': 100000, # Rimosso per permettere grafici vs. budget
    # 'CPU': 0.05,
    # 'AT': 0.1
}

# Impostazioni globali per i colori
sns.set_theme(style="whitegrid")
PALETTE = sns.color_palette("colorblind")

print(f"Script di analisi V2 (Specifiche 11/11) avviato.")
print(f"Grafici salvati in: {os.path.abspath(OUTPUT_DIR)}")
print(f"Filtri globali applicati: {GLOBAL_FILTERS}")


# ----------------------------------------------------------------------
# FUNZIONI HELPER
# ----------------------------------------------------------------------

def disambiguate_solvers(df):
    """
    Se il solver è ILP e ci sono colonne w_e/w_R, rinomina il solver
    in 'ILP (0.7, 0.3)' per separarli nei grafici.
    """
    if df is None or df.empty:
        return df

    # Verifica se le colonne esistono
    if 'w_e' in df.columns and 'w_R' in df.columns:
        # Crea una maschera per le righe ILP che hanno i pesi definiti
        mask = (df['solver'] == 'ILP') & (df['w_e'].notna()) & (df['w_R'].notna())

        if mask.any():
            print("  Disambiguazione ILP in corso...")
            # Applica la rinomina: ILP -> ILP (we=0.7, wR=0.3)
            df.loc[mask, 'solver'] = df.loc[mask].apply(
                lambda row: f"ILP ($w_e$={row['w_e']}, $w_R$={row['w_R']})", axis=1
            )
    return df

def apply_filters(df, filters):
    """Applica i filtri globali al DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()
    df_filtered = df.copy()
    print(f"  Filtraggio... (righe iniziali: {len(df_filtered)}) ", end="")
    for col, value in filters.items():
        if value is not None and col in df_filtered.columns:
            try:
                col_type = df_filtered[col].dtype
                typed_value = pd.Series([value]).astype(col_type).iloc[0]
                mask = (df_filtered[col] == typed_value)
            except Exception:
                mask = (df_filtered[col] == value)
            df_filtered = df_filtered[mask]
    print(f"-> (righe finali: {len(df_filtered)})")
    if df_filtered.empty:
        print("  [Attenzione] Nessun dato rimasto dopo il filtraggio.")
    return df_filtered


def save_and_close(fig, path):
    """Funzione helper per salvare e chiudere i plot."""
    try:
        fig.savefig(path, bbox_inches="tight")
        print(f"  ✅ Salvato: {os.path.abspath(path)}")
        plt.close(fig)
    except Exception as e:
        print(f"  [Errore Salvataggio] {e}")


def create_line_plot_with_errors(
        df_stats, x_col, y_col_mean, y_col_error,
        x_label, y_label, filename, legend_title="Algorithms",
        use_precomputed_errors=False  # True = uso mean/error nel DF; False = seaborn calcola CI/SE dai dati grezzi
):
    """
    Funzione di plotting che supporta due modalità:
      - use_precomputed_errors=True: df_stats contiene mean + error (SEM). Rimuove duplicati e
        disegna lineplot senza banda di seaborn (errorbar=None) e disegna ax.errorbar coi SEM.
      - use_precomputed_errors=False: df_stats è il DF grezzo (più righe per seed). Lascia a seaborn
        il calcolo dell'errorbar/band (errorbar='se' o ('ci',95)). NON chiamare ax.errorbar in questo caso.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Ensure numeric x
        df_stats = df_stats.copy()
        df_stats[x_col] = pd.to_numeric(df_stats[x_col], errors='coerce')
        df_stats = df_stats.sort_values(by=[x_col])

        df_stats['Algorithms'] = df_stats['solver']

        if use_precomputed_errors:
            # 1. Preparazione Dati
            df_plot = df_stats.drop_duplicates(subset=['Algorithms', x_col]).copy()
            unique_algorithms = sorted(df_plot['Algorithms'].unique())

            # 2. Palette Dinamica: Assegna un COLORE diverso a OGNI nome completo
            # (es. ILP 0.7 avrà un colore diverso da ILP 0.5)
            palette = dict(zip(unique_algorithms, sns.color_palette(PALETTE, len(unique_algorithms))))

            # 3. Definizione Stili BASE (Famiglie): Definisce solo Marker e Linea
            base_styles = {
                'DTS-APopt': {'marker': 'o', 'ls': '-'},  # Cerchio, Solida
                'DTS-base': {'marker': 'x', 'ls': '--'},  # X, Tratteggiata
                'ILP': {'marker': 's', 'ls': ':'},  # Quadrato, Puntinata (Dotted)
                'OrbitAware': {'marker': 'D', 'ls': '-.'}  # Diamante, Dash-dot
            }

            # === PASSO 1: Disegno con logica "Contiene" ===
            for algorithm in unique_algorithms:
                subset = df_plot[df_plot['Algorithms'] == algorithm]
                if subset.empty:
                    continue

                # -- Logica di Matching Dinamico --
                # Default (se qualcosa non matcha nulla)
                marker = 'o'
                ls = '-'

                # Cerchiamo se il nome dell'algoritmo (es. "ILP (we=0.7...)")
                # CONTIENE una delle chiavi base (es. "ILP")
                found_style = False
                for key, props in base_styles.items():
                    if key in algorithm:
                        marker = props['marker']
                        ls = props['ls']
                        found_style = True
                        break

                # Opzionale: Se vuoi saltare algoritmi non riconosciuti, scommenta qui:
                # if not found_style: continue

                # Il colore lo prendiamo dalla palette specifica per QUESTA variante precisa
                color = palette[algorithm]

                x_vals = pd.to_numeric(subset[x_col], errors='coerce').values
                y_vals = pd.to_numeric(subset[y_col_mean], errors='coerce').values
                y_errs = pd.to_numeric(subset[y_col_error], errors='coerce').values

                ax.errorbar(
                    x_vals, y_vals, yerr=y_errs,
                    fmt=marker + ls,  # Combina marker e stile linea (es. 's:')
                    capsize=5,
                    color=color,  # Usa il colore specifico della variante
                    elinewidth=1.8,
                    label=algorithm,  # Nome completo per la legenda
                    markersize=6
                )

            # Ridisegniamo la legenda
            ax.legend(title=legend_title)

            x_ticks = sorted(df_plot[x_col].unique())
            ax.set_xticks(x_ticks)

        else:
            # Modalità: lascia che seaborn calcoli il CI/SE dai dati grezzi
            # Qui y_col_mean deve essere il nome della colonna "grezza" da aggregare (es. 'Time in system')
            unique_algorithms = sorted(df_stats['Algorithms'].unique())
            palette = dict(zip(unique_algorithms, sns.color_palette(PALETTE, len(unique_algorithms))))

            # Esempio: mostra SE (errorbar='se') oppure 95% CI (errorbar=('ci', 95))
            # Scegli a piacere 'se' o ('ci', 95)
            sns.lineplot(
                data=df_stats, x=x_col, y=y_col_mean,
                hue='Algorithms', style='Algorithms',
                markers=True, markersize=8, lw=2.5,
                ax=ax, palette=palette,
                legend=True,
                estimator='mean',
                errorbar='se',  # oppure errorbar=('ci', 95)
                err_style='band'  # 'band' o 'bars'
            )
            # NON disegnare ax.errorbar in questa modalità

            x_ticks = sorted(df_stats[x_col].unique())
            ax.set_xticks(x_ticks)

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Imposta il minimo a 0 e aggiungi piccolo margine sopra
        ax.set_ylim(bottom=0)
        current_max_y = ax.get_ylim()[1]
        ax.set_ylim(top=current_max_y * 1.05)

        # Legenda
        if legend_title:
            ax.legend(title=legend_title)
        else:
            ax.legend(title=None)

        save_and_close(fig, os.path.join(OUTPUT_DIR, filename))

    except Exception as e:
        print(f"  [Errore] Creazione grafico {filename}: {e}")


# ----------------------------------------------------------------------
# PRE-PROCESSING (Esecuzione unica)
# ----------------------------------------------------------------------

def preprocess_data(results_df, energy_df):
    """
    Esegue tutti i calcoli statistici "per-seed" (per seme)
    e restituisce i DataFrame aggregati (media e errore) pronti per il plot.
    """
    print("Avvio pre-processing dei dati...")

    # --- 1. Inverti AT -> Arrival Rate ---
    at_map = {0.5: 2, 0.25: 4, 0.166: 6, 0.16: 6, 0.125: 8, 0.1: 10}

    def map_at(df):
        if 'AT' in df.columns:
            df['AT_rounded'] = df['AT'].round(3)
            at_map_rounded = {round(k, 3): v for k, v in at_map.items()}
            df['Arrival Rate'] = df['AT_rounded'].map(at_map_rounded)
            if df['Arrival Rate'].isna().any():
                unmapped = df[df['Arrival Rate'].isna()]['AT'].unique()
                print(f"  [Attenzione] Trovati valori AT non mappati: {unmapped}. Tento inversione...")
                df['Arrival Rate'] = df['Arrival Rate'].fillna(1 / pd.to_numeric(df['AT'], errors='coerce'))
            df = df.drop(columns=['AT_rounded'])
        return df

    results_df = map_at(results_df)
    energy_df = map_at(energy_df)

    # Colonne che definiscono un "esperimento" (il 'seed' è la variabile N)
    experiment_cols = ['solver', 'Arrival Rate', 'energy_budget']

    # --- 2. Calcola Success/Failure Rate (per-seed) ---
    df_success_rate = pd.DataFrame()
    if 'Status' in results_df.columns:
        df_sr_per_seed = results_df.groupby(
            experiment_cols + ['seed']
        )['Status'].apply(lambda x: (x == 'Completed').mean()).reset_index(name='Success Rate')

        # Calcola media e SEM (Errore Standard della Media) sui seed
        df_success_rate = df_sr_per_seed.groupby(experiment_cols)['Success Rate'].agg(
            mean='mean',
            error='sem'  # <-- Errore Standard della Media (SEM)
        ).reset_index()

    # --- 3. Calcola Energy per Task (per-seed) ---
    df_energy_per_task = pd.DataFrame()
    if 'Energy_TOTAL [J]' in results_df.columns:
        df_agg = results_df.groupby(
            experiment_cols + ['seed']
        ).agg(
            Total_Energy_Consumed=('Energy_TOTAL [J]', 'sum'),
            Num_Completed_Tasks=('Status', lambda x: (x == 'Completed').sum())
        )
        df_agg['Energy per Task'] = df_agg['Total_Energy_Consumed'] / df_agg['Num_Completed_Tasks']
        df_agg.replace([np.inf, -np.inf], 0, inplace=True)
        df_energy_per_task = df_agg.groupby(experiment_cols)['Energy per Task'].agg(
            mean='mean', error='sem'
        ).reset_index()

    # --- 4. Calcola Energy per SEN (per-seed) ---
    df_energy_per_sen = pd.DataFrame()
    if 'Total_Energy_Consumed_J' in energy_df.columns:
        energy_df['is_Used'] = (energy_df['Total_Tasks_Completed'] > 0).astype(int)
        df_sen_agg = energy_df.groupby(
            experiment_cols + ['seed']
        ).agg(
            Total_Energy_Consumed=('Total_Energy_Consumed_J', 'sum'),
            Num_Used_Servers=('is_Used', 'sum')
        )
        df_sen_agg['Energy per SEN'] = df_sen_agg['Total_Energy_Consumed'] / df_sen_agg['Num_Used_Servers']
        df_sen_agg.replace([np.inf, -np.inf], 0, inplace=True)
        df_energy_per_sen = df_sen_agg.groupby(experiment_cols)['Energy per SEN'].agg(
            mean='mean', error='sem'
        ).reset_index()

    # --- 5. Calcola Response Time e Routing Time (per-seed) ---
    df_response_time = pd.DataFrame()
    df_routing_time = pd.DataFrame()
    df_hops = pd.DataFrame()

    df_completed = results_df[results_df['Status'] == 'Completed'].copy()

    if 'Time in system' in df_completed.columns:
        df_rt_agg = df_completed.groupby(experiment_cols + ['seed'])['Time in system'].agg(
            Avg='mean',
            P90=lambda x: x.quantile(0.90),
            P95=lambda x: x.quantile(0.95)
        ).reset_index()
        df_response_time = df_rt_agg.groupby(experiment_cols).agg(
            Avg_mean=('Avg', 'mean'), Avg_error=('Avg', 'sem'),
            P90_mean=('P90', 'mean'), P90_error=('P90', 'sem'),
            P95_mean=('P95', 'mean'), P95_error=('P95', 'sem'),
        ).reset_index()

    if 'Routing Duration' in results_df.columns:
        df_rd_agg = results_df.groupby(experiment_cols + ['seed'])['Routing Duration'].agg(
            Avg='mean',
            P90=lambda x: x.quantile(0.90),
            P95=lambda x: x.quantile(0.95)
        ).reset_index()
        df_routing_time = df_rd_agg.groupby(experiment_cols).agg(
            Avg_mean=('Avg', 'mean'), Avg_error=('Avg', 'sem'),
            P90_mean=('P90', 'mean'), P90_error=('P90', 'sem'),
            P95_mean=('P95', 'mean'), P95_error=('P95', 'sem'),
        ).reset_index()

    # Calcolo Hops (media, min, max) ---
    if 'Num Hops Routing' in results_df.columns and 'Num Hops' in results_df.columns:
        df_hops_agg = results_df.groupby(experiment_cols + ['seed']).agg(
            Routing_Avg=('Num Hops Routing', 'mean'),
            Routing_Max=('Num Hops Routing', 'max'),
            Routing_Min=('Num Hops Routing', 'min'),
        ).reset_index()
        df_hops_agg_exec = df_completed.groupby(experiment_cols + ['seed']).agg(
            Exec_Avg=('Num Hops', 'mean'),
            Exec_Max=('Num Hops', 'max'),
            Exec_Min=('Num Hops', 'min')
        ).reset_index()

        # Unisci i due tipi di hop
        df_hops_agg = pd.merge(df_hops_agg, df_hops_agg_exec, on=experiment_cols + ['seed'])

        # Calcola media e SEM sui seed per tutte le metriche
        df_hops = df_hops_agg.groupby(experiment_cols).agg(
            Routing_Avg_mean=('Routing_Avg', 'mean'), Routing_Avg_error=('Routing_Avg', 'sem'),
            Routing_Max_mean=('Routing_Max', 'mean'), Routing_Max_error=('Routing_Max', 'sem'),
            Routing_Min_mean=('Routing_Min', 'mean'), Routing_Min_error=('Routing_Min', 'sem'),
            Exec_Avg_mean=('Exec_Avg', 'mean'), Exec_Avg_error=('Exec_Avg', 'sem'),
            Exec_Max_mean=('Exec_Max', 'mean'), Exec_Max_error=('Exec_Max', 'sem'),
            Exec_Min_mean=('Exec_Min', 'mean'), Exec_Min_error=('Exec_Min', 'sem')
        ).reset_index()

    print("Pre-processing completato.")

    return results_df, df_success_rate, df_energy_per_task, df_energy_per_sen, df_response_time, df_routing_time, df_hops


def preprocess_data_load_balancing(energy_df):
    """
    Calcola il Dynamic Load Balancing Control Degree (Rho) per ogni esperimento.
    Rho = Average Server Load / Maximum Server Load
    """
    print("Avvio pre-processing: Dynamic Load Balancing Degree...")
    if 'Total_Tasks_Completed' not in energy_df.columns:
        print("  [Errore] Colonna 'Total_Tasks_Completed' non trovata in energy_df.")
        return pd.DataFrame()

    experiment_cols = ['solver', 'Arrival Rate', 'energy_budget', 'seed']

    # 1. Calcola Avg e Max carico PER esperimento (per-seed)
    df_agg = energy_df.groupby(experiment_cols)['Total_Tasks_Completed'].agg(
        Avg_Load='mean',
        Max_Load='max'
    ).reset_index()

    # 2. Calcola il "Degree" (Rho)
    df_agg['Load Balancing Degree'] = df_agg['Avg_Load'] / df_agg['Max_Load']
    # Se Max_Load è 0 (nessun task), imposta il degree a 0 (o 1, a seconda della logica)
    df_agg['Load Balancing Degree'] = df_agg['Load Balancing Degree'].fillna(0)

    # 3. Calcola la Media e l'Errore (SEM) sui seed
    stats_cols = ['solver', 'Arrival Rate', 'energy_budget']
    df_stats = df_agg.groupby(stats_cols)['Load Balancing Degree'].agg(
        mean='mean',
        error='sem'
    ).reset_index()

    return df_stats


# ----------------------------------------------------------------------
# FUNZIONI DI PLOTTING
# ----------------------------------------------------------------------
# Ogni funzione prende un DataFrame di statistiche pre-calcolato

# --- 1. Success/Failure Rate ---
def plot_1_success_rate(df_stats):
    print("Generazione grafici 1.x: Success Rate")
    # vs. Arrival Rate
    create_line_plot_with_errors(
        df_stats,
        x_col='Arrival Rate', y_col_mean='mean', y_col_error='error',
        x_label='Arrival Rate (requests/second)', y_label='Success Rate (%)',
        filename="1.1_success_rate_vs_arrival_rate.png"
    )
    # vs. Energy Budget
    create_line_plot_with_errors(
        df_stats,
        x_col='energy_budget', y_col_mean='mean', y_col_error='error',
        x_label='Energy Budget (J)', y_label='Success Rate (%)',
        filename="1.2_success_rate_vs_energy_budget.png"
    )


# --- 2. Energy Consumption ---
def plot_2_energy(df_energy_per_task, df_energy_per_sen, df_success_rate=None):
    print("Generazione grafici 2.x: Energy Consumption")
    # --- NUOVO BLOCCO: Calcolo e Stampa Tabella Sintetica (Arrival Rate = 10) ---
    if df_success_rate is not None and 'Arrival Rate' in df_energy_per_task.columns:
        try:
            # 1. Unione dei dataframe (Energia completati + Success Rate)
            # Usiamo 'inner' join su solver, Arrival Rate ed energy_budget
            merged = pd.merge(
                df_energy_per_task,
                df_success_rate,
                on=['solver', 'Arrival Rate', 'energy_budget'],
                suffixes=('_energy', '_sr')
            )

            # 2. Calcolo dell'Energia Reale per ogni riga
            # E_real = E_completed * Success_Rate + (0 * Failure_Rate)
            merged['E_real_avg'] = merged['mean_energy'] * merged['mean_sr']

            # 3. Filtra per Arrival Rate = 10
            subset_10 = merged[merged['Arrival Rate'] == 10.0].copy()

            if not subset_10.empty:
                # 4. Aggregazione: Facciamo la media sui diversi Energy Budget
                # (visto che dai grafici sappiamo che il consumo per task è costante al variare del budget)
                summary = subset_10.groupby('solver').agg({
                    'mean_energy': 'mean',  # Media dell'energia (solo completati)
                    'mean_sr': 'mean',  # Media del Success Rate
                    'E_real_avg': 'mean'  # Media dell'Energia Reale
                }).reset_index()

                print("\n" + "=" * 85)
                print(f"{'ANALISI CONSUMO ENERGETICO (Arrival Rate = 10 req/s)':^85}")
                print(f"{'(Media aggregata sui vari energy budget)':^85}")
                print("=" * 85)
                print(f"{'ALGORITHM':<15} | {'E_AVG (Compl.)':<18} | {'SUCCESS RATE':<15} | {'E_REAL (Total)':<15}")
                print("-" * 85)

                for _, row in summary.iterrows():
                    print(f"{row['solver']:<15} | "
                          f"{row['mean_energy']:>10.2f} J      | "
                          f"{row['mean_sr'] * 100:>10.1f} %    | "
                          f"{row['E_real_avg']:>10.2f} J")
                print("=" * 85 + "\n")
            else:
                print("  [Info] Nessun dato trovato per Arrival Rate = 10.")

        except Exception as e:
            print(f"  [Errore Calcolo Tabella Energia] {e}")
    # --------------------------------------------------

    # Energy per Task vs. Arrival Rate
    create_line_plot_with_errors(
        df_energy_per_task,
        x_col='Arrival Rate', y_col_mean='mean', y_col_error='error',
        x_label='Arrival Rate (requests/second)', y_label='Average Energy per Task (J)',
        filename="2.1_energy_per_task_vs_arrival_rate.png"
    )
    # Energy per Task vs. Energy Budget
    # create_line_plot_with_errors( # COMMENTATO
    #     df_energy_per_task,
    #     x_col='energy_budget', y_col_mean='mean', y_col_error='error',
    #     x_label='Energy Budget (J)', y_label='Average Energy per Task (J)',
    #     filename="2.2_energy_per_task_vs_energy_budget.png"
    # )
    # Energy per SEN vs. Arrival Rate
    create_line_plot_with_errors(
        df_energy_per_sen,
        x_col='Arrival Rate', y_col_mean='mean', y_col_error='error',
        x_label='Arrival Rate (requests/second)', y_label='Average Energy per Used SEN (J)',
        filename="2.3_energy_per_sen_vs_arrival_rate.png"
    )
    # Energy per SEN vs. Energy Budget
    create_line_plot_with_errors(
        df_energy_per_sen,
        x_col='energy_budget', y_col_mean='mean', y_col_error='error',
        x_label='Energy Budget (J)', y_label='Average Energy per Used SEN (J)',
        filename="2.4_energy_per_sen_vs_energy_budget.png"
    )


# --- 3. Response Time (Time in system) ---
def plot_3_response_time(df_stats):
    print("Generazione grafici 3.x: Response Time (Avg, P90, P95)")
    # vs. Arrival Rate
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Avg_mean', 'Avg_error', 'Arrival Rate (requests/second)',
                                 'AVG Response Time (s)', '3.1_response_time_AVG_vs_arrival_rate.png')
    # create_line_plot_with_errors(df_stats, 'Arrival Rate', 'P90_mean', 'P90_error', 'Arrival Rate (requests/second)',
    #                              'P90 Response Time (s)', '3.2_response_time_P90_vs_arrival_rate.png') # COMMENTATO
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'P95_mean', 'P95_error', 'Arrival Rate (requests/second)',
                                 'P95 Response Time (s)', '3.3_response_time_P95_vs_arrival_rate.png')
    # vs. Energy Budget
    create_line_plot_with_errors(df_stats, 'energy_budget', 'Avg_mean', 'Avg_error', 'Energy Budget (J)',
                                 'AVG Response Time (s)', '3.4_response_time_AVG_vs_energy_budget.png')
    # create_line_plot_with_errors(df_stats, 'energy_budget', 'P90_mean', 'P90_error', 'Energy Budget (J)',
    #                              'P90 Response Time (s)', '3.5_response_time_P90_vs_energy_budget.png') # COMMENTATO
    create_line_plot_with_errors(df_stats, 'energy_budget', 'P95_mean', 'P95_error', 'Energy Budget (J)',
                                 'P95 Response Time (s)', '3.6_response_time_P95_vs_energy_budget.png')


# --- 4. Routing Time ---
def plot_4_routing_time(df_stats):
    print("Generazione grafici 4.x: Routing Time (Avg, P90, P95)")
    # vs. Arrival Rate
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Avg_mean', 'Avg_error', 'Arrival Rate (requests/second)',
                                 'AVG Routing Time (s)', '4.1_routing_time_AVG_vs_arrival_rate.png')
    # create_line_plot_with_errors(df_stats, 'Arrival Rate', 'P90_mean', 'P90_error', 'Arrival Rate (requests/second)',
    #                              'P90 Routing Time (s)', '4.2_routing_time_P90_vs_arrival_rate.png') # COMMENTATO
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'P95_mean', 'P95_error', 'Arrival Rate (requests/second)',
                                 'P95 Routing Time (s)', '4.3_routing_time_P95_vs_arrival_rate.png')
    # vs. Energy Budget
    create_line_plot_with_errors(df_stats, 'energy_budget', 'Avg_mean', 'Avg_error', 'Energy Budget (J)',
                                 'AVG Routing Time (s)', '4.4_routing_time_AVG_vs_energy_budget.png')
    # create_line_plot_with_errors(df_stats, 'energy_budget', 'P90_mean', 'P90_error', 'Energy Budget (J)',
    #                              'P90 Routing Time (s)', '4.5_routing_time_P90_vs_energy_budget.png') # COMMENTATO
    create_line_plot_with_errors(df_stats, 'energy_budget', 'P95_mean', 'P95_error', 'Energy Budget (J)',
                                 'P95 Routing Time (s)', '4.6_routing_time_P95_vs_energy_budget.png')


# --- 5. Hops ---
def plot_5_hops(df_stats):
    print("Generazione grafici 5.x: Hops (Routing e Executed)")
    # vs. Arrival Rate
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Routing_Avg_mean', 'Routing_Avg_error',
                                 'Arrival Rate (requests/second)', 'AVG Hops (Routing Decision)',
                                 '5.1_routing_hops_AVG_vs_arrival_rate.png')
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Routing_Max_mean', 'Routing_Max_error',
                                 'Arrival Rate (requests/second)', 'MAX Hops (Routing Decision)',
                                 '5.2_routing_hops_MAX_vs_arrival_rate.png')
    # create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Routing_Min_mean', 'Routing_Min_error',
    #                              'Arrival Rate (requests/second)', 'MIN Hops (Routing Decision)',
    #                              '5.3_routing_hops_MIN_vs_arrival_rate.png') # COMMENTATO

    # Grafico 5.4: Utilizza la media (Exec_Avg_mean) come richiesto
    create_line_plot_with_errors(df_stats, 'Arrival Rate', 'Exec_Avg_mean', 'Exec_Avg_error',
                                 'Arrival Rate (requests/second)', 'AVG Hops (Executed)',
                                 '5.4_executed_hops_AVG_vs_arrival_rate.png')
    # vs. Energy Budget
    create_line_plot_with_errors(df_stats, 'energy_budget', 'Routing_Avg_mean', 'Routing_Avg_error',
                                 'Energy Budget (J)', 'AVG Hops (Routing Decision)',
                                 '5.5_routing_hops_AVG_vs_energy_budget.png')
    create_line_plot_with_errors(df_stats, 'energy_budget', 'Exec_Avg_mean', 'Exec_Avg_error', 'Energy Budget (J)',
                                 'AVG Hops (Executed)', '5.6_executed_hops_AVG_vs_energy_budget.png')


# --- 6. Failure Reason ---

def plot_6_1_failure_reason(df_results, arrival_rates_to_plot):
    """
    GRAFICO 6: Motivo dei Fallimenti (Stacked Bar) in percentuale.
    """
    print(f"Generazione grafico: 6.0 Motivo Fallimenti per AT={arrival_rates_to_plot} (in %)")
    try:

        df_failed = df_results[
            (df_results['Status'] != 'Completed') &
            (df_results['Rejection Reason'].notna())
            ].copy()

        # CORREZIONE: Pulisci gli spazi bianchi per unire categorie duplicate
        df_failed['Rejection Reason'] = df_failed['Rejection Reason'].str.strip()

        # Filtra solo per i tassi di arrivo richiesti
        df_filtered = df_failed[df_failed['Arrival Rate'].isin(arrival_rates_to_plot)]
        if df_filtered.empty:
            print("  [Skip] Nessun dato sui fallimenti per gli AT specificati.")
            return

        # Calcola il conteggio totale per (solver, AT, Reason, seed)
        df_agg = df_filtered.groupby(
            ['solver', 'Arrival Rate', 'Rejection Reason', 'seed']
        ).size().reset_index(name='Count')

        # Calcola la media dei conteggi sui seed
        df_stats = df_agg.groupby(
            ['solver', 'Arrival Rate', 'Rejection Reason']
        )['Count'].mean().reset_index()

        # Crea un grafico separato per ogni AT
        for at in arrival_rates_to_plot:
            df_plot = df_stats[df_stats['Arrival Rate'] == at]
            if df_plot.empty: continue

            # --- NUOVO PASSAGGIO: CALCOLO DELLA PERCENTUALE ---

            # 1. Calcola il totale dei fallimenti (Count_Total) per ogni solver
            df_total_failures = df_plot.groupby('solver')['Count'].sum().reset_index(name='Count_Total')

            # 2. Unisci il totale ai dati di plot
            df_plot = pd.merge(df_plot, df_total_failures, on='solver')

            # 3. Calcola la percentuale di ogni motivo di fallimento
            df_plot['Percentage'] = (df_plot['Count'] / df_plot['Count_Total']) * 100

            # ----------------------------------------------------

            df_pivot = df_plot.pivot(
                index='solver', columns='Rejection Reason', values='Percentage'  # USA 'Percentage'
            ).fillna(0)

            fig, ax = plt.subplots(figsize=(11, 7))
            df_pivot.plot(
                kind='bar', stacked=True, ax=ax, cmap='Set3', width=0.5
            )

            ax.set_xlabel('Algorithms', fontsize=12)
            # MODIFICA 1: Etichetta Asse Y
            ax.set_ylabel('Rejection Reasons', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            # MODIFICA 2: Rimuovi il titolo della Legenda
            ax.legend(title=None, loc='best')

            ax.grid(True, linestyle='--', alpha=0.6, axis='y')
            # Imposta l'asse Y per arrivare a 100% (se non è già il caso)
            ax.set_ylim(0, 100)
            sns.despine(ax=ax)
            save_and_close(fig, os.path.join(OUTPUT_DIR,
                                             f"6.0_failure_reason_AT_{at}_PERCENT.png"))  # Aggiornamento Nome File

    except Exception as e:
        print(f"  [Errore] plot_6_failure_reason: {e}")


# --- 7. Load Balancing Heatmap ---
def plot_load_balancing_heatmap_V2(df_energy):
    """
    GRAFICO 7: Heatmap del Bilanciamento del Carico (Top 15 SEN)
    """
    # print("Generazione grafico: 7.0 Heatmap Bilanciamento del Carico (Top 15 SEN)") # COMMENTATO
    # try:
    #     # ... Logica omessa ...
    #     save_and_close(fig, os.path.join(OUTPUT_DIR, "7.0_load_balancing_heatmap.png"))
    # except Exception as e:
    #     print(f"  [Errore] plot_load_balancing_heatmap: {e}")
    pass  # Funzione disabilitata


def plot_7_load_balancing_heatmaps(df_energy):
    """
    GRAFICO 7: Heatmap del Bilanciamento del Carico (Top 15 SEN)
    """
    # print("Generazione grafici 7.x: Heatmap Bilanciamento Carico") # COMMENTATO

    # # --- GRAFICO 7.1: vs. Arrival Rate ---
    # try:
    #     # ... Logica omessa ...
    #     save_and_close(g.fig, os.path.join(OUTPUT_DIR, "7.1_heatmap_load_vs_arrival_rate.png"))
    # except Exception as e:
    #     print(f"  [Errore] plot_7_load_balancing_heatmaps (vs AT): {e}")

    # # --- GRAFICO 7.2: vs. Energy Budget ---
    # try:
    #     # ... Logica omessa ...
    #     save_and_close(g.fig, os.path.join(OUTPUT_DIR, "7.2_heatmap_load_vs_energy_budget.png"))
    # except Exception as e:
    #     print(f"  [Errore] plot_7_load_balancing_heatmaps (vs Budget): {e}")
    pass  # Funzione disabilitata


def plot_8_load_balancing_degree(df_stats):
    """
    GRAFICO 8: Dynamic Load Balancing Control Degree
    """
    print("Generazione grafici 8.x: Dynamic Load Balancing Degree")

    # --- 8.1 vs. Arrival Rate ---
    plot_data_at = df_stats[df_stats['Arrival Rate'].notna()]
    if not plot_data_at.empty:
        create_line_plot_with_errors(
            plot_data_at,
            x_col='Arrival Rate', y_col_mean='mean', y_col_error='error',
            x_label='Arrival Rate (requests/second)', y_label='Dynamic Load Balancing Degree (Ratio)',
            filename="8.1_load_balancing_degree_vs_arrival_rate.png"
        )

    # --- 8.2 vs. Energy Budget ---
    plot_data_eb = df_stats[df_stats['energy_budget'].notna()]
    if not plot_data_eb.empty:
        create_line_plot_with_errors(
            plot_data_eb,
            x_col='energy_budget', y_col_mean='mean', y_col_error='error',
            x_label='Energy Budget (J)', y_label='Dynamic Load Balancing Degree (Ratio)',
            filename="8.2_load_balancing_degree_vs_energy_budget.png"
        )


# ----------------------------------------------------------------------
# ESECUZIONE MAIN
# ----------------------------------------------------------------------
def main():
    # --- 1. CARICAMENTO DATI ---
    dataframes = {}
    for key, path in FILE_PATHS.items():
        if os.path.exists(path):
            try:
                print(f"Caricamento: {path}")
                dataframes[key] = pd.read_csv(path)
                print(f"File caricato: {path} ({len(dataframes[key])} righe)")
            except Exception as e:
                print(f"Errore caricamento {path}: {e}")
                dataframes[key] = pd.DataFrame()
        else:
            print(f"File non trovato (saltato): {path}")
            dataframes[key] = pd.DataFrame()

    df_results_raw = dataframes.get("results")
    df_energy_raw = dataframes.get("energy")

    df_results_raw = disambiguate_solvers(df_results_raw)
    df_energy_raw = disambiguate_solvers(df_energy_raw)

    if df_results_raw is None or df_results_raw.empty:
        print("ERRORE: 'MERGED_all_results.csv' non trovato o vuoto. Impossibile continuare.")
        return
    if df_energy_raw is None or df_energy_raw.empty:
        print("ERRORE: 'MERGED_all_energy_stats.csv' non trovato o vuoto. Impossibile continuare.")
        return

    # --- 2. FILTRAGGIO GLOBALE ---
    df_results_filtered = apply_filters(df_results_raw, GLOBAL_FILTERS)
    df_energy_filtered = apply_filters(df_energy_raw, GLOBAL_FILTERS)

    # --- 3. PRE-PROCESSING ---
    # Esegui i calcoli (inversione AT, metriche /task, /sen)
    (df_results_processed,  # Questo è df_results filtrato E con 'Arrival Rate'
     df_success_rate,
     df_energy_per_task,
     df_energy_per_sen,
     df_response_time,
     df_routing_time,
     df_hops) = preprocess_data(df_results_filtered, df_energy_filtered)

    # --- 4. PRE-PROCESSING ---
    # Calcola la nuova metrica di Load Balancing da df_energy_filtered
    df_load_balancing_stats = preprocess_data_load_balancing(df_energy_filtered)

    # --- 4.1 ESECUZIONE PLOTTING ---

    # Filtra per assi X
    if 'Arrival Rate' in df_results_processed.columns:
        print("\n--- INIZIO GRAFICI (vs. Arrival Rate) ---")
        plot_1_success_rate(df_success_rate[df_success_rate['Arrival Rate'].notna()])
        plot_2_energy(
            df_energy_per_task[df_energy_per_task['Arrival Rate'].notna()],
            df_energy_per_sen[df_energy_per_sen['Arrival Rate'].notna()],
            df_success_rate[df_success_rate['Arrival Rate'].notna()]  # Nuovo argomento
        )
        '''plot_2_energy(df_energy_per_task[df_energy_per_task['Arrival Rate'].notna()],
                      df_energy_per_sen[df_energy_per_sen['Arrival Rate'].notna()])'''
        plot_3_response_time(df_response_time[df_response_time['Arrival Rate'].notna()])
        plot_4_routing_time(df_routing_time[df_routing_time['Arrival Rate'].notna()])
        plot_5_hops(df_hops[df_hops['Arrival Rate'].notna()])

    # Filtra per assi X
    if 'energy_budget' in df_results_processed.columns:
        print("\n--- INIZIO GRAFICI (vs. Energy Budget) ---")
        plot_1_success_rate(df_success_rate[df_success_rate['energy_budget'].notna()])
        plot_2_energy(df_energy_per_task[df_energy_per_task['energy_budget'].notna()],
                      df_energy_per_sen[df_energy_per_sen['energy_budget'].notna()])
        plot_3_response_time(df_response_time[df_response_time['energy_budget'].notna()])
        plot_4_routing_time(df_routing_time[df_routing_time['energy_budget'].notna()])
        plot_5_hops(df_hops[df_hops['energy_budget'].notna()])

    print("\n--- INIZIO GRAFICI SPECIALI ---")
    # solo per AT 10 e 8 (inter-arrival 0.1 e 0.125)
    plot_6_1_failure_reason(df_results_processed, arrival_rates_to_plot=[10.0, 8.0])

    # Per la heatmap usiamo tutti i dati filtrati (non servono quelli pre-processati)
    # plot_7_load_balancing_heatmaps(df_energy_filtered) # COMMENTATO (7.1, 7.2)
    # plot_load_balancing_heatmap_V2(df_energy_filtered) # COMMENTATO (7.0)

    # --- 6. PLOT NUOVA METRICA ---
    print("\n--- INIZIO GRAFICI (Dynamic Load Balancing Degree) ---")
    plot_8_load_balancing_degree(df_load_balancing_stats)

    print("\n--- Analisi aggregata V2 completata. ---")
    print(f"Grafici salvati in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()