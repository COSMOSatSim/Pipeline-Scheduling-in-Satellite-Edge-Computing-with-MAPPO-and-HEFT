"""Driver SEC-MPH (Algorithm 1) sopra onpolicy."""
import time
import os

from onpolicy.train.secors import run_lock
import csv
import numpy as np
import torch
from onpolicy.runner.core.base_runner import Runner, salva_atomico
from onpolicy.runner.secors.training_html import generate_training_html
from onpolicy.runner.secors.training_timing import TrainingTiming


def _t2n(x):
    return x.detach().cpu().numpy()


class SECORSRunner(Runner):
    """Runner per il training/eval di SECORS (parameter sharing, CTDE)."""

    # ------------------------------------------------------------------ #
    # Setup & resume
    # ------------------------------------------------------------------ #
    def __init__(self, config):
        super(SECORSRunner, self).__init__(config)
        # Confine fra indici SAT e indici SAS/GND nella device list: e' il numero di SAT, non quello
        # degli agenti (solo un sottoinsieme dei SAT genera).
        self.n_sat = config.get("n_sat", self.num_agents)
        # File di output (layout data/models/<run>/).
        self.csv_path    = os.path.join(str(self.run_dir), "training_log.csv")
        self.html_path   = os.path.join(str(self.run_dir), "training_curves.html")
        self._state_path = os.path.join(self.save_dir, "training_state.pt")
        self._timing = TrainingTiming(self.run_dir, self.device, type(self.envs).__name__)
        # Frequenze (da Training Settings di secors.json5, default sensati)
        self._plot_every               = getattr(self.all_args, "plot_every", max(self.log_interval, 1))
        self._print_distribution_every = getattr(self.all_args, "print_distribution_every", max(self.log_interval, 1))

        # RESUME: base_runner ha già ricaricato actor.pt/critic.pt (se model_dir!=None).
        self.start_episode = 0
        resuming = getattr(self.all_args, "secors_resume", False) and os.path.isfile(self._state_path)
        dai_pesi = getattr(self.all_args, "secors_ripresa_pesi", False)
        if resuming:
            self.start_episode = self._resume_state()
            print(f"[SECORS] RESUME: continuo da episodio {self.start_episode} "
                  f"(stato: {self._state_path})", flush=True)
        elif dai_pesi:
            # Il CSV non viene riscritto: le righe si accodano a quelle gia' registrate.
            self.start_episode = int(getattr(self.all_args, "secors_start_episode", 0))
            print(f"[SECORS] RIPRESA DAI PESI: continuo da episodio {self.start_episode}; "
                  f"optimizer e ValueNorm ripartono da zero", flush=True)
        else:
            self._init_csv_header()

    def _init_csv_header(self):
        """Nuova run: scrive l'header del CSV (tronca eventuale precedente)."""
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "reward", "C_T", "C_E", "actor_loss", "critic_loss", "SAT", "SAS", "GND"]
            )

    def _resume_state(self) -> int:
        """Ripristina optimizer + value_normalizer + episodio da training_state.pt."""
        st = torch.load(self._state_path, map_location=self.device)
        if getattr(self.trainer, "_use_valuenorm", False) and not st.get("value_normalizer"):
            raise RuntimeError(
                "Il checkpoint non contiene le statistiche ValueNorm: la ripresa "
                "non puo' ricostruire la scala dei valori del critic. Conservare "
                "questo modello per l'inferenza e avviare un nuovo training in "
                "una cartella diversa, oppure usare un checkpoint con statistiche complete.")
        self.trainer.policy.actor_optimizer.load_state_dict(st["actor_optimizer"])
        self.trainer.policy.critic_optimizer.load_state_dict(st["critic_optimizer"])
        if getattr(self.trainer, "_use_valuenorm", False) and st.get("value_normalizer") is not None:
            self.trainer.value_normalizer.load_state_dict(st["value_normalizer"])
        return int(st.get("episode", 0)) + 1

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def save(self, episode: int):
        """Salva actor.pt/critic.pt (base_runner) + stato per il resume
        (optimizer, valuenorm, episodio)."""
        super().save()
        salva_atomico({
            "episode":          episode,
            "actor_optimizer":  self.trainer.policy.actor_optimizer.state_dict(),
            "critic_optimizer": self.trainer.policy.critic_optimizer.state_dict(),
            "value_normalizer": (self.trainer.value_normalizer.state_dict()
                                 if getattr(self.trainer, "_use_valuenorm", False) else None),
        }, self._state_path)
        run_lock.aggiorna(os.path.dirname(self.save_dir))

    # ------------------------------------------------------------------ #
    # Loop Algorithm 1
    # ------------------------------------------------------------------ #
    def run(self):
        self.warmup()

        self._session_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(self.start_episode, episodes):
            self._timing.start_episode()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            act_counts = {"SAT": 0, "SAS": 0, "GND": 0}
            for step in range(self.episode_length):
                # Sample actions (r.13-15)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self._timing.call(
                    "collect", self.collect, step)
                self._count_actions(actions, act_counts)

                # Step nell'env: azione congiunta + EFT + reward (r.17-19)
                obs, share_obs, rewards, dones, infos, available_actions = self._timing.call(
                    "env_step", self.envs.step, actions)

                self._timing.call("insert", self.insert, (obs, share_obs, rewards, dones, infos, available_actions,
                             values, actions, action_log_probs, rnn_states, rnn_states_critic))  # r.20

            # GAE + return (r.23) e update PPO (r.28-35)
            self._timing.call("returns", self.compute)
            train_infos = self._timing.call("ppo", self.train)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self._timing.call("checkpoint", self.save, episode)

            # Metriche dell'iterazione -> riga su training_log.csv
            metrics = self._episode_metrics(infos, train_infos)
            self._write_csv_row(episode, metrics, act_counts)

            if episode % self.log_interval == 0:
                train_infos["average_episode_rewards"] = metrics["reward"]
                train_infos["C_T"] = metrics["c_t"]
                train_infos["C_E"] = metrics["c_e"]
                self._report(episode, episodes, total_num_steps, metrics)
                self.log_train(train_infos, total_num_steps)

            # Distribuzione azioni SAT/SAS/GND (frequenza = print_distribution_every)
            if episode % self._print_distribution_every == 0:
                self._report_distribution(act_counts)

            # Rigenera training_curves.html (frequenza = plot_every)
            if episode % self._plot_every == 0:
                self._timing.call("plot", generate_training_html, self.csv_path, self.html_path)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
            self._timing.finish_episode(episode)

        # Plot finale
        generate_training_html(self.csv_path, self.html_path)

    def warmup(self):
        """Inizializza il buffer al passo 0 con la prima osservazione (Algorithm 1, r.8)."""
        obs, share_obs, available_actions = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs
        # Slot [0] = istante iniziale dell'episodio (gli step 1..L lo riempiono dopo).
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        """Campiona l'azione congiunta per il passo `step` (Algorithm 1, r.13-15)."""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),        # stato globale s_t -> critic
                np.concatenate(self.buffer.obs[step]),              # obs locale o_t   -> actor
                np.concatenate(self.buffer.rnn_states[step]),       # memoria RNN actor
                np.concatenate(self.buffer.rnn_states_critic[step]),# memoria RNN critic
                np.concatenate(self.buffer.masks[step]),            # 1=attivo / 0=episodio finito
                np.concatenate(self.buffer.available_actions[step]),# maschera is_assignable
            )
        # Ri-separa per thread di rollout: (n_thread·n_agent, ...) -> (n_thread, n_agent, ...)
        values            = np.array(np.split(_t2n(value),            self.n_rollout_threads))
        actions           = np.array(np.split(_t2n(action),           self.n_rollout_threads))
        action_log_probs  = np.array(np.split(_t2n(action_log_prob),  self.n_rollout_threads))
        rnn_states        = np.array(np.split(_t2n(rnn_state),        self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Scrive il risultato dello step nel buffer (Algorithm 1, r.20)."""
        (obs, share_obs, rewards, dones, infos, available_actions,
         values, actions, action_log_probs, rnn_states, rnn_states_critic) = data

        dones_env = np.all(dones, axis=1)   # True per i thread il cui episodio è concluso

        # Reset memoria RNN (actor e critic) sui thread terminati.
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        # mask: 1 = transizione valida, 0 = subito dopo un done (interrompe il bootstrap GAE).
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions,
                           action_log_probs, values, rewards, masks,
                           available_actions=available_actions)

    # ------------------------------------------------------------------ #
    # Metriche / logging / reporting
    # ------------------------------------------------------------------ #
    def _count_actions(self, actions, act_counts):
        """Accumula la distribuzione device-type."""
        flat = np.asarray(actions).reshape(-1)
        act_counts["SAT"] += int((flat <  self.n_sat).sum())
        act_counts["SAS"] += int((flat == self.n_sat).sum())
        act_counts["GND"] += int((flat == self.n_sat + 1).sum())

    def _episode_metrics(self, infos, train_infos) -> dict:
        """Estrae le metriche dell'iterazione da infos (env) + train_infos (trainer)."""
        c_t = float(np.mean([info[0]["C_T"] for info in infos if "C_T" in info[0]])) if len(infos) else 0.0
        c_e = float(np.mean([info[0]["C_E"] for info in infos if "C_E" in info[0]])) if len(infos) else 0.0
        return {
            "reward": float(np.mean(self.buffer.rewards) * self.episode_length),
            "c_t":    c_t,
            "c_e":    c_e,
            "a_loss": float(train_infos.get("policy_loss", 0.0)),
            "c_loss": float(train_infos.get("value_loss", 0.0)),
        }

    def _write_csv_row(self, episode, m, act_counts):
        """Appende la riga dell'episodio su training_log.csv (per plot_log.py)."""
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([episode, m["reward"], m["c_t"], m["c_e"], m["a_loss"], m["c_loss"],
                                    act_counts["SAT"], act_counts["SAS"], act_counts["GND"]])

    def _report(self, episode, episodes, total_num_steps, m):
        """Stampa metriche dell'iterazione + ETA/FPS (sui soli episodi di questa sessione)."""
        elapsed       = time.time() - self._session_start
        done_episodes = episode - self.start_episode + 1
        sec_per_ep    = elapsed / max(done_episodes, 1)
        eta_sec       = max(episodes - episode - 1, 0) * sec_per_ep
        # Formato manuale: ore senza limite.
        hh, rem       = divmod(int(eta_sec), 3600)
        mm, ss        = divmod(rem, 60)
        eta_str       = f"{hh:02d}:{mm:02d}:{ss:02d}"
        fps           = int(total_num_steps / elapsed) if elapsed > 0 else 0

        print(f"\n[SECORS] ep {episode}/{episodes} | steps {total_num_steps}/{self.num_env_steps} "
              f"| FPS {fps} | ETA {eta_str}")
        print(f"  reward(ep)={m['reward']:.2f} | C_T={m['c_t']:.2f} ms | C_E={m['c_e']:.2f} Wms "
              f"| actor_loss={m['a_loss']:.4f} | critic_loss={m['c_loss']:.4f}", flush=True)

    def _report_distribution(self, act_counts):
        """Stampa la distribuzione delle azioni per tipo di device."""
        print(f"  [azioni] SAT/SAS/GND = "
              f"{act_counts['SAT']}/{act_counts['SAS']}/{act_counts['GND']}", flush=True)
