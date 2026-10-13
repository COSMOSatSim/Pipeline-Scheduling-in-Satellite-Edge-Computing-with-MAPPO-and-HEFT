import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space


class Actor(nn.Module):
    """
    Rete Actor (policy π) di MAPPO. Dato l'osservazione, produce l'azione.

    Composizione: base (MLPBase) -> rnn (GRU) -> act (ACTLayer).
    :param args: (argparse.Namespace) iperparametri del modello (hidden_size, recurrent_N, gain, ...).
    :param obs_space: (gym.Space) spazio delle osservazioni.
    :param action_space: (gym.Space) spazio delle azioni.
    :param device: (torch.device) device su cui girare (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Calcola l'azione dagli input (usata mentre si "gioca": raccolta dati in training e inferenza).
        :param obs: (np.ndarray / torch.Tensor) osservazione in ingresso alla rete.
        :param rnn_states: (np.ndarray / torch.Tensor) stato nascosto (memoria) della GRU.
        :param masks: (np.ndarray / torch.Tensor) maschera: se azzerare la memoria (a inizio episodio).
        :param available_actions: (np.ndarray / torch.Tensor) quali azioni sono ammesse per l'agente
                                                              (se None, tutte ammesse).
        :param deterministic: (bool) True -> moda/argmax (inferenza); False -> campiona (training, esplora).

        :return actions: (torch.Tensor) azione scelta.
        :return action_log_probs: (torch.Tensor) log-probabilità dell'azione scelta.
        :return rnn_states: (torch.Tensor) stato nascosto GRU aggiornato.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Ricalcola log-probabilità ed entropia per azioni GIÀ fatte (usata solo nell'update PPO).
        :param obs: (torch.Tensor) osservazione in ingresso alla rete.
        :param action: (torch.Tensor) azioni di cui valutare log-probabilità ed entropia.
        :param rnn_states: (torch.Tensor) stato nascosto (memoria) della GRU.
        :param masks: (torch.Tensor) maschera: se azzerare la memoria.
        :param available_actions: (torch.Tensor) quali azioni sono ammesse per l'agente
                                                              (se None, tutte ammesse).
        :param active_masks: (torch.Tensor) se un agente è attivo o "morto" (da noi sempre attivi).

        :return action_log_probs: (torch.Tensor) log-probabilità delle azioni passate.
        :return dist_entropy: (torch.Tensor) entropia della distribuzione (serve per l'entropy bonus).
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy
