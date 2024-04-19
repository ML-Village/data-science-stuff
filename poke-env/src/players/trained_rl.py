import torch
import torch.nn as nn
from poke_env.environment import AbstractBattle
from poke_env.player import Player, EnvPlayer


class TrainedTorchRLPlayer(Player):
    """ Player that choose moves based on predictions from a trained RL model.

    References:
    [1] Need help with the method of connecting self RL agent to showdown.
        https://github.com/hsahovic/poke-env/issues/378
    """

    def __init__(self, env: EnvPlayer, model: nn.Module, device: str, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        self.env = env
        self.model = model
        self.device = device

    def choose_move(self, battle: AbstractBattle):
        state = self.env.embed_battle(battle)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.model(state).max(1).indices.view(1, 1)
        return self.env.action_to_move(action, battle)
    