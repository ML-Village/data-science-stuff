from poke_env.environment import AbstractBattle
from poke_env.player import Player


class MaxDamagePlayer(Player):
    """ Adapted from poke-env documentation.

    References:
    [1] Creating a simple max damage player.
        https://poke-env.readthedocs.io/en/stable/examples/max_damage_player.html
    """
    
    def choose_move(self, battle: AbstractBattle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

