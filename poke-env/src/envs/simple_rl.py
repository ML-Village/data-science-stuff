import numpy as np
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer
from poke_env.environment import Battle


class SimpleRLEnv(Gen8EnvSinglePlayer):
    """ Adapted from poke-env documentation.
    
    References:
    [1] Reinforcement learning with the OpenAI Gym wrapper.
        https://poke-env.readthedocs.io/en/stable/examples/rl_with_open_ai_gym_wrapper.html
    [2] Type Chart causing a KeyError only containing a type as the error message.
        https://github.com/hsahovic/poke-env/issues/484
    """

    def action_available(self, action: int, battle: Battle) -> np.ndarray:
        if action == -1:
            return False
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return True
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return True
        elif 0 <= action - 16 < len(battle.available_switches):
            return True
        else:
            return False
    

    def calc_reward(self, last_battle: Battle, current_battle: Battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            # print(move)
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    