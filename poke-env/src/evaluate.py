import torch.nn as nn
from poke_env.player import RandomPlayer, cross_evaluate
from tabulate import tabulate

from .envs.simple_rl import SimpleRLEnv
from .players import MaxDamagePlayer, TrainedTorchRLPlayer


async def evaluate_trained_model(
    env: SimpleRLEnv,
    model: nn.Module,
    device: str = 'cpu',
    n_challenges: int = 20,
):
    """ Play trained model against RandomPlayer and MaxDamagePlayer
    """
    # Create three random players
    battle_format = env.agent._format
    players = [
        RandomPlayer(battle_format=battle_format),
        MaxDamagePlayer(battle_format=battle_format),
        TrainedTorchRLPlayer(
            battle_format=battle_format,
            env=env,
            model=model,
            device=device,
        ),
    ]

    # Cross evaluate players: each player plays 20 games against every other player
    cross_evaluation = await cross_evaluate(players, n_challenges=n_challenges)

    # Prepare results for display
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Display results
    print(tabulate(table))
