import torch
from poke_env.player import RandomPlayer, cross_evaluate
from tabulate import tabulate

from src.envs.simple_rl import SimpleRLEnv
from src.players import MaxDamagePlayer, TrainedTorchRLPlayer
from src.trainers.torch import Transition, ReplayMemory, DQN, TorchTrainer


battle_format = "gen8randombattle"


async def main():

    # define players
    opponent = MaxDamagePlayer(battle_format=battle_format)

    # define gym environment
    env = SimpleRLEnv(
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
    )

    # training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n      # Get number of actions from gym action space
    state, info = env.reset()           # Get the number of state observations
    n_observations = len(state)
    model = DQN(n_observations, n_actions).to(device)
    memory = ReplayMemory(10000)
    trainer = TorchTrainer(
        model=model,
        env=env,
        memory=memory,
        gamma=0.99,
        lr=1e-4,
        tau=0.005,
        batch_size=128,
        device=device,
    )

    # start training model
    trainer.train_model()

    # Create three random players
    players = [
        RandomPlayer(battle_format=battle_format),
        MaxDamagePlayer(battle_format=battle_format),
        TrainedTorchRLPlayer(battle_format=battle_format, env=env, model=trainer.policy_net, device=device),
    ]

    # Cross evaluate players: each player plays 20 games against every other player
    cross_evaluation = await cross_evaluate(players, n_challenges=20)

    # Prepare results for display
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Display results
    print(tabulate(table))

    env.close()

if __name__ == "__main__":
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
    