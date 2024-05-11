import tempfile
import torch
import uuid
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data.tensor_specs import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.data.replay_buffers import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.trainers import Trainer, LogReward

from src.envs.simple_rl import SimpleRLEnv
from src.players import MaxDamagePlayer, TrainedTorchRLPlayer
from src.trainers.torch.model import DQN
from src.trainers.torchrl.config import CONFIG_DQN as CONFIG


class TorchRlEnv(EnvBase):
    """
    
    References:
    [1] Pendulum: Writing your environment and transforms with TorchRL. 
        https://pytorch.org/tutorials/advanced/pendulum.html
    """

    def __init__(self, poke_env: SimpleRLEnv, device: str = 'cpu'):
        super().__init__(device=device, batch_size=[])
        self.poke_env = poke_env
        box = self.poke_env.describe_embedding()
        self.observation_spec = CompositeSpec({
            'state': BoundedTensorSpec(low=box.low, high=box.high, shape=(10,), dtype=torch.float32, device=self.device),
        })
        self.state_spec = self.observation_spec.clone()
        # self.action_spec = BoundedTensorSpec(low=-1, high=21, shape=(1,), dtype=torch.int8, device=self.device)
        self.action_spec = DiscreteTensorSpec(1)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), device=self.device)

    def _reset(self, tensordict: TensorDict):
        # print(f'\n_reset = {tensordict}\n')
        state, info = self.poke_env.reset()
        state = TensorDict({'state': state}, batch_size=self.batch_size, device=self.device)
        # print(f'\n_reset = {state}\n')
        return state

    def _step(self, tensordict: TensorDict):
        # print(f'\n_step = {tensordict}\n')
        observation, reward, terminated, truncated, _ = self.poke_env.step(tensordict['action'].item())
        out = TensorDict({
            'state': observation,
            'reward': reward,
            'done': terminated,
            # 'truncated': truncated,
        }, batch_size=tensordict.shape, device=self.device)
        return out

    def _set_seed(self):
        pass


async def main():

    # https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_atari.py

    battle_format = 'gen8randombattle'
    device = 'cpu'

    opponent = MaxDamagePlayer(battle_format=battle_format)

    simple_env = SimpleRLEnv(
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
    )

    env = TorchRlEnv(poke_env=simple_env)

    # check_env_specs(env)
    # train_env.close()
    # env.close()

    # https://pytorch.org/rl/stable/tutorials/coding_dqn.html
    
    # Correct for frame_skip
    frame_skip = 4
    total_frames = CONFIG['collector']['total_frames'] // frame_skip
    frames_per_batch = CONFIG['collector']['frames_per_batch'] // frame_skip
    init_random_frames = CONFIG['collector']['init_random_frames'] // frame_skip

    # Make the components
    n_actions = simple_env.action_space.n      # Get number of actions from gym action space
    state, info = simple_env.reset()           # Get the number of state observations
    n_observations = len(state)
    model = QValueActor(
        module=DQN(n_observations, n_actions).to(device),
        spec=CompositeSpec(action=env.specs["input_spec", "full_action_spec", "action"]),
        in_keys=["state"],
    )
    greedy_module = EGreedyModule(spec=model.spec, **CONFIG['greedy_module'])
    policy = TensorDictSequential(model, greedy_module).to(device)

    # Create the collector
    # https://pytorch.org/rl/stable/reference/collectors.html
    # https://pytorch.org/rl/stable/reference/generated/torchrl.collectors.SyncDataCollector.html
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy, 
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=init_random_frames,
    )

    # Create the loss module
    loss_module = DQNLoss(value_network=model, loss_function="l2", delay_value=True)
    # loss_module.set_keys(done='end-of-life', terminated='end-of-life')
    loss_module.make_value_estimator(gamma=CONFIG['loss']['gamma'])

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), **CONFIG['optimizer'])

    # Create the logger
    # exp_name = f"dqn_exp_{uuid.uuid1()}"
    # tmpdir = tempfile.TemporaryDirectory()
    # logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir)

    exp_name = generate_exp_name(**CONFIG['generate_exp_name'])
    logger = get_logger(experiment_name=exp_name, **CONFIG['get_logger'])
    
    # https://pytorch.org/rl/stable/tutorials/coding_dqn.html
    trainer = Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=1,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        # save_trainer_file='test/',
        log_interval=50,
    )
    
    log_reward = LogReward(log_pbar=True)
    log_reward.register(trainer)

    trainer.train()
    trainer.save_trainer(True)

    simple_env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
    