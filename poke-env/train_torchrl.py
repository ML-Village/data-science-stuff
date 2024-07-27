from datetime import datetime
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.tensor_specs import (
    BoundedTensorSpec, 
    CompositeSpec, 
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers import get_logger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
)

from src.envs.simple_rl import SimpleRLEnv
from src.evaluate import evaluate_trained_model
from src.players import MaxDamagePlayer
from src.trainers.torch.model import DQN


DEVICE = 'cpu'
BATCH_SIZE = 256
STORAGE_MAX_SIZE = 12_800

CONFIG = {
    'collector': {
        'total_frames': -1,           # if -1, endless collector
        'max_frames_per_traj': -1,    # once a trajector reaches this no of steps, environment is reset
        'frames_per_batch': 32,       # the number of frames delivered at each iteration over the collector
        'init_random_frames': None,   # number of random steps (steps where env.rand_step() is being called)
        'device': DEVICE,
        'storing_device': DEVICE,
    },
    'loss': {
        'gamma': 0.99,
        'hard_update_freq': 10_000,
    },
    'optimizer': {
        'lr': 0.00025,
    },
    'greedy_module': {
        'eps_init': 1.0,
        'eps_end': 0.01,
    },
    'get_logger': {
        'logger_type': 'tensorboard',
        'logger_name': 'runs', # log path
    },
    'trainer': {
        'frame_skip': 1,
        'log_interval': 1,           # how often values should be logged, in frame count.
        'optim_steps_per_batch': 1,
        'progress_bar': True,
        'total_frames': 256,
    },
}


class TorchRlEnv(EnvBase):
    """
    
    References:
    [1] Pendulum: Writing your environment and transforms with TorchRL. 
        https://pytorch.org/tutorials/advanced/pendulum.html
    """

    def __init__(self, poke_env: SimpleRLEnv, device: str = 'cpu'):
        super().__init__(device=device, batch_size=[])
        self.poke_env = poke_env
        self.action_space = self.poke_env.action_space_size()
        
        box = self.poke_env.describe_embedding()
        self.observation_spec = CompositeSpec({
            'state': BoundedTensorSpec(
                low=box.low,
                high=box.high,
                shape=box.shape,
                dtype=torch.float32,
                device=self.device,
            ),
        })
        self.state_spec = self.observation_spec.clone()
        self.action_spec = DiscreteTensorSpec(n=self.action_space)
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
            'terminated': terminated,
            'truncated': truncated,
            'done': terminated or truncated,
        }, batch_size=tensordict.shape, device=self.device)
        return out

    def _set_seed(self):
        pass


async def main():

    # https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_atari.py

    battle_format = 'gen8randombattle'

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
    
    # Make the components
    n_observations = simple_env.describe_embedding().shape[0]
    n_actions = simple_env.action_space.n
    model = DQN(n_observations, n_actions).to(DEVICE)
    actor = QValueActor(
        module=model,
        spec=env.action_spec,
        in_keys=['state'],
    )
    greedy_module = EGreedyModule(spec=actor.spec, **CONFIG['greedy_module'])
    policy = TensorDictSequential(actor, greedy_module).to(DEVICE)

    # Create the collector
    # https://pytorch.org/rl/stable/reference/collectors.html
    # https://pytorch.org/rl/stable/reference/generated/torchrl.collectors.SyncDataCollector.html
    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy, 
        **CONFIG['collector'],
    )

    # Create the loss module
    loss_module = DQNLoss(value_network=actor, loss_function="l2", delay_value=True)
    # loss_module.set_keys(done='end-of-life', terminated='end-of-life')
    loss_module.make_value_estimator(gamma=CONFIG['loss']['gamma'])

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), **CONFIG['optimizer'])

    # Create the logger
    exp_name = '_'.join(['DQN', datetime.now().strftime("%Y%m%d%H%M%S")])
    logger = get_logger(experiment_name=exp_name, **CONFIG['get_logger'])
    trainer_file_path = Path(CONFIG['get_logger']['logger_name']) / exp_name
    trainer_file_path.mkdir(exist_ok=True, parents=True)
    
    # https://pytorch.org/rl/stable/tutorials/coding_dqn.html
    trainer = Trainer(
        collector=collector,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        # save_trainer_interval=1,
        # save_trainer_file=trainer_file_path / 'model_file.pt',
        **CONFIG['trainer'],
    )

    # add hooks to store collected data
    buffer = TensorDictReplayBuffer(
        batch_size=BATCH_SIZE,
        storage=LazyMemmapStorage(
            max_size=STORAGE_MAX_SIZE,
            scratch_dir=trainer_file_path / 'buffer',
        ),
    )
    buffer_hook = ReplayBufferTrainer(replay_buffer=buffer, flatten_tensordicts=True)
    buffer_hook.register(trainer)

    # add hooks to update target network
    target_updater = SoftUpdate(loss_module, eps=0.995)
    trainer.register_op('post_optim', target_updater.step)

    # add hooks to log rewards
    log_reward = LogReward(log_pbar=True, logname='reward')
    log_reward.register(trainer)

    trainer.train()
    # trainer.save_trainer(True)
    
    # save trained model
    model = (
        trainer.loss_module
        .value_network.module[0].module
        .to_empty(device=DEVICE)
    )
    torch.save(model.state_dict(), trainer_file_path / 'trained_model.pth')

    # end of training

    ###########################################################################

    # start of evaluation

    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(trainer_file_path / 'trained_model.pth'))
    await evaluate_trained_model(env=simple_env, model=model, device=DEVICE)

    # end of evaluation

    ###########################################################################
    
    simple_env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
    