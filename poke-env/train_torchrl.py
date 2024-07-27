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
from torchrl.envs import (
    Compose,
    EnvBase,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.transforms import (
    ActionMask,
    StepCounter,
)
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


BATCH_SIZE = 256
DEVICE = 'cpu'
FRAMES_PER_BATCH = 512      # the number of frames delivered at each iteration over the collector
INIT_RANDOM_FRAMES = 1_024  # number of random steps (steps where env.rand_step() is being called)
OPTIM_STEPS_PER_BATCH = 5
POST_EVAL = True
STORAGE_MAX_SIZE = 12_800
TOTAL_FRAMES = 512_000

# # for testing
# BATCH_SIZE = 25
# FRAMES_PER_BATCH = 25
# INIT_RANDOM_FRAMES = 0
# OPTIM_STEPS_PER_BATCH = 10
# POST_EVAL = False
# TOTAL_FRAMES = 25


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
            'action_mask': OneHotDiscreteTensorSpec(n=self.action_space),
        })
        self.state_spec = self.observation_spec.clone()
        self.action_spec = DiscreteTensorSpec(n=self.action_space)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), device=self.device)

    def _reset(self, tensordict: TensorDict):
        # print(f'\n_reset = {tensordict}\n')
        state, info = self.poke_env.reset()
        state = TensorDict({'state': state}, batch_size=self.batch_size, device=self.device)
        state['action_mask'] = [
            self.poke_env.action_available(i, self.poke_env.current_battle)
            for i in range(self.action_space)
        ]
        return state

    def _step(self, tensordict: TensorDict):
        # print(f'\n_step = {tensordict}\n')
        observation, reward, terminated, truncated, _ = self.poke_env.step(tensordict['action'].item())
        action_mask = [
            self.poke_env.action_available(i, self.poke_env.current_battle)
            for i in range(self.action_space)
        ]

        # scenario with no valid moves >> randomly set 1 true
        if sum(action_mask) == 0:
            print('NO VALID MOVES! RANDOMLY ALLOW ONE')
            action_mask[0] = True
        
        out = TensorDict({
            'state': observation,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'done': terminated or truncated,
            'action_mask': action_mask,
        }, batch_size=tensordict.shape, device=self.device)
        return out

    def _set_seed(self, seed = None):
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

    env = TransformedEnv(
        env=TorchRlEnv(poke_env=simple_env),
        transform=Compose(
            ActionMask(),
            StepCounter(),
        ),
    )

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
        action_mask_key='action_mask',
    )
    greedy_module = EGreedyModule(
        action_mask_key='action_mask',
        annealing_num_steps=TOTAL_FRAMES,
        eps_init=1.0,
        eps_end=0.01,
        spec=actor.spec,
    )
    policy = TensorDictSequential(actor, greedy_module).to(DEVICE)

    # Create the collector
    # https://pytorch.org/rl/stable/reference/collectors.html
    # https://pytorch.org/rl/stable/reference/generated/torchrl.collectors.SyncDataCollector.html
    collector = SyncDataCollector(
        create_env_fn=env,
        device=DEVICE,
        frames_per_batch=FRAMES_PER_BATCH,
        init_random_frames=INIT_RANDOM_FRAMES,
        policy=policy, 
        storing_device=DEVICE,
        total_frames=-1,         # if -1, endless collector
        max_frames_per_traj=-1,  # once a trajector reaches this no of steps, environment is reset
    )

    # Create the loss module
    loss_module = DQNLoss(
        value_network=actor,
        loss_function='l2',
        delay_value=True,
        double_dqn=True,
    )
    loss_module.make_value_estimator(gamma=0.99)

    # Create the optimizer
    optimizer = torch.optim.Adam(
        params=loss_module.parameters(), 
        lr=0.0025,
        weight_decay=0.00001,
        betas=(0.9, 0.999),
    )

    # Create the logger
    exp_name = '_'.join(['DQN', datetime.now().strftime("%Y%m%d%H%M%S")])
    logger_name = 'runs'
    logger = get_logger(
        experiment_name=exp_name,
        logger_type='tensorboard',
        logger_name=logger_name,
    )
    trainer_file_path = Path(logger_name) / exp_name
    trainer_file_path.mkdir(exist_ok=True, parents=True)
    
    # https://pytorch.org/rl/stable/tutorials/coding_dqn.html
    trainer = Trainer(
        collector=collector,
        frame_skip=1,
        log_interval=1,     # how often values should be logged, in frame count.
        logger=logger,
        loss_module=loss_module,
        optim_steps_per_batch=OPTIM_STEPS_PER_BATCH,
        optimizer=optimizer,
        progress_bar=True,
        # save_trainer_file=trainer_file_path / 'model_file.pt',
        # save_trainer_interval=1,
        seed=123,
        total_frames=TOTAL_FRAMES,
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

    if POST_EVAL:
        model = DQN(n_observations, n_actions)
        model.load_state_dict(torch.load(trainer_file_path / 'trained_model.pth'))
        await evaluate_trained_model(env=simple_env, model=model, device=DEVICE, n_challenges=100)

    # end of evaluation

    ###########################################################################
    
    simple_env.close()


if __name__ == "__main__":
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
    