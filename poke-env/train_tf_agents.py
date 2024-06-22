import numpy as np
import reverb
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import (
    tf_py_environment,
    py_environment,
    utils as env_utils,
    wrappers as env_wrappers,
)
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory, time_step
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.utils import common


from poke_env.player import EnvPlayer as PokeEnvPlayer


class PokeTfAgentEnv(py_environment.PyEnvironment):
    """
    References:
    [1] Create your own Python Environment.
        https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#creating_your_own_python_environment
    """

    def __init__(self, poke_env: PokeEnvPlayer):
        self.poke_env = poke_env
        
        box = self.poke_env.describe_embedding()
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=box.low.shape, dtype=np.float32, minimum=box.low, maximum=box.high, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=21, name='action')
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        state, info = self.poke_env.reset()
        return time_step.restart(state)

    def _step(self, action):

        if self.poke_env.current_battle.finished:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        observation, reward, terminated, truncated, _ = self.poke_env.step(action[0])
        if terminated:
            return time_step.termination(observation, reward)
        else:
            return time_step.transition(observation, reward)


class TfAgentDQNTrainer:
    """
    References:
    [1] Train a deep Q network. https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
    """

    def __init__(
        self, 
        train_env: tf_py_environment.TFPyEnvironment,
        eval_env: tf_py_environment.TFPyEnvironment,
        learning_rate: float = 1e-3,
        num_iterations: int = 20000,
        num_eval_episodes: int = 10,
        eval_interval: int = 1000,
        log_interval: int = 200,
        collect_steps_per_iteration: int = 1,
        batch_size: int = 64,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.batch_size = batch_size

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=self.q_net(),
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()

        self.table_name = 'uniform_table'

        self.replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        
        self.replay_buffer_signature = tensor_spec.add_outer_dim(
            self.replay_buffer_signature)

        self.table = reverb.Table(
            self.table_name,
            max_size=self.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=self.replay_buffer_signature)

        self.reverb_server = reverb.Server([self.table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=self.table_name,
            sequence_length=2,
            local_server=self.reverb_server)
        
        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            self.table_name,
            sequence_length=2)
        
        self.dataset = self.datasetreplay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        self.iterator = iter(self.dataset)


    def dense_layer(self, num_units: int):
        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal'
        )
    )

    def q_net(self):
        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(self.train_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        return q_net
    
    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
        
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    def train(self):

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step.
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        returns = [avg_return]

        # Reset the environment.
        time_step = self.train_env.reset()

        # Create a driver to collect experience.
        collect_driver = py_driver.PyDriver(
            self.train_env,
            py_tf_eager_policy.PyTFEagerPolicy(self.agent.collect_policy, use_tf_function=True),
            [self.rb_observer],
            max_steps=self.collect_steps_per_iteration)

        for _ in range(self.num_iterations):

            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(self.iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)


from src.envs.simple_rl import SimpleRLEnv
from src.players.max_damage import MaxDamagePlayer


async def main():

    battle_format = 'gen8randombattle'
    opponent = MaxDamagePlayer(battle_format=battle_format)
    poke_env = SimpleRLEnv(battle_format=battle_format, opponent=opponent, start_challenging=True)
    # environment = PokeTfAgentEnv(poke_env=poke_env)
    # env_utils.validate_py_environment(environment, episodes=5)

    train_py_env = PokeTfAgentEnv(poke_env=poke_env)
    eval_py_env = PokeTfAgentEnv(poke_env=poke_env)

    train_env = tf_py_environment.TFPyEnvironment(
        train_py_env,
        # env_wrappers.FlattenActionWrapper(train_py_env, np.int8), isolation=True,
    )
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    trainer = TfAgentDQNTrainer(
        train_env=train_env,
        eval_env=eval_env,
    )
    trainer.train()

if __name__ == "__main__":

    import os
    # Keep using keras-2 (tf-keras) rather than keras-3 (keras).
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import asyncio
    asyncio.get_event_loop().run_until_complete(main())
    