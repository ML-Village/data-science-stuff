
EXPERIMENT_NAME = 'PokeEnv'

MODEL_NAME = 'DQN'

CONFIG_DQN = {
    'buffer': {
        'buffer_size': 1_000_000,
        'batch_size': 32,
        'scratch_dir': None,
    },
    'collector': {
        'total_frames': 100_000,
        # 'total_frames': 16_000,
        'frames_per_batch': 16,         # the number of frames delivered at each iteration over the collector
        'init_random_frames': 200_000,  # number of random steps (steps where env.rand_step() is being called)
    },
    'loss': {
        'gamma': 0.99,
        'hard_update_freq': 10_000,
    },
    'optimizer': {
        'lr': 0.00025,
    },
    'greedy_module': {
        'annealing_num_steps': 4_000_000,
        'eps_init': 1.0,
        'eps_end': 0.01,
    },
    'generate_exp_name': {
        'model_name': MODEL_NAME,
        'experiment_name': EXPERIMENT_NAME,
    },
    'get_logger': {
        'logger_type': 'csv',
        'logger_name': 'dqn',
    },
}
