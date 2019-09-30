from pearl.runner.runner import Runner

# instantiate runner
experiment_ID = 'ddpg_linear_pendulum_5'
config_file = 'ddpg_linear_config.yaml'
runner = Runner(config_file, experiment_ID)

# run experiment
runner.run_experiment()