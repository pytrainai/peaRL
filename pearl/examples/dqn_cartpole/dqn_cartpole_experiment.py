from pearl.runner.runner import Runner

# instantiate runner
experiment_ID = 'dqn_linear_cartpole_1'
config_file = 'dqn_linear_config_3.yaml'
runner = Runner(config_file, experiment_ID)

# run experiment
runner.run_experiment()