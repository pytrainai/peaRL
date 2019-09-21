# Target main code for training any agent

import yaml
import os
from datetime import date
from factory import factory as f
from utils.misc import *

# Configs ***************

#get all config paths
config_paths = get_config("factory/default_paths.yaml") # fill with experiment's config folder containing all config yaml files
config_env, config_agent, config_memory, config_process, config_trainer  = f.MakeConfig(config_paths)


# Experiment ************

# make process components
env, env_dims = f.MakeEnvironment(config_env)
agent, device, used_device = f.MakeAgent(config_agent, env_dims)
memory = f.MakeMemory(config_memory)

# make process
process = f.MakeProcess(agent, env, memory, device, config_process)

# make trainer
trainer = f.MakeTrainer(process, config_trainer)

# train
scores = trainer.train()

# plot_results
#trainer.plot_learning_curve()

# save results
#trainer.save_results()

# join and save experiment configs
#date = {'date': date.today().strftime("%d-%m-%y-%s")}
#experiment = {**date, **config_env, **env_params, **config_agent, **config_memory, **used_device, **config_process, **config_trainer}
#save_config(experiment)

# evaluate trained agent
#process.evaluate()

# visualize trained agent
#process.enjoy()
