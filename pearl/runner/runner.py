import os
from datetime import date
from pearl.factory import factory as f
from pearl.utils.misc import *

# Runner class:
class Runner:
    """ Class that instantiates an experiment and allows to run and save results """
    def __init__(self, config_file, experiment_name):

        self.experiment_name = experiment_name
        # Configs ***************
        #get all config paths
        self.config_env,        \
        self.config_agent,      \
        self.config_memory,     \
        self.config_process,    \
        self.config_trainer  = f.MakeConfig(config_file)

        # Experiment ************

        # make process components
        self.env, self.env_dims = f.MakeEnvironment(self.config_env)
        self.agent, self.device, self.used_device = f.MakeAgent(self.config_agent, self.env_dims)
        self.memory = f.MakeMemory(self.config_memory)

        # make process
        self.process = f.MakeProcess(self.agent, self.env, self.memory, 
                                     self.device, self.config_process)

        # make trainer
        self.trainer = f.MakeTrainer(self.process, self.config_trainer)

        # join and save experiment configs
        self.date = {'date': date.today().strftime("%d-%m-%y-%s")}
        self.experiment_kwargs = {**self.date, 
                                  **self.config_env, 
                                  **self.env_dims, 
                                  **self.config_agent, 
                                  **self.config_memory, 
                                  **self.used_device, 
                                  **self.config_process, 
                                  **self.config_trainer}
    
    def run_experiment(self,
                      plot_results = True,
                      save_results = False):
        
        # experiment folder
        experiment_folder = 'results/' + self.experiment_name
        
        # asserting folders exists, else make them
        if not os.path.isdir(os.path.join(os.getcwd(), 'results')): os.mkdir('results')
        if not os.path.isdir(os.path.join(os.getcwd(), experiment_folder)): os.mkdir(experiment_folder)

        # train agent
        experiment_scores, experiment_avg_scores = self.trainer.train()

        # Plot Results
        if plot_results:
            self.trainer.plot_agent_learning_curve(experiment_scores, experiment_folder + '/' + self.experiment_name + '_plot.png')

        # Save results
        if save_results:
            self.trainer.save_results()

        # evaluate trained agent
        #process.evaluate()

        # visualize trained agent
        #process.enjoy()
