import os
import sys
import random
import datetime

import traci

from TLCS.testing_simulation import Simulation
from TLCS.utils import import_train_configuration, set_sumo, set_train_path

random.seed(2022)

sumoBinary = "C:/Users/张晶/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/SUMO/sumo-gui.exe"
sumoCmd = [sumoBinary, "-c", "D:/myfile/bishe/DQN -1/TLCS/intersection/sumo_config.sumocfg"]



listLanes = ['N2TL','S2TL','E2TL','W2TL']

def check_sumo():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

def start_sumo(sumo_cmd_str):
    traci.start(sumo_cmd_str)
    for i in range(300):
        traci.simulationStep()

def end_sumo():
    traci.close()

def get_overall_queue_length(listLanes):
    overall_queue_length = 0
    for lane in listLanes:
        overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)
    return overall_queue_length


import matplotlib.pyplot as plt
import os
import pydot

# import graphviz
class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi = dpi

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_' + filename + '.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_' + filename + '_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n" % value)


if __name__ == '__main__':
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Visualization = Visualization(
        path,
        dpi=96
    )
    episode = 0
    timestamp_start = datetime.datetime.datetime.now()

    while episode < config['total_episodes']:
       print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
       epsilon = 1.0 - (episode / config[
             'total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
       simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
       print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
             round(simulation_time + training_time, 1), 's')
       episode += 1

       Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode',
                                 ylabel='Average queue length (vehicles)')

