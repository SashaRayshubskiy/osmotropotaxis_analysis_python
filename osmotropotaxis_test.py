__author__ = 'sasha'

import fly_trajectory_analyzer as fta
import fly_trajectory_classifier as ftc

class osmotropotaxis_test:
    def __init__(self):
        pass

    def run(self, data_loader, exp_mdata):

        fly_traj_analyzer = fta.fly_trajectory_analyzer(exp_mdata, data_loader.trial_data_all)
        fly_traj_analyzer.show_classifier('AgglomerativeClustering')


