__author__ = 'sasha'

import fly_trajectory_analyzer as fta

class osmotropotaxis_test:
    def __init__(self, exp_mdata, data_loader):
        self.fly_traj_analyzer = fta.fly_trajectory_analyzer(exp_mdata, data_loader.trial_data_all)

    def show_analysis(self):
        # self.fly_traj_analyzer.show_classifier('AgglomerativeClustering')

        self.fly_traj_analyzer.show_avg_velocity_response()


