__author__ = 'sasha'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import fly_trajectory_utils as ftu
import fly_trajectory_classifier as ftc

import TrialData

class fly_trajectory_analyzer:
    def __init__(self, exp_meta, trial_data_all):
        self.exp_meta = exp_meta
        self.trial_data_all = trial_data_all
        self.fly_traj_utils = ftu.fly_trajectory_utils( self.exp_meta)
        self.fly_traj_utils.rotate( self.trial_data_all )
        self.fly_traj_utils.calculate_velocity( self.trial_data_all )
        self.bdata_griddy = self.fly_traj_utils.griddify( self.trial_data_all )
        self.classifier = ftc.fly_trajectory_classifier(self.exp_meta, self.bdata_griddy)

    def show_classifier(self, type):
        self.classifier.classify(type)

    def show_avg_velocity_response(self):

        fig = plt.figure(figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        leftTrialIdx = TrialData.TrialData.getTrialIndexForName('Left_Odor')
        rightTrialIdx = TrialData.TrialData.getTrialIndexForName('Right_Odor')

        ax1 = fig.add_subplot(1,2,1)
        mean_of_trials_fwd_left = np.mean(self.bdata_griddy[leftTrialIdx][:,:,self.VEL_Y],0)
        mean_of_trials_fwd_right = np.mean(self.bdata_griddy[rightTrialIdx][:,:,self.VEL_Y],0)
        ax1.plot( self.time_grid, mean_of_trials_fwd_left, color='r', label='Left Odor' )
        ax1.plot( self.time_grid, mean_of_trials_fwd_right, color='b', label='Right Odor' )
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Fwd vel (au/s)')
        ax1.set_title( 'Avg fwd vel all trials' )
        ax1.set_xlim((0, 6.5))
        ylim = (-400, 2000)
        ax1.set_ylim(ylim)
        plt.legend(frameon=False)
        ax1.grid()

        p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                ylim[1]-ylim[0], linewidth=0, color='wheat' )
        ax1.add_patch(p)

        ax1 = fig.add_subplot(1,2,2)
        mean_of_trials_lat_left = np.mean(self.bdata_griddy[leftTrialIdx][:,:,self.VEL_X],0)
        mean_of_trials_lat_right = np.mean(self.bdata_griddy[rightTrialIdx][:,:,self.VEL_X],0)
        ax1.plot( self.time_grid, mean_of_trials_lat_left, color='r', label='Left Odor' )
        ax1.plot( self.time_grid, mean_of_trials_lat_right, color='b', label='Right Odor' )
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Lateral vel (au/s)')
        ax1.set_title( 'Avg lat vel all trials ' )
        ax1.set_xlim((0, 6.5))
        ylim = (-400, 800)
        ax1.set_ylim(ylim)
        ax1.grid()

        p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                ylim[1]-ylim[0], linewidth=0, color='wheat')
        ax1.add_patch(p)

        plt.legend(frameon=False)

        plt.show()
        filepath = self.exp_meta.analysisPath + '/test_file'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)
