__author__ = 'sasha'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import fly_trajectory_utils as ftu
import fly_trajectory_classifier as ftc
import fly_trajectory_griddy as ftg
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

import TrialData

class fly_trajectory_analyzer:
    def __init__(self, exp_meta, trial_data_all):
        self.exp_meta = exp_meta
        self.trial_data_all = trial_data_all
        self.fly_traj_utils = ftu.fly_trajectory_utils( self.exp_meta)
        self.fly_traj_utils.rotate( self.trial_data_all )
        self.fly_traj_utils.calculate_velocity( self.trial_data_all )


        self.FWD_VELOCITY_THRESHOLD = 1000
        self.trial_data_vel_filtered = self.fly_traj_utils.filter_fwd_velocity(self.trial_data_all, self.FWD_VELOCITY_THRESHOLD, self.exp_meta.preStimTime, self.exp_meta.stimTime)

        self.griddy = ftg.fly_trajectory_griddy(exp_meta, self.trial_data_vel_filtered)
        self.classifier = ftc.fly_trajectory_classifier(self.exp_meta, self.griddy)

    def show_classifier(self, type, N_CLUSTERS):
        self.classifier.classify(type, N_CLUSTERS)

    def show_PCA_2D_scatter_analysis(self):
        bdata_griddy = self.griddy.get_data()
        BEGIN_TIME = 2.9
        END_TIME = 4.5

        BEGIN_TIME_FRAME = BEGIN_TIME*self.griddy.TIME_GRID_SPACING
        END_TIME_FRAME = END_TIME*self.griddy.TIME_GRID_SPACING


        fig = plt.figure(figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        lagend_helper = {}
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        for trialListIdx, trialList in enumerate(self.trial_data_vel_filtered):
            cur_data = bdata_griddy[trialListIdx]
            X = cur_data[:,BEGIN_TIME_FRAME:END_TIME_FRAME,self.griddy.VEL_X]
            print X.shape
            N_COMPONENTS = 4
            pca = PCA(n_components=N_COMPONENTS)
            pca.fit(X)
            X_tran = pca.transform(X)

            trialName = TrialData.TrialData.getTrialNameForIdx(trialListIdx)

            axs = fig.add_subplot(1,3,trialListIdx+1, projection='3d')
            axs.scatter(X_tran[:,0], X_tran[:,1], X_tran[:,2] )
            axs.set_xlabel('PC1')
            axs.set_ylabel('PC2')
            axs.set_zlabel('PC3')
            axs.set_title(trialName)

        plt.show()
        filepath = self.exp_meta.analysisPath + '/behavior_3D_scatter_PCA'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)



    def show_PCA_analysis(self):

        bdata_griddy = self.griddy.get_data()
        BEGIN_TIME = 2.9
        END_TIME = 4.5

        BEGIN_TIME_FRAME = BEGIN_TIME*self.griddy.TIME_GRID_SPACING
        END_TIME_FRAME = END_TIME*self.griddy.TIME_GRID_SPACING

        N_COMPONENTS = 4
        pca = PCA(n_components=N_COMPONENTS)

        fig, axs = plt.subplots(N_COMPONENTS, 3, sharex=True, sharey=True, figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        lagend_helper = {}

        for trialListIdx, trialList in enumerate(self.trial_data_vel_filtered):
            cur_data = bdata_griddy[trialListIdx]
            X = cur_data[:,BEGIN_TIME_FRAME:END_TIME_FRAME,self.griddy.VEL_X]
            print X.shape
            pca.fit(X)
            components = pca.components_

            trialName = TrialData.TrialData.getTrialNameForIdx(trialListIdx)

            i=0
            while i < N_COMPONENTS:
                lagend_helper[i] = axs[ i, trialListIdx ].plot( self.griddy.get_time_grid()[BEGIN_TIME_FRAME:END_TIME_FRAME], components[i], label='Component ' + str(i) )

                if i == N_COMPONENTS-1:
                    axs[ i, trialListIdx ].set_xlabel('Time (s)')

                if trialListIdx == len(self.trial_data_vel_filtered):
                    axs[ i, trialListIdx ].set_ylabel('Lat Vel (au)')

                if i == 0:
                    axs[ i, trialListIdx ].set_title(trialName + ' component ' + str(i))
                else:
                    axs[ i, trialListIdx ].set_title('Component ' + str(i))

                ylim = (-0.5, 0.5)
                axs[ i, trialListIdx ].set_ylim(ylim)
                axs[ i, trialListIdx ].set_xlim((BEGIN_TIME, END_TIME))

                p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                ylim[1]-ylim[0], linewidth=0, color='wheat' )
                axs[ i, trialListIdx ].add_patch(p)
                i = i + 1


        plt.show()
        filepath = self.exp_meta.analysisPath + '/behavior_PCA'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)


    def show_trajectories(self):

        N_CLUSTERS = 4
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        # Clustering epoch
        BEGIN_TIME = 2.9
        END_TIME = 4.5

        bdata_griddy = self.griddy.get_data()


        # Only show velocity filtered data
        for trialListIdx, trialList in enumerate(self.trial_data_vel_filtered):

            # This is for fly_health_94, Both_Odor, Left_Odor, Right_Odor
            correct_labels = [ 0, 1, 0 ]

            labels, n_clusters = self.classifier.classify_core(N_CLUSTERS, 'AgglomerativeClustering', bdata_griddy[trialListIdx], BEGIN_TIME, END_TIME )

            plot_helper = {}
            for trialIdx, trial in enumerate(trialList):

                t = trial.t
                dx = trial.dx_rot
                dy = trial.dy_rot
                tz = t-t[0]

                stim_t = np.nonzero((tz >= self.exp_meta.preStimTime) & (tz < self.exp_meta.preStimTime+self.exp_meta.stimTime))

                traj_x, traj_y = self.fly_traj_utils.calculate_trial_trajectory(dx,dy)

                cur_clr = None
                trial_label = None

                cur_clr = plt.cm.jet(1.0*labels[trialIdx]/N_CLUSTERS)
                trial_label = 'clust: ' + str(labels[trialIdx])
                cur_x_offset = labels[trialIdx]*2000

                plot_helper[trial_label], = axs[trialListIdx].plot(traj_x-traj_x[stim_t][0]+cur_x_offset, traj_y-traj_y[stim_t][0], color=cur_clr, label=trial_label)
                axs[trialListIdx].hold(True)
                axs[trialListIdx].plot(traj_x[stim_t]-traj_x[stim_t][0]+cur_x_offset, traj_y[stim_t]-traj_y[stim_t][0], marker='x', color=cur_clr)

            trialName = TrialData.TrialData.getTrialNameForIdx(trialListIdx)

            correct_cnt = np.nonzero(labels == correct_labels[trialListIdx])[0].shape[0]
            total_cnt = labels.shape[0]

            title_str = '%s per corr: %.02f (%d/%d)' % (trialName, float(correct_cnt*1.0/total_cnt), correct_cnt, total_cnt)

            axs[trialListIdx].set_title(title_str)
            axs[trialListIdx].set_ylabel('Y distance (au)')
            axs[trialListIdx].set_xlabel('X distance (au)')
            axs[trialListIdx].set_ylim((-1000,8000))
            axs[trialListIdx].set_xlim((-4000,8000))
            # axs[trialListIdx].set_xticklabels(axs[trialListIdx].get_xticklabels(), rotation=45)
            axs[trialListIdx].grid()

            if trialListIdx == 0:
                axs[trialListIdx].legend( plot_helper.itervalues(), plot_helper.iterkeys(), frameon=False)

        plt.show()
        filepath = self.exp_meta.analysisPath + '/trajectory_all_runs'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)

    def show_fwd_and_lat_velocity_bar_plots_for_each_trial(self):

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')
        width = 1.0

        rects_all = []

        for trialListIdx, trialList in enumerate(self.trial_data_all):

            trial_cnt = len(trialList)
            avg_vel_x_l = np.zeros(trial_cnt)
            avg_vel_y_l = np.zeros(trial_cnt)
            std_vel_x_l = np.zeros(trial_cnt)
            std_vel_y_l = np.zeros(trial_cnt)

            for trialIdx, trial in enumerate(trialList):

                t = trial.t

                tz = t-t[0]

                prestim_t = np.nonzero(tz < self.exp_meta.preStimTime)
                stim_t = np.nonzero((tz >= self.exp_meta.preStimTime) & (tz < self.exp_meta.preStimTime+self.exp_meta.stimTime))

                avg_vel_x = np.mean(trial.vel_x[stim_t])
                std_vel_x = np.std(trial.vel_x[stim_t])
                avg_vel_y = np.mean(trial.vel_y[prestim_t])
                std_vel_y = np.std(trial.vel_y[prestim_t])
                avg_vel_x_l[ trialIdx ] = avg_vel_x
                avg_vel_y_l[ trialIdx ] = avg_vel_y
                std_vel_x_l[ trialIdx ] = std_vel_x
                std_vel_y_l[ trialIdx ] = std_vel_y

            #rects1 = axs[trialListIdx].bar(np.arange(trial_cnt), avg_vel_x_l, width=0.35, color='blue', yerr=std_vel_x_l)
            #rects2 = axs[trialListIdx].bar(np.arange(trial_cnt), avg_vel_y_l, width=0.35, color='red', yerr=std_vel_y_l)
            rects2 = axs[trialListIdx].bar(np.arange(trial_cnt)+width, avg_vel_y_l, width, color='red', linewidth=0)
            rects1 = axs[trialListIdx].bar(np.arange(trial_cnt), avg_vel_x_l, width, color='blue', linewidth=0)

            trialName = TrialData.TrialData.getTrialNameForIdx(trialListIdx)
            axs[trialListIdx].set_title(trialName + ' avg velocity')
            axs[trialListIdx].set_ylabel('Velocity (au)')
            axs[trialListIdx].set_xlabel('Trial number')
            axs[trialListIdx].set_ylim((-1500,4000))
            axs[trialListIdx].set_xlim((0,trial_cnt))

            if trialListIdx == 0:
                axs[trialListIdx].legend((rects1[0], rects2[0]),('Forward pre-stim', 'Lateral stim'))

        plt.show()
        filepath = self.exp_meta.analysisPath + '/velocity_all_trials_bar_plot'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)

    def show_avg_velocity_response(self):

        bdata_griddy = self.griddy.get_data()

        fig = plt.figure(figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        leftTrialIdx = TrialData.TrialData.getTrialIndexForName('Left_Odor')
        rightTrialIdx = TrialData.TrialData.getTrialIndexForName('Right_Odor')

        ax1 = fig.add_subplot(1,2,1)
        mean_of_trials_fwd_left = np.mean( bdata_griddy[leftTrialIdx][:,:,self.griddy.VEL_Y],0)
        mean_of_trials_fwd_right = np.mean( bdata_griddy[rightTrialIdx][:,:,self.griddy.VEL_Y],0)
        ax1.plot( self.griddy.get_time_grid(), mean_of_trials_fwd_left, color='r', label='Left Odor' )
        ax1.plot( self.griddy.get_time_grid(), mean_of_trials_fwd_right, color='b', label='Right Odor' )
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
        mean_of_trials_lat_left = np.mean(bdata_griddy[leftTrialIdx][:,:,self.griddy.VEL_X],0)
        mean_of_trials_lat_right = np.mean(bdata_griddy[rightTrialIdx][:,:,self.griddy.VEL_X],0)
        ax1.plot( self.griddy.get_time_grid(), mean_of_trials_lat_left, color='r', label='Left Odor' )
        ax1.plot( self.griddy.get_time_grid(), mean_of_trials_lat_right, color='b', label='Right Odor' )
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Lateral vel (au/s)')
        ax1.set_title( 'Avg lat vel all trials ' )
        ax1.set_xlim((0, 6.5))
        ylim = (-1000, 1500)
        ax1.set_ylim(ylim)
        ax1.grid()

        p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                ylim[1]-ylim[0], linewidth=0, color='wheat')
        ax1.add_patch(p)

        plt.legend(frameon=False)

        plt.show()
        filepath = self.exp_meta.analysisPath + '/avg_velocity_response'
        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)