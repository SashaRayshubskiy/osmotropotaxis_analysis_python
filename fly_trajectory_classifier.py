__author__ = 'sasha'

import fly_trajectory_utils
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib.patches as patches
import TrialData

from sklearn.cluster import KMeans

class fly_trajectory_classifier:
    def __init__(self, exp_meta):
        self.exp_meta = exp_meta
        self.preStimTime  = exp_meta.preStimTime
        self.stimTime     = exp_meta.stimTime
        self.trialTypes   = exp_meta.trialTypes
        self.trialTypeCnt = len(self.trialTypes)
        self.T            = 0
        self.DX           = 1
        self.DY           = 2
        self.VEL_X        = 3
        self.VEL_Y        = 4


    # Put all the behavioral data on a common grid to compare between trials
    # NOTE: Assumes the input comes with velocity
    def griddify(self, trial_data_all):

        bdata_griddy = []

        self.TIME_GRID_SPACING = 60 # per second
        self.TIME_GRID_SIZE    = self.exp_meta.preStimTime + self.exp_meta.stimTime + self.exp_meta.postStimTime + 2.0

        self.time_grid = np.arange( 0, self.TIME_GRID_SIZE, 1.0/self.TIME_GRID_SPACING )        
        
        trial_type_cnt = len(trial_data_all)

        trialTypeIdx = 0
        while trialTypeIdx<trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data_all[trialTypeIdx])
            trialIdx = 0

            # { trial, time point, [t,dx,dy,vel_x, vel_y] }
            # Time points that are not occupied get assigned [t, 0, 0]
            cur_trial_type_grid_data = np.empty( (trials_in_trial_type_cnt, self.time_grid.shape[0], 5) )

            bdata_griddy.append( cur_trial_type_grid_data )

            while trialIdx < trials_in_trial_type_cnt:
                t = trial_data_all[trialTypeIdx][trialIdx].t
                dx_rot = trial_data_all[trialTypeIdx][trialIdx].dx_rot
                dy_rot = trial_data_all[trialTypeIdx][trialIdx].dy_rot
                vel_x = trial_data_all[trialTypeIdx][trialIdx].vel_x
                vel_y = trial_data_all[trialTypeIdx][trialIdx].vel_y

                t_z = t - t[ 0 ]

                time_grid_idx = 0
                for i, t_i in enumerate(t_z[1:]):
                    while self.time_grid[time_grid_idx] < t_i:
                        time_grid_idx = time_grid_idx  + 1

                    # This sets the format for behavioral data on a grid
                    bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.T ] = t_i
                    bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.DX ] = dx_rot[ i+1 ]
                    bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.DY ] = dy_rot[ i+1 ]
                    bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.VEL_X ] = vel_x[ i ]
                    bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.VEL_Y ] = vel_y[ i ]

                trialIdx = trialIdx + 1
            trialTypeIdx = trialTypeIdx + 1

        return bdata_griddy

    def rotate(self, trial_data_all):

        # This value determines how of the pre stimulus trajectory, before
        # the stimulation is turned on, to consider in the rotation
        self.PERIOD_BEFORE_STIM_ONSET = 0.5

        trial_type_cnt = len(trial_data_all)

        trialTypeIdx = 0
        while trialTypeIdx < trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data_all[trialTypeIdx])
            trialIdx = 0
            while trialIdx < trials_in_trial_type_cnt:
                t = trial_data_all[trialTypeIdx][trialIdx].t
                dx = trial_data_all[trialTypeIdx][trialIdx].dx
                dy = trial_data_all[trialTypeIdx][trialIdx].dy

                t_z = t - t[ 0 ]

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #% Rotate the trial run by the direction of the pre_stim
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                qual_pre_stim_t = np.nonzero((t_z > ( self.preStimTime - self.PERIOD_BEFORE_STIM_ONSET)) & (t_z < self.preStimTime))[0]

                if qual_pre_stim_t.shape[0] <= 1:
                    trial_data_all[trialTypeIdx][trialIdx].dx_rot = dx
                    trial_data_all[trialTypeIdx][trialIdx].dy_rot = dy
                    trialIdx = trialIdx + 1
                    continue

                dir_pre_x = np.sum(dx[qual_pre_stim_t])
                dir_pre_y = np.sum(dy[qual_pre_stim_t])
                pre_angle_rad = math.atan2( dir_pre_y, dir_pre_x )

                rot_rad = pre_angle_rad - math.pi/2.0
                R = np.array([[math.cos(rot_rad), -1*math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]])
                v = np.array([dx, dy]).transpose()
                vR = np.dot(v, R)

                dx_rot = vR[:, 0]
                dy_rot = vR[:, 1]
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                trial_data_all[trialTypeIdx][trialIdx].dx_rot = dx_rot
                trial_data_all[trialTypeIdx][trialIdx].dy_rot = dy_rot
                # print 'trialTypeIdx = ' + str(trialTypeIdx) + ' trialIdx=' + str(trialIdx)
                trialIdx = trialIdx + 1

            trialTypeIdx = trialTypeIdx + 1
        
    def calculate_velocity( self, trial_data_all ):
        trial_type_cnt = len(trial_data_all)

        trialTypeIdx = 0
        while trialTypeIdx<trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data_all[trialTypeIdx])
            trialIdx = 0
            while trialIdx < trials_in_trial_type_cnt:
                t = trial_data_all[trialTypeIdx][trialIdx].t
                dx = trial_data_all[trialTypeIdx][trialIdx].dx
                dy = trial_data_all[trialTypeIdx][trialIdx].dy

                t_z = t - t[ 0 ]
    
                t_diff = np.diff(t_z)
                vel_x = dx[1:] / t_diff
                vel_y = dy[1:] / t_diff

                trial_data_all[trialTypeIdx][trialIdx].vel_x = vel_x
                trial_data_all[trialTypeIdx][trialIdx].vel_y = vel_y
                trialIdx = trialIdx + 1

            trialTypeIdx = trialTypeIdx + 1

    def prime(self, trial_data_all):
        self.rotate( trial_data_all )
        self.calculate_velocity( trial_data_all )
        self.bdata_griddy = self.griddify( trial_data_all )

    def plot_avg_velocity_response(self):

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

    def classify(self, trial_data_all):
        self.classify_kmeans( trial_data_all )

    def classify_kmeans(self, trial_data_all):

        N_CLUSTERS = 3

        BEGIN_TIME = 0
        END_TIME = 6.5

        BEGIN_TIME_FRAME = BEGIN_TIME*self.TIME_GRID_SPACING
        END_TIME_FRAME = END_TIME*self.TIME_GRID_SPACING

        trialT = ['Left_Odor', 'Right_Odor']
        fig = plt.figure(figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

        for trialTypeIdx, trialType in enumerate(trialT):


            # Trying kmeans
            myTrialIdx = TrialData.TrialData.getTrialIndexForName(trialType)
            data = self.bdata_griddy[myTrialIdx][:,BEGIN_TIME_FRAME:END_TIME_FRAME,self.VEL_X]

            kmeans = KMeans(n_clusters=N_CLUSTERS)
            kmeans.fit(data)
            labels = kmeans.labels_

            #for c in range(N_CLUSTERS):
            #    ax1.plot(self.time_grid, kmeans.cluster_centers_[c], label='Cluster center: ' + str(c))

            for c in range(N_CLUSTERS):
                ax1 = plt.subplot2grid((N_CLUSTERS,2), (c, trialTypeIdx))

                found_labels = np.nonzero( labels == c )

                cur_labeled_data = data[found_labels,:]
                mean_cur_labeled_data = np.mean(cur_labeled_data,0)

                cur_labeled_data_len = cur_labeled_data.shape[0]
                idx = 0
                while idx < cur_labeled_data_len:
                    ax1.plot(self.time_grid, cur_labeled_data[idx,:], color='grey')

                ax1.plot(self.time_grid, mean_cur_labeled_data, label='Avg', color='k')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Lateral vel (au/s)')
                ax1.set_title('kmeans(' + str(N_CLUSTERS) + ') lat vel : ' + trialType)
                ax1.set_xlim((0, 6.5))
                ylim = (-5000, 5000)
                ax1.set_ylim(ylim)
                p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                    ylim[1]-ylim[0], linewidth=0, color='wheat')

                ax1.add_patch(p)

                plt.legend(frameon=False)

        plt.show()

        filepath = self.exp_meta.analysisPath + '/behaviour_kmeans_clustering_' \
                   + str(N_CLUSTERS) + '_begT_' + str(BEGIN_TIME) + '_end_T_' + str(END_TIME)

        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)



