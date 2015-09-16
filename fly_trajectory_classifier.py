__author__ = 'sasha'

import fly_trajectory_utils
import numpy as np
import math
import matplotlib.pylab as plt

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


    # This is the port from Matlab's classification code
    def classify(self, trial_data_all):

        self.rotate( trial_data_all )
        self.calculate_velocity( trial_data_all )

        bdata_griddy = self.griddify( trial_data_all )

        plt.figure()
        
        for trialTypeIdx, cur_trial_type_bdata in enumerate( bdata_griddy ):

            # { trial, time point, [t,dx,dy] }
            # Time points that are not occupied get assigned [t, 0, 0]
            # cur_trial_type_grid_data = np.empty( (len(cur_trial_type_bdata), time_grid.shape[0], 5) )

            plt.subplot(1,len(bdata_griddy),trialTypeIdx)

            print 'cur_trial_type_bdata.shape= ' + str(cur_trial_type_bdata.shape)

            mean_of_trials = np.mean(cur_trial_type_bdata[:,:,self.VEL_X],0)

            plt.plot(self.time_grid, mean_of_trials )
            plt.xlabel('Time (s)')
            plt.ylabel('Lateral vel (au/s)')
            plt.title( 'Avg lat vel all trials: ' + self.exp_meta.trialTypes[ trialTypeIdx ] )

        plt.show()


        """
        print 'Processing run: { ' + str(trialTypeIdx) + ' , ' + str(trialIdx) + ' }'
        if trialTypeIdx == 2 and trialIdx == 157:
            plt.figure()
        # Plot the trajectory for this trial
        plt.plot(t_z[1:], vel_x)
        plt.xlabel('Time (s)')
        plt.ylabel('Lateral vel (au/s)')
        plt.title( self.exp_meta.trialTypes[ trialTypeIdx ] + ' trialIdx: ' + str(trialIdx) )
        plt.xlim((0, 6.5))
        plt.show()
                """


