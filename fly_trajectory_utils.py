__author__ = 'sasha'

import numpy as np
import math

class fly_trajectory_utils:
    def __init__(self, exp_meta):
        self.exp_meta = exp_meta
        self.preStimTime  = exp_meta.preStimTime
        self.stimTime     = exp_meta.stimTime
        self.trialTypes   = exp_meta.trialTypes
        self.trialTypeCnt = len(self.trialTypes)

    def calculate_trial_trajectory(self, dx, dy):
        return (np.cumsum(dx), np.cumsum(dy))

    # NOTE: Changes trial_data in place
    def rotate(self, trial_data):
        # This value determines how of the pre stimulus trajectory, before
        # the stimulation is turned on, to consider in the rotation
        self.PERIOD_BEFORE_STIM_ONSET = self.preStimTime

        trial_type_cnt = len(trial_data)

        trialTypeIdx = 0
        while trialTypeIdx < trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data[trialTypeIdx])
            trialIdx = 0
            while trialIdx < trials_in_trial_type_cnt:
                t = trial_data[trialTypeIdx][trialIdx].t
                dx = trial_data[trialTypeIdx][trialIdx].dx
                dy = trial_data[trialTypeIdx][trialIdx].dy

                t_z = t - t[ 0 ]

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #% Rotate the trial run by the direction of the pre_stim
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                qual_pre_stim_t = np.nonzero((t_z > ( self.preStimTime - self.PERIOD_BEFORE_STIM_ONSET)) & (t_z < self.preStimTime))[0]

                if qual_pre_stim_t.shape[0] <= 1:
                    trial_data[trialTypeIdx][trialIdx].dx_rot = dx
                    trial_data[trialTypeIdx][trialIdx].dy_rot = dy
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

                trial_data[trialTypeIdx][trialIdx].dx_rot = dx_rot
                trial_data[trialTypeIdx][trialIdx].dy_rot = dy_rot
                # print 'trialTypeIdx = ' + str(trialTypeIdx) + ' trialIdx=' + str(trialIdx)
                trialIdx = trialIdx + 1

            trialTypeIdx = trialTypeIdx + 1

    # NOTE: Changes trial_data in place
    def calculate_velocity( self, trial_data ):
        trial_type_cnt = len(trial_data)

        trialTypeIdx = 0
        while trialTypeIdx<trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data[trialTypeIdx])
            trialIdx = 0
            while trialIdx < trials_in_trial_type_cnt:
                t = trial_data[trialTypeIdx][trialIdx].t
                dx = trial_data[trialTypeIdx][trialIdx].dx_rot
                dy = trial_data[trialTypeIdx][trialIdx].dy_rot

                t_z = t - t[ 0 ]

                t_diff = np.diff(t_z)
                vel_x = dx[1:] / t_diff
                vel_y = dy[1:] / t_diff

                trial_data[trialTypeIdx][trialIdx].vel_x = vel_x
                trial_data[trialTypeIdx][trialIdx].vel_y = vel_y
                trialIdx = trialIdx + 1

            trialTypeIdx = trialTypeIdx + 1

    def filter_fwd_velocity(self, trial_data, vel_threshold, preStimT, stimT):
        trial_data_vel_filtered = []

        trial_type_cnt = len(trial_data)

        trialTypeIdx = 0
        while trialTypeIdx<trial_type_cnt:
            trials_in_trial_type_cnt = len(trial_data[ trialTypeIdx ])
            trial_data_vel_filtered.append([])
            trialIdx = 0
            while trialIdx < trials_in_trial_type_cnt:

                t = trial_data[ trialTypeIdx ][ trialIdx ].t
                tz = t-t[0]

                pre_stim_t = np.nonzero( tz < preStimT )
                stim_t = np.nonzero((tz >= preStimT) & (tz < (preStimT+stimT)))

                vel_y = trial_data[ trialTypeIdx ][ trialIdx ].vel_y

                if np.mean(vel_y[pre_stim_t]) > vel_threshold and stim_t[0].shape[0] > 0:
                    trial_data_vel_filtered[ trialTypeIdx ].append(trial_data[ trialTypeIdx ][ trialIdx ])
                    #print 'Accepted trial (' + str(trialTypeIdx) + ',' + str(trialIdx) + ')'

                trialIdx = trialIdx + 1

            trialTypeIdx = trialTypeIdx + 1

        return trial_data_vel_filtered
        