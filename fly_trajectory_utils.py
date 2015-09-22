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


    # NOTE: Changes trial_data_all in place
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

    # NOTE: Changes trial_data_all in place
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
