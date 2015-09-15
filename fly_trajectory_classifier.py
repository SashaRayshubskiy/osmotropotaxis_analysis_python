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

        # This value determines how of the pre stimulus trajectory, before
        # the stimulation is turned on, to consider in the rotation
        self.PERIOD_BEFORE_STIM_ONSET = 0.5

    # This method classifies into N clusters
    def classify(self, bdata):
        for trialTypeIdx, cur_trial_type_bdata in enumerate(bdata):
            for trialIdx, trial in enumerate(cur_trial_type_bdata):
                trial_data = trial[ 2 ]

                t = trial_data[ 't' ]
                t = t.reshape(t.shape[1],)
                dx = trial_data[ 'dx' ]
                dx = dx.reshape(dx.shape[1],)
                dy = trial_data[ 'dy' ]
                dy = dy.reshape(dy.shape[1],)

                #print 't.ndim: ' + str(t.ndim) + ' dx.ndim: ' + str(dx.ndim) + ' dy.ndim: ' + str(dy.ndim)
                #print 't.shape: ' + str(t.shape) + ' dx.shape: ' + str(dx.shape) + ' dy.shape: ' + str(dy.shape)

                t_z = t - t[ 0 ]

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #% Rotate the trial run by the direction of the pre_stim
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                qual_pre_stim_t = np.nonzero((t_z > ( self.preStimTime - self.PERIOD_BEFORE_STIM_ONSET)) & (t_z < self.preStimTime))[0]

                #print 'qual_pre_stim_t: ' + str(qual_pre_stim_t)
                #print 'qual_pre_stim_t.shape: ' + str(qual_pre_stim_t.shape)

                if qual_pre_stim_t.shape[0] <= 1:
                    continue

                dir_pre_x = np.sum(dx[qual_pre_stim_t])
                dir_pre_y = np.sum(dy[qual_pre_stim_t])
                pre_angle_rad = math.atan2( dir_pre_y, dir_pre_x )

                rot_rad = pre_angle_rad - math.pi/2.0
                R = np.array([[math.cos(rot_rad), -1*math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]])

                v = np.array([dx, dy]).transpose()

                #print v

                #print 'v.shape =', str(v.shape)
                vR = np.dot(v, R)

                #print 'vR.shape=' + str(vR.shape)
                dx = vR[:, 0]
                dy = vR[:, 1]
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                pre_stim_t =   np.nonzero(t_z < self.preStimTime)
                stim_t = np.nonzero((self.preStimTime >= self.preStimTime) & (t_z < (self.preStimTime+self.stimTime)))

                t_diff = np.diff(t_z)
                #print 't_diff.shape: ' + str(t_diff.shape) + ' dx.shape: ' + str(dx.shape) + ' dy.shape: ' + str(dy.shape)
                vel_x = dx[1:] / t_diff
                vel_y = dy[1:] / t_diff


                if trialTypeIdx == 1 and trialIdx == 2:
                    plt.figure()
                    # Plot the trajectory for this trial
                    plt.plot(t_z[1:], vel_x)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Lateral vel (au/s)')
                    plt.title( self.exp_meta.trialTypes[ trialTypeIdx ] + ' trialIdx: ' + str(trialIdx) )
                    plt.xlim((0, 6.5))
                    plt.show()
