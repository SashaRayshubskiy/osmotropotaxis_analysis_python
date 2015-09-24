__author__ = 'sasha'

import numpy as np

class fly_trajectory_griddy:
    def __init__(self, exp_meta, trial_data):
        self.trial_data = trial_data
        self.exp_meta = exp_meta
        self.T            = 0
        self.DX           = 1
        self.DY           = 2
        self.VEL_X        = 3
        self.VEL_Y        = 4

        # Grid related, think about making this a separate class
        self.TIME_GRID_SPACING = 60 # per second
        self.TIME_GRID_SIZE    = self.exp_meta.preStimTime + self.exp_meta.stimTime + self.exp_meta.postStimTime + 2.0
        self.time_grid = np.arange( 0, self.TIME_GRID_SIZE, 1.0/self.TIME_GRID_SPACING )

        self.bdata_griddy = None
        self.griddify()

    def get_time_grid(self):
        return self.time_grid

    def get_data(self):
        return self.bdata_griddy

    def griddify(self):

        self.bdata_griddy = []
        trial_type_cnt = len(self.trial_data)

        trialTypeIdx = 0
        while trialTypeIdx<trial_type_cnt:
            trials_in_trial_type_cnt = len(self.trial_data[trialTypeIdx])
            trialIdx = 0

            # { trial, time point, [t,dx,dy,vel_x, vel_y] }
            # Time points that are not occupied get assigned [t, 0, 0]
            cur_trial_type_grid_data = np.zeros( (trials_in_trial_type_cnt, self.time_grid.shape[0], 5) )

            self.bdata_griddy.append( cur_trial_type_grid_data )

            while trialIdx < trials_in_trial_type_cnt:
                t = self.trial_data[trialTypeIdx][trialIdx].t
                dx_rot = self.trial_data[trialTypeIdx][trialIdx].dx_rot
                dy_rot = self.trial_data[trialTypeIdx][trialIdx].dy_rot
                vel_x = self.trial_data[trialTypeIdx][trialIdx].vel_x
                vel_y = self.trial_data[trialTypeIdx][trialIdx].vel_y

                t_z = t - t[ 0 ]

                time_grid_idx = 0
                for i, t_i in enumerate(t_z[1:]):
                    while self.time_grid[time_grid_idx] < t_i:
                        time_grid_idx = time_grid_idx  + 1

                    # This sets the format for behavioral data on a grid
                    self.bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.T ] = t_i
                    self.bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.DX ] = dx_rot[ i+1 ]
                    self.bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.DY ] = dy_rot[ i+1 ]
                    self.bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.VEL_X ] = vel_x[ i ]
                    self.bdata_griddy[ trialTypeIdx ][ trialIdx ][ time_grid_idx ][ self.VEL_Y ] = vel_y[ i ]

                trialIdx = trialIdx + 1
            trialTypeIdx = trialTypeIdx + 1
