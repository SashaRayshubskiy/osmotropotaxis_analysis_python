__author__ = 'sasha'

import glob
import numpy as np
import os.path as ospath
import scipy.io as sio
import skimage.io as skio
import time
import ExperimentMetadata
import TrialData

class DataLoader:
    def __init__(self, expMeta):
        self.expMeta = expMeta
        self.datapath = self.expMeta.experimentPath

        self.trial_data_all = None

    def load_behavioral_data(self):
        bdata_path = self.datapath + '/ball/'

        files = []
        for sidIdx in self.expMeta.sid:
            files = files + glob.glob( bdata_path + '/*_sid_' + str(sidIdx) + '_*.mat' )

        trial_type_cnt = len(self.expMeta.trialTypes)

        self.trial_data_all = []
        for i in range(trial_type_cnt):
            self.trial_data_all.append([])

        for filepath in files:
            filename = ospath.basename( filepath )
            fs = filename.split('_')
            tmp = fs[9].split('.')

            trial_ord = tmp[0]

            print fs

            trial_sid = fs[4]
            trial_type = fs[5] + '_' + fs[6]
            trial_id   = fs[0] + '_' + fs[1] + '_' + fs[2] + '_' + trial_ord

            trial_date_time = fs[0] + '_' + fs[1] + '_' + fs[2]

            trial_type_idx = TrialData.TrialData.getTrialIndexForName(trial_type)

            print 'Trial filename: ' + filename
            print 'Trial id: ' + trial_id
            print 'Trial type idx: ' + str(trial_type_idx)
            print 'Trial type: ' + trial_type

            ddd = sio.loadmat( filepath )

            if len(ddd['t'][0]) <= 1:
                continue

            t = ddd['t'].reshape(ddd['t'].shape[1],)
            dx = ddd['dx'].reshape(ddd['dx'].shape[1],)
            dy = ddd['dy'].reshape(ddd['dy'].shape[1],)

            # This defines the format of the behavioral data in this analysis package.
            # NOTE: This format only changes here!
            trial_data = TrialData.TrialData( trial_sid, int(trial_ord), time.strptime( trial_date_time, '%Y_%m%d_%I%M%S'), t, dx, dy )
            self.trial_data_all[ trial_type_idx ].append( trial_data )

        for trial_list in self.trial_data_all:
            trial_list.sort()

    def load_calcium_imaging_data(self):
        cdata_path = self.datapath + '/2p/'

        trial_type_cnt = len(self.expMeta.trialTypes)

        for tt in range(trial_type_cnt):
            files = []
            for sidIdx in self.expMeta.sid:
                files = files + glob.glob( cdata_path + '/*' + self.expMeta.trialTypes[ tt ] + '_' + str(sidIdx) + '_*.tif' )

            cur_data_list = []
            for filepath in files:
                filename = ospath.basename( filepath )
                fs = filename.split('.')
                tmp = fs[ 0 ].split('_')

                trial_idx = int( tmp[ -1 ] )
                trial_sid = int( tmp[ -2 ] )

                im_collection = skio.MultiImage( filepath )

                # Convert to np array
                cur_data = next(iter(im_collection))
                cur_data = np.transpose(cur_data, (2,0,1))
                cur_data_list.append( ((trial_sid, trial_idx), cur_data) )
                print 'Loaded file: ' + filename

            cur_data_list.sort()

            cur_data_list_len = len(cur_data_list)
            i = 0
            while i<cur_data_list_len:
                self.trial_data_all[tt][i].cdata = cur_data_list[i][1]
                i = i + 1
