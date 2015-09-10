__author__ = 'sasha'

import glob
import numpy as np
import os.path as ospath
import scipy.io as sio
import skimage.io as skio
import time
import ExperimentMetadata

class DataLoader:
    def __init__(self, expMeta):
        self.expMeta = expMeta
        self.datapath = self.expMeta.experimentPath

    def load_behavioral_data(self, sids, trial_types):
        bdata_path = self.datapath + '/ball/'

        files = []
        for sidIdx in self.expMeta.sids:
            files = files + glob.glob( bdata_path + '/*_sid_' + str(sidIdx) + '_*.mat' )

        trial_type_cnt = len(self.expMeta.trial_types)

        trial_data = []
        for i in range(trial_type_cnt):
            trial_data.append([])

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

            trial_type_idx = -1

            if trial_type == 'Both_Odor':
                trial_type_idx = 0
            elif trial_type == 'Left_Odor':
                trial_type_idx = 1
            elif trial_type == 'Right_Odor':
                trial_type_idx = 2
            elif trial_type == 'Both_Air':
                trial_type_idx = 3
            elif trial_type == 'Left_Air':
                trial_type_idx = 4
            elif trial_type == 'Right_Air':
                trial_type_idx = 5
            else:
                print 'ERROR: Trial type: ' + trial_type + ' not matched.'

            print 'Trial filename: ' + filename
            print 'Trial id: ' + trial_id
            print 'Trial type idx: ' + str(trial_type_idx)
            print 'Trial type: ' + trial_type

            ddd = sio.loadmat( filepath )

            if len(ddd['t'][0]) <= 1:
                continue

            trial_data[ trial_type_idx ].append( ( (int(trial_sid), int(trial_ord)), time.strptime( trial_date_time, '%Y_%m%d_%I%M%S'), ddd ) )

        for trial_list in trial_data:
            trial_list.sort()

        return ( trial_data )

    def load_calcium_imaging_data(self):
        cdata_path = self.datapath + '/2p/'

        trial_type_cnt = len(self.expMeta.trial_types)

        trial_data = []
        for tt in range(trial_type_cnt):
            trial_data.append([])

        for tt in range(trial_type_cnt):
            files = []
            for sidIdx in self.expMeta.sids:
                files = files + glob.glob( cdata_path + '/*' + self.expMeta.trial_types[ tt ] + '_' + str(sidIdx) + '_*.tif' )

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
                trial_data[ tt ].append( ((trial_sid, trial_idx), cur_data) )
                print 'Loaded file: ' + filename

            trial_data[ tt ].sort()

        return trial_data