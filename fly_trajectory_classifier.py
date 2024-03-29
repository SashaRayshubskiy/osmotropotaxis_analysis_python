__author__ = 'sasha'

import fly_trajectory_utils as ftu
import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib.patches as patches
import TrialData

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from scipy.io import savemat

class fly_trajectory_classifier:
    def __init__(self, exp_meta, griddy):
        self.exp_meta = exp_meta
        self.preStimTime  = exp_meta.preStimTime
        self.stimTime     = exp_meta.stimTime
        self.trialTypes   = exp_meta.trialTypes
        self.trialTypeCnt = len(self.trialTypes)
        self.griddy = griddy
        self.fly_traj_utils = ftu.fly_trajectory_utils(self.exp_meta)

    def classify_core(self, N_CLUSTERS, clusterType, data_for_trial_type, begin_time, end_time):

        BEGIN_TIME_FRAME = begin_time*self.griddy.TIME_GRID_SPACING
        END_TIME_FRAME = end_time*self.griddy.TIME_GRID_SPACING

        data = data_for_trial_type[:,BEGIN_TIME_FRAME:END_TIME_FRAME,self.griddy.VEL_X]

        labels = None
        if clusterType == 'kmeans':
            kmeans = KMeans(n_clusters=N_CLUSTERS)
            kmeans.fit(data)
            labels = kmeans.labels_
        elif clusterType == 'affinity_propagation':
            ap = AffinityPropagation(damping=0.75)
            ap.fit(data)
            labels = ap.labels_
            N_CLUSTERS = np.max(self.labels)+1
        elif clusterType == 'DBSCAN':
            dbscan = DBSCAN()
            dbscan.fit(data)
            labels = dbscan.labels_
            N_CLUSTERS = np.max(labels)+1
            print 'N_CLUSTERS=' + str(N_CLUSTERS)
        elif clusterType == 'AgglomerativeClustering':
            ac = AgglomerativeClustering(n_clusters=N_CLUSTERS)
            ac.fit(data)
            labels = ac.labels_
        else:
            print 'ERROR: clusterType: ' + clusterType + ' is not recognized'

        return (labels, N_CLUSTERS)

    def classify(self, clusterType, N_CLUSTERS):
        trialT = ['Left_Odor', 'Right_Odor']
        BEGIN_TIME = 2.9
        END_TIME = 4.5

        bdata_griddy = self.griddy.get_data()

        first_pass = True

        dict_to_mat = {}

        for trialTypeIdx, trialType in enumerate(trialT):

            myTrialIdx = TrialData.TrialData.getTrialIndexForName(trialType)

            labels, N_CLUSTERS = self.classify_core(N_CLUSTERS, clusterType, bdata_griddy[myTrialIdx], BEGIN_TIME, END_TIME)

            data_all = bdata_griddy[myTrialIdx][:,:,self.griddy.VEL_X]

            if trialTypeIdx == 0:
                fig, axs = plt.subplots(N_CLUSTERS, 2, sharex=True, sharey=True, figsize=(14, 6.7), dpi=100, facecolor='w', edgecolor='k')

            #for c in range(N_CLUSTERS):
            #    ax1.plot(self.time_grid, kmeans.cluster_centers_[c], label='Cluster center: ' + str(c))

            for c in range(N_CLUSTERS):

                ax1 = axs[c, trialTypeIdx]
                #ax1 = axs[c]

                found_labels = np.nonzero( labels == c )

                dict_to_mat_key = trialType + '_clust_' + str(c)
                dict_to_mat[ dict_to_mat_key ] = found_labels[0]

                if len(found_labels[0]) == 1:
                    cur_labeled_data = data_all[found_labels[0],:]
                else:
                    cur_labeled_data = np.squeeze(data_all[found_labels[0],:])

                mean_cur_labeled_data = np.mean(cur_labeled_data,0)

                cur_labeled_data_len = cur_labeled_data.shape[0]
                idx = 0
                while idx < cur_labeled_data_len:
                    ax1.plot(self.griddy.get_time_grid(), cur_labeled_data[idx,:], color='grey')
                    idx = idx + 1

                ax1.plot(self.griddy.get_time_grid(), mean_cur_labeled_data, label='cluster: ' + str(c) + ' [' + str(cur_labeled_data_len) + ']', color='k')

                if c == N_CLUSTERS-1:
                    ax1.set_xlabel('Time (s)', labelpad=2)

                if trialTypeIdx == 0:
                    ax1.set_ylabel('Lateral vel (au/s)')

                if c == 0:
                    ax1.set_title( clusterType+'(' + str(N_CLUSTERS) + ') lat vel : ' + trialType)

                xlim = (0, 6.5)
                ylim = (-5000, 5000)
                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                p = patches.Rectangle((self.exp_meta.preStimTime,ylim[0]), self.exp_meta.stimTime, \
                                    ylim[1]-ylim[0], linewidth=0, color='wheat')

                ax1.add_patch(p)

                p = patches.Rectangle((BEGIN_TIME,ylim[0]), END_TIME-BEGIN_TIME, \
                                    1000, linewidth=0, color='peru')

                ax1.add_patch(p)
                # ax1.set_text(xlim[1]-2, ylim[1]-1000, 'cluster: ' + str(c) + ' [' + str(cur_labeled_data_len) + ']')

                ax1.legend(frameon=False, borderaxespad=0.05, borderpad=0.05)

        fig.subplots_adjust(hspace=0.1)
        # plt.tight_layout()
        plt.show()

        filepath = self.exp_meta.analysisPath + '/behaviour_' + clusterType + '_clustering_' \
                   + str(N_CLUSTERS) + '_begT_' + str(BEGIN_TIME) + '_end_T_' + str(END_TIME)

        plt.savefig(filepath+'.png', bbox_inches='tight')
        plt.savefig(filepath+'.pdf', bbox_inches='tight')
        plt.savefig(filepath+'.eps', bbox_inches='tight', format='eps', dpi=1000)

        savemat( filepath +'.mat', dict_to_mat, long_field_names=True)



