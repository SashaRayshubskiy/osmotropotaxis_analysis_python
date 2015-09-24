import ExperimentMetadata
import DataLoader
import fly_trajectory_classifier as ftc

class osmotropotaxis_main:
    def __init__(self):
        #  Initialize parameters that change with each experiment
        basepath = '/Users/sasha/Documents/Wilson lab/Data/trackball/'
        fly_datapath = basepath + 'fly_health_94/'
        sid = [ 0 ]
        ##############################################################

        self.exp_mdata = ExperimentMetadata.ExperimentMetadata(experimentPath=fly_datapath, sid=sid)
        self.loader = None


    def load_data( self ):
        self.loader = DataLoader.DataLoader( self.exp_mdata )

        # Returns a list of lists, { trial_type, list of fly running trajectories, sorted by trial ordinal }
        self.loader.load_behavioral_data()

        # Load calcium imaging data
        #self.loader.load_calcium_imaging_data()

    def run(self):
        # Load in behavioral and calcium imaging data
        self.load_data()

