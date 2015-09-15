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

    def load_data( self ):
        loader = DataLoader.DataLoader( self.exp_mdata )

        # Returns a list of lists, { trial_type, list of fly running trajectories, sorted by trial ordinal }
        bdata = loader.load_behavioral_data()

        # Load calcium imaging data
        cdata = loader.load_calcium_imaging_data()

        # Check that the behavioral and imaging data line up correctly
        img_cnt = len(cdata)
        run_cnt = len(bdata)
        assert(img_cnt == run_cnt)

        for tt in range(img_cnt):
            cdata_tt_cnt = len(cdata[tt])
            bdata_tt_cnt = len(bdata[tt])
            assert(cdata_tt_cnt == bdata_tt_cnt)

            # ATTENTION ATTENTION ATTENTION
            # Add the code here that compares the behavioral trial id with imaging trial id.
            # Behavioral tracking code will send a unique trial id to the imaging code.

        return ( bdata, cdata )

    def run(self):
        # Load in behavioral and calcium imaging data
        self.bdata, self.cdata = self.load_data()

