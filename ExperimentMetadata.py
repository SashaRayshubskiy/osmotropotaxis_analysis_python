__author__ = 'sasha'

class ExperimentMetadata:
    def __init__(self, experimentPath, sid):
        self.frameRate = 7.8
        self.presStimTime = 3.0
        self.stimTime = 0.5
        self.postStimTime = 3.0
        self.sid = sid
        self.experimentPath = experimentPath
        self.analysisPath = self.experimentPath + '/analysis/'
        self.trialTypes = ('Both_Odor', 'Left_Odor', 'Right_Odor')
