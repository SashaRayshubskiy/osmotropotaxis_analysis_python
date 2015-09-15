__author__ = 'sasha'

import fly_trajectory_classifier as ftc

class osmotropotaxis_test:
    def __init__(self):
        pass

    def run(self, bdata, exp_mdata):
        classifier = ftc.fly_trajectory_classifier(exp_mdata)
        classifier.classify(bdata)
