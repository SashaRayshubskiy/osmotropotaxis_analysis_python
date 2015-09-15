import fly_trajectory_classifier

class osmotropotaxis_test:
    def __init__(self):
        classifier = ftc.fly_trajectory_classifier(self.exp_mdata)
        classifier.classify(self.bdata)
