__author__ = 'sasha'

class TrialData:
    def __init__(self, sid, ordinal, timestamp, t, dx, dy):

        # Trial metadata
        self.trial_sid = sid
        self.trial_ord = ordinal
        self.timestamp = timestamp

        # Raw behavioral data
        self.t = t
        self.dx = dx
        self.dy = dy

        # Rotated behavioral variables for
        self.vel_x = None
        self.vel_y = None
        self.dx_rot = None
        self.dy_rot = None

        # Calcium imaging data
        self.cdata = None

    def __cmp__(self, other):
        if self.trial_sid < other.trial_sid:
            return -1
        elif self.trial_sid == other.trial_sid:
            if self.trial_ord < other.trial_ord:
                return -1
            elif self.trial_ord == other.trial_ord:
                return 0
            else:
                return 1
        else:
            return 1

    @staticmethod
    def getTrialIndexForName(trial_name):

        trial_type_idx = -1

        if trial_name == 'Both_Odor':
            trial_type_idx = 0
        elif trial_name == 'Left_Odor':
            trial_type_idx = 1
        elif trial_name == 'Right_Odor':
            trial_type_idx = 2
        else:
            print 'ERROR: Trial type: ' + trial_name + ' not matched.'

        return trial_type_idx

    @staticmethod
    def getTrialNameForIdx(idx):

        trial_name = 'None'

        if idx == 0:
            trial_name = 'Both_Odor'
        elif idx == 1:
            trial_name = 'Left_Odor'
        elif idx == 2:
            trial_name = 'Right_Odor'
        else:
            print 'ERROR: Trial index -- ' + str(idx) + ' not matched.'

        return trial_name