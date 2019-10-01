import numpy as np

class Bag():
    def __init__(self, features, bag_label=None):
        """
        Creates an object containing data
        :param features: list of instances
        :param bag_label: int describing class of bucket
        """
        self.features = np.array(features)
        if bag_label is not None:
            if bag_label != 1:
                self.bag_label = -1
            else:
                self.bag_label = 1

    def __str__(self):
        return 'Features = {}, Label={}\n'.format(self.features, self.bag_label)

    def __repr__(self):
        return 'Label={}\n'.format( self.bag_label)
