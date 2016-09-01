"""
    Written by Kiani Lannoye & Gilles Vandewiele
    Commissioned by UGent.

    Design of a diagnose- and follow-up platform for patients with chronic headaches
"""


class TreeConstructor(object):
    """
    This class is an interface for all tree induction algorithms.
    """

    def __init__(self):
        """
        In the init method, all hyperparameters should be set.
        """
        raise NotImplementedError("This method needs to be implemented")

    def get_name(self):
        """
        This just returns the name of the induction algorithm implemented.
        """
        raise NotImplementedError("This method needs to be implemented")

    def construct_tree(self, training_feature_vectors, labels):
        """
        :param training_feature_vectors: a pandas dataframe containing all feature vectors for all samples
        :param labels: a pandas dataframe with the same ordering as the feature vectors, containing the classes
        :return: a decision tree object (objects.decisiontree)
        """
        raise NotImplementedError("This method needs to be implemented")