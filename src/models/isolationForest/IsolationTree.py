import random

from src.models.isolationForest.IsolationNode import IsolationNode


class IsolationTree(object):

    def __init__(self, input_features, height_limit):
        self.input_features = input_features
        self.height_limit = height_limit
        self.root = self.create(0)

    def create(self, depth):
        rnd_feature_index = random.randint(0, len(self.input_features) - 1)
        selected_feature = self.input_features[rnd_feature_index]

        #cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or len(selected_feature.samples_set) <=1:
            return None

        split_value = random.randint(0, len(selected_feature.samples_set) - 1)

        inode = IsolationNode(selected_feature.name,
                              selected_feature.samples_set[split_value].value,
                              selected_feature.samples_set[split_value].observation_id)



        return inode

