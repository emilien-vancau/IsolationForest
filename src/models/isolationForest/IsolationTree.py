import random

from src.models.isolationForest.IsolationNode import IsolationNode


class IsolationTree(object):

    def __init__(self, input_features, height_limit):
        self.input_features = input_features
        self.height_limit = height_limit
        self.root = self.create(0, self.input_features)

    def create(self, depth, input_features):
        rnd_feature_index = random.randint(0, len(input_features) - 1)
        selected_feature = input_features[rnd_feature_index]

        #cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or len(selected_feature.samples_set) <=1:
            return None

        split_index = random.randint(0, len(selected_feature.samples_set) - 1)

        inode = IsolationNode(selected_feature.name,
                              selected_feature.samples_set[split_index].value,
                              selected_feature.samples_set[split_index].observation_id)

        left_features, right_features = self.split_feature_sets(input_features, split_index)

        inode.set_left_child(self.create(depth+1, left_features))
        inode.set_right_child(self.create(depth+1, right_features))

        return inode

    def split_feature_sets(self, feature_sets, split_index):
        left_features = []
        right_features = []
        for feature in feature_sets:
            left_feature, right_feature = feature.split_samples(split_index)
            left_features.append(left_feature)
            right_features.append(right_feature)

        return left_features, right_features
