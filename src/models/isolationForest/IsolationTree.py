import random

from src.models.isolationForest.IsolationNode import IsolationNode


class IsolationTree(object):

    def __init__(self, input_data, height_limit):
        self.input_data = input_data #input features set with its observations
        self.height_limit = height_limit #height limit of the tree
        self.root = self.create(0, self.input_data) #call create method which will create the whole

    def create(self, depth, input_data):
        rnd_feature_index = random.randint(0, len(input_data) - 1)
        selected_feature = input_data[rnd_feature_index]

        #cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or len(selected_feature.samples_set) <=1:
            return None

        #find the split value index
        split_index = random.randint(0, len(selected_feature.samples_set) - 1)

        #create inode with current value and its corresponding observation id
        inode = IsolationNode(selected_feature.name,
                              rnd_feature_index,
                              selected_feature.samples_set[split_index].value,
                              selected_feature.samples_set[split_index].observation_id)

        #split the feature sets to give to left/right child
        left_features, right_features = self.split_feature_sets(input_data, split_index)

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
