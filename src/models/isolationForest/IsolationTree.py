import random
import numpy as np

from src.models.feature.FeaturesSet import FeaturesSet
from src.models.isolationForest.IsolationLeaf import IsolationLeaf
from src.models.isolationForest.IsolationNode import IsolationNode


class IsolationTree(object):

    def __init__(self, input_data, height_limit):
        self.input_data = input_data #input features set with its observations
        self.height_limit = height_limit #height limit of the tree
        self.root = self.create(0, self.input_data)
        #self.root = self.create_experimental(0, self.input_data) #call create_experimental method which will create the whole tree recursively

    def create(self, depth, input_values):

        # cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or input_values.shape[0] <= 1:
            return IsolationLeaf(input_values.shape[0])

        # get feature to split
        selected_feature = random.randint(0, input_values.shape[1]-1)
        #get random value to use as a split point
        split_value = np.random.uniform(min(input_values[:, selected_feature]), max(input_values[:, selected_feature]))

        # create inode with current value and its corresponding observation id
        # feature name and observation id not needed in this case
        inode = IsolationNode(None,
                              selected_feature,
                              split_value,
                              None)

        #split the feature sets to give to left/right child
        left_values = input_values[input_values[:, selected_feature] < split_value]
        right_values = input_values[input_values[:, selected_feature] >= split_value]

        #create recusively left/right subtree
        inode.set_left_child(self.create(depth + 1, left_values))
        inode.set_right_child(self.create(depth + 1,right_values))

        return inode

    def create_experimental(self, depth, input_data):
        '''this method is experimental'''

        #cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or input_data.size <=1:
            return IsolationLeaf(input_data.size)

        rnd_feature_index = random.randint(0, len(input_data.features_set) - 1)
        selected_feature = input_data.features_set[rnd_feature_index]

        #find the split value index
        split_index = random.randint(0, len(selected_feature.samples_set) - 1)

        #create inode with current value and its corresponding observation id
        inode = IsolationNode(selected_feature.name,
                              rnd_feature_index,
                              selected_feature.samples_set[split_index].value,
                              selected_feature.samples_set[split_index].observation_id)

        #split the feature sets to give to left/right child
        left_features_set, right_features_set = self.split_features_set(input_data, split_index)

        inode.set_left_child(self.create_experimental(depth + 1, left_features_set))
        inode.set_right_child(self.create_experimental(depth + 1, right_features_set))

        return inode

    def split_features_set(self, input_data, split_index):

        max_left_size = 0
        max_right_size = 0
        left_features = []
        right_features = []
        for feature in input_data.features_set:
            left_feature, right_feature, left_size, right_size = feature.split_samples(split_index)

            if left_size > max_left_size:
                max_left_size = left_size
            if right_size > max_right_size:
                max_right_size = right_size

            left_features.append(left_feature)
            right_features.append(right_feature)

        left_features_set = FeaturesSet(max_left_size, left_features)
        right_features_set = FeaturesSet(max_right_size, right_features)

        return left_features_set, right_features_set
