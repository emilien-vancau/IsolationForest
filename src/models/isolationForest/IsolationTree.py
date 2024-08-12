import random
import numpy as np

from src.models.isolationForest.IsolationLeaf import IsolationLeaf
from src.models.isolationForest.IsolationNode import IsolationNode


class IsolationTree(object):

    def __init__(self, input_data, height_limit):
        self.input_data = input_data #input features set with its observations
        self.height_limit = height_limit #height limit of the tree
        self.inode_cnt = 0
        self.ileaf_cnt = 0
        self.root = self.__create(0, self.input_data)
        #self.root = self.__create_experimental(0, self.input_data) #call create_experimental method which will create the whole tree recursively

    def __create(self, depth, input_values):
        """
            Recursively create isolation tree nodes.

            Args:
                depth (int): Current depth in the tree.
                input_values (numpy.ndarray): Input values for creating the tree.

            Returns:
                IsolationNode or IsolationLeaf: Root node of the subtree.
        """

        # cut recursion when depth over height limit or if no data left in feature set
        if depth >= self.height_limit or input_values.shape[0] <= 1:
            self.ileaf_cnt += 1
            return IsolationLeaf(input_values.shape[0])

        # get random feature
        selected_feature = random.randint(0, input_values.shape[1]-1)
        #get random value into the selected feature to use as a split point
        split_value = np.random.uniform(min(input_values[:, selected_feature]), max(input_values[:, selected_feature]))

        # create inode with current value and its corresponding observation id
        # feature name and observation id not needed in this case
        inode = IsolationNode(None,
                              selected_feature,
                              split_value,
                              input_values.shape[0],
                              None)
        self.inode_cnt += 1

        #split the feature sets to give to left/right child
        left_values = input_values[input_values[:, selected_feature] < split_value]
        right_values = input_values[input_values[:, selected_feature] >= split_value]

        #create recusively left/right subtree
        inode.set_left_child(self.__create(depth + 1, left_values))
        inode.set_right_child(self.__create(depth + 1, right_values))

        return inode