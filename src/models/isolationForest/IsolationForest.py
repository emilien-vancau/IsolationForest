import math

from src.models.feature.Feature import Feature
from src.models.feature.FeaturesSet import FeaturesSet
from src.models.isolationForest.IsolationLeaf import IsolationLeaf
from src.models.isolationForest.IsolationTree import IsolationTree
from src.models.sample.Sample import Sample


class IsolationForest(object):

    def __init__(self, input_data, nb_tree, subsampling_size):
        self.input_data = input_data #store original dataframe
        self.nb_tree = nb_tree #number of trees inside the forest
        self.subsampling_size = subsampling_size #subsampling size of the forest
        self.input_features_set = None #converted data used for building the forest
        self.tree_list = [] #store all the trees composing the forest

    def init(self):
        """convert input dataframe to features set and sort them by ascending order"""
        input_features = []

        # get dataframe index values
        obs_indexes = self.input_data.index.values

        for column in self.input_data.columns:
            # get current feature set
            feature_column = self.input_data[column].tolist()
            feature = Feature(column)
            for obs in range(0, len(obs_indexes)):
                # create sample and add it to the corresponding feature set
                sample = Sample(obs_indexes[obs], feature_column[obs])
                feature.add_sample(sample)

            feature.sort_value_set()
            input_features.append(feature)

        self.input_features_set = FeaturesSet(len(obs_indexes), input_features)

    def create(self):
        # forest initialization
        self.init()

        # check if data exists
        if len(self.input_features_set.features_set) < 1:
            print("no data inside the forest")
            return None

        # calculate trees height limit
        # round up the log2 of sumbsampling
        height_limit = math.ceil(math.log2(self.subsampling_size))

        for i in range(0, self.nb_tree):
            # select random feature set and create a new tree
            tree = IsolationTree(self.input_features_set, height_limit)
            self.tree_list.append(tree)

    def path_length(self, instance, inode, height_limit, current_pathlength):
        if current_pathlength > height_limit or isinstance(inode, IsolationLeaf):
            return current_pathlength + self.c(inode.size)
        if instance[inode.feature_index] < inode.split_value:
            return self.path_length(instance, inode.left_child, height_limit, current_pathlength+1)
        else:
            return self.path_length(instance, inode.right_child, height_limit, current_pathlength+1)

    def score(self, dataset, height_limit):
        if self.tree_list is not None:
            for row in dataset.values:
                path_row = []
                for tree in self.tree_list:
                    pathlength = self.path_length(row, tree.root, height_limit, 1)
                    path_row.append(pathlength)
                print("t")

    def c(self, size):
        if size > 2:
            return 2 * (math.log2(size - 1) + 0.5772156649) - 2 * (size - 1) / size
        if size == 2:
            return 1
        return 0
