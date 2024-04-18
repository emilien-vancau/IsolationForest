import math
import statistics as stats
import time

import numpy as np
from sklearn import metrics

from src.models.feature.Feature import Feature
from src.models.feature.FeaturesSet import FeaturesSet
from src.models.isolationForest.IsolationLeaf import IsolationLeaf
from src.models.isolationForest.IsolationNode import IsolationNode
from src.models.isolationForest.IsolationTree import IsolationTree
from src.models.sample.Sample import Sample


class IsolationForest(object):

    def __init__(self, input_data, nb_tree, subsampling_size):
        self.input_dataframe = input_data #store original dataframe
        self.input_features_set = None  # converted data used for building the forest in experimental mode
        self.nb_tree = nb_tree #number of trees inside the forest
        self.height_limit = 0 #height limit of trees
        self.subsampling_size = subsampling_size #subsampling size of the forest
        self.tree_list = [] #store all the trees composing the forest
        self.training_time = 0
        self.evaluation_time = 0

    def init(self):
        """convert input dataframe to features set and sort them by ascending order"""
        input_features = []

        # get dataframe index values
        obs_indexes = self.input_dataframe.index.values

        for column in self.input_dataframe.columns:
            # get current feature set
            feature_column = self.input_dataframe[column].tolist()
            feature = Feature(column)
            for obs in range(0, len(obs_indexes)):
                # create sample and add it to the corresponding feature set
                sample = Sample(obs_indexes[obs], feature_column[obs])
                feature.add_sample(sample)

            feature.sort_value_set()
            input_features.append(feature)

        self.input_features_set = FeaturesSet(len(obs_indexes), input_features)

    def fit(self, is_experimental=False):
        '''

        :param is_experimental:
        :return:
        '''

        training_start_time = time.time()

        # check if data exists
        if self.input_dataframe.empty:
            print("no data available")
            return None

        # check subsampling size
        if self.subsampling_size > self.input_dataframe.shape[0]:
            print(f"not enought data to set subsampling size to {self.subsampling_size}. Subsampling size set to max dataframe size")
            self.subsampling_size = self.input_dataframe.shape[0]

        # forest initialization
        if is_experimental:
            self.init()

        # compute trees height limit, round up the log2 of sumbsampling
        self.height_limit = math.ceil(math.log2(self.subsampling_size))

        #extract numpy array that contains all the values of the dataframe
        input_values = self.input_dataframe.values

        self.__create_forest(input_values)

        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time

    def __create_forest(self, input_values):
        '''

        :param input_values:
        :return:
        '''

        for i in range(0, self.nb_tree):
            # subsample the whole dataset
            sub_index = np.random.choice(input_values.shape[0], self.subsampling_size, replace=False)
            sub_values = input_values[sub_index]

            # create a new tree and add it to the forest
            tree = IsolationTree(sub_values, self.height_limit)
            self.tree_list.append(tree)

    def score(self, dataset, height_limit):
        '''

        :param dataset:
        :param height_limit:
        :return:
        '''

        if height_limit > self.height_limit:
            height_limit = self.height_limit
            print(f"height limit is above tree height. Use of tree height as height limit")
        print(f"height limit set to: {height_limit}")

        path_start_time = time.time()
        pathlengths_mean = self.__mean_tree_length(dataset, height_limit)
        path_end_time = time.time()
        total_path_time = path_end_time - path_start_time
        print(f"Mean Pathlength compute in: {total_path_time} seconds")

        score_start_time = time.time()
        anomalies_scores = self.__anomaly_scoring(pathlengths_mean)
        score_end_time = time.time()
        total_score_time = score_end_time - score_start_time
        print(f"Anomalies scoring compute in: {total_score_time}")

        self.evaluation_time = total_score_time + total_path_time

        return np.array(anomalies_scores)

    def __anomaly_scoring(self, pathlengths_mean):
        anomalies_scores = []
        for pathlength in pathlengths_mean:
            anomalies_scores.append(self.s(pathlength))
        return anomalies_scores

    def __mean_tree_length(self, dataset, height_limit):
        '''

        :param dataset:
        :param height_limit:
        :return:
        '''
        # compute pathlenght for each observation inside each tree,
        # then compute mean for one observation for each tree
        if self.tree_list is not None:
            pathlengths_mean = []
            for row in dataset.values:
                pathlength_row = []
                for tree in self.tree_list:
                    #pathlength = self.original_path_length(row, tree.root, height_limit, 1)
                    pathlength = self.__path_length(row, tree.root)
                    pathlength_row.append(pathlength)
                pathlengths_mean.append(stats.mean(pathlength_row))
            return pathlengths_mean

    def __path_length(self, instance, inode):
        '''

        :param instance:
        :param inode:
        :return:
        '''
        depth = 0
        while isinstance(inode, IsolationNode):
            if instance[inode.feature_index] < inode.split_value:
                inode = inode.left_child
            else:
                inode = inode.right_child
            depth += 1
        return depth + self.c(inode.size)


    def __original_path_length(self, instance, inode, height_limit, current_pathlength):
        """
        Original pathlength function of Forest Isolation. This method is deprecated due to an higher execution time.
        Because of the use of recursion in pathlength, this can lead to take more time with higher dimensionnal data

        :param instance: one observation from evaluation dataset
        :param inode: an isolation node from the tree
        :param height_limit: max depth for the recursion
        :param current_pathlength: current depth of the recursion

        :return: pathlength of an instance
        """
        if current_pathlength > height_limit or isinstance(inode, IsolationLeaf):
            return current_pathlength + self.c(inode.size)
        if instance[inode.feature_index] < inode.split_value:
            return self.__original_path_length(instance, inode.left_child, height_limit, current_pathlength + 1)
        else:
            return self.__original_path_length(instance, inode.right_child, height_limit, current_pathlength + 1)

    # def path_length_experimental(self, instance, inode, height_limit, current_pathlength):
    #     if current_pathlength > height_limit or isinstance(inode, IsolationLeaf):
    #         return current_pathlength + self.c(inode.size)
    #     if instance[inode.feature_index] < inode.split_value: #and instance.observation_id
    #         return self.original_path_length(instance, inode.left_child, height_limit, current_pathlength + 1)
    #     else:
    #         return self.original_path_length(instance, inode.right_child, height_limit, current_pathlength + 1)

    def percentile_threshold(self, scores, percentile_percentage):
        '''

        :param scores:
        :param percentile_percentage:
        :return:
        '''
        sorted_scores = np.sort(scores)[::-1]
        threshold_index = int(percentile_percentage / 100 * len(sorted_scores))
        return sorted_scores[threshold_index]

    def tpr_threshold(self, y, scores, desired_tpr):
        '''

        :param y:
        :param scores:
        :param desired_tpr:
        :return:
        '''

        threshold = 1.0
        while threshold >= 0:
            prediction = self.predict_anomalies(scores, threshold)
            confusion_matrix = metrics.confusion_matrix(y, prediction)
            tn, fp, fn, tp = confusion_matrix.flat
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            if tpr >= desired_tpr:
                return threshold, tpr, fpr

            threshold -= 0.01
        print("unable to reach desired TPR")
        return


    def predict_anomalies(self, scores, threshold):
        '''

        :param scores:
        :param threshold:
        :return:
        '''

        return np.array([1 if score > threshold else 0 for score in scores])

    def c(self, size):
        if size > 2:
            return 2 * (math.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size
        if size == 2:
            return 1
        return 0

    def s(self, path_length):
        return math.pow(2, -path_length/self.c(self.subsampling_size))