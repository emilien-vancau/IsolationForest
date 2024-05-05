import math
import statistics as stats
import time

import numpy as np
from pandas import DataFrame
from sklearn import metrics

from src.models.feature.Feature import Feature
from src.models.feature.FeaturesSet import FeaturesSet
from src.models.isolationForest.IsolationLeaf import IsolationLeaf
from src.models.isolationForest.IsolationNode import IsolationNode
from src.models.isolationForest.IsolationTree import IsolationTree
from src.models.sample.Sample import Sample


class IsolationForest(object):

    def __init__(self, input_data, tree_nb, subsampling_size):
        self.input_dataframe = input_data  # store original dataframe
        self.input_features_set = None  # converted data used for building the forest in experimental mode
        self.tree_nb = tree_nb  # number of trees inside the forest
        self.height_limit = 0  # height limit of trees
        self.subsampling_size = subsampling_size  # subsampling size of the forest
        self.tree_list = []  # store all the trees composing the forest
        self.training_time = 0
        self.evaluation_time = 0
        self.total_nodes_nb = 0

    def init(self):
        """
            Initialize the input features for the dataset.

            This function iterates through the columns of the input dataframe, creating Feature objects
            for each column and populating them with samples from the dataframe. Each sample consists of
            an observation index and the corresponding value in the column. The Feature objects are sorted
            based on their values, and then added to the input_features_set.

            Returns:
                None
        """

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
        """
            Fit the random forest model to the dataset.

            Args:
                is_experimental (bool, optional): Flag indicating whether to use experimental initialization.
                    Defaults to False.

            Returns:
                None
        """

        training_start_time = time.time()

        # check if data exists
        if self.input_dataframe.empty:
            print("no data available")
            return None

        # check subsampling size
        if self.subsampling_size > self.input_dataframe.shape[0]:
            print(
                f"not enought data to set subsampling size to {self.subsampling_size}. Subsampling size set to max dataframe size")
            self.subsampling_size = self.input_dataframe.shape[0]

        # forest initialization
        if is_experimental:
            self.init()

        # compute trees height limit, round up the log2 of sumbsampling
        self.height_limit = math.ceil(math.log2(self.subsampling_size))

        # extract numpy array that contains all the values of the dataframe
        input_values = self.input_dataframe.values

        self.__create_forest(input_values)

        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time

    def __create_forest(self, input_values):
        """
            Create a forest of isolation trees.

            Args:
                input_values (numpy.ndarray): Input values from the dataframe.

            Returns:
                None
        """

        for i in range(0, self.tree_nb):
            # subsample the whole dataset
            sub_index = np.random.choice(input_values.shape[0], self.subsampling_size, replace=False)
            sub_values = input_values[sub_index]

            # create a new tree and add it to the forest
            tree = IsolationTree(sub_values, self.height_limit)
            self.tree_list.append(tree)
            self.total_nodes_nb += tree.inode_cnt + tree.ileaf_cnt

    def score(self, dataset: DataFrame, height_limit: int = 0):
        """
            Score the dataset for anomalies using the isolation forest.

            Args:
                dataset (DataFrame): Dataset to be scored for anomalies.
                height_limit (int, optional): Height limit for the trees. If set to 0, uses the default
                    height limit of the forest. Defaults to 0.

            Returns:
                numpy.ndarray: An array of anomaly scores for each data point in the dataset.
        """

        if dataset is None:
            print("no data to evaluate")
            return

        if height_limit == 0:
            height_limit = self.height_limit
            print(f"height limit parameter set to tree height limit as default value")

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
        """
            Compute anomaly scores based on mean path lengths.

            Args:
                pathlengths_mean (list): List of mean path lengths for each data point.

            Returns:
                list: Anomalies scores computed based on the mean path lengths.
        """
        anomalies_scores = []
        for pathlength in pathlengths_mean:
            anomalies_scores.append(self.s(pathlength))
        return anomalies_scores

    def __mean_tree_length(self, dataset, height_limit):
        """
            Compute the mean path length for each observation in the dataset across all trees.

            Args:
                dataset (DataFrame): Dataset containing observations.
                height_limit (int): Height limit for the trees.

            Returns:
                list: List of mean path lengths for each observation.
        """

        # compute pathlenght for each observation inside each tree,
        # then compute mean for one observation for each tree
        if self.tree_list is not None:
            pathlengths_mean = []
            for row in dataset.values:
                pathlength_row = []
                for tree in self.tree_list:
                    # pathlength = self.__original_path_length(row, tree.root, height_limit, 1)
                    pathlength = self.__path_length(row, tree.root, height_limit)
                    pathlength_row.append(pathlength)
                pathlengths_mean.append(stats.mean(pathlength_row))
            return pathlengths_mean

    def __path_length(self, instance, inode, height_limit):
        """
            Compute the path length for a given instance in the tree.

            Args:
                instance (list or array-like): Instance for which path length is computed.
                inode (IsolationNode): Current node being evaluated in the tree.
                height_limit (int): Height limit for the tree traversal.

            Returns:
                int: The path length for the given instance.
        """

        depth = 0
        while isinstance(inode, IsolationNode) and depth <= height_limit:
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
        """
            Compute the threshold value based on the given percentile of scores.

            Args:
                scores (array-like): An array of anomaly scores.
                percentile_percentage (float): The percentile value (in percentage) to compute the threshold.

            Returns:
                float: The threshold value computed based on the given percentile.
        """

        sorted_scores = np.sort(scores)[::-1]
        threshold_index = int(percentile_percentage / 100 * len(sorted_scores))
        return sorted_scores[threshold_index]

    def tpr_threshold(self, y, scores, desired_tpr=.9):
        """
            Compute the threshold value based on the desired true positive rate (TPR).

            Args:
                y (array-like): True labels.
                scores (array-like): An array of anomaly scores.
                desired_tpr (float, optional): Desired true positive rate. Defaults to 0.9.

            Returns:
                tuple: A tuple containing the threshold value, achieved TPR, and corresponding FPR.
                    If the desired TPR cannot be reached, returns None.
        """

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
        """
            Predict anomalies based on the given anomaly scores and threshold.

            Args:
                scores (array-like): An array of anomaly scores.
                threshold (float): Threshold value for classifying anomalies.

            Returns:
                numpy.ndarray: An array of binary labels indicating anomalies (1) and non-anomalies (0).
        """

        return np.array([1 if score > threshold else 0 for score in scores])

    def get_parameters(self):
        """
            Get parameters of the isolation forest model.

            Returns:
                dict: A dictionary containing various parameters of the model.
        """

        return {
            "model name": "iForest",
            "training dataset size": self.input_dataframe.shape[0],
            "features size": self.input_dataframe.shape[1],
            "trees number": self.tree_nb,
            "total nodes number": self.total_nodes_nb,
            "height limit": self.height_limit,
            "subsampling size": self.subsampling_size,
            "training time": self.training_time,
            "evaluation time": self.evaluation_time
        }

    def c(self, size):
        """
            Compute the adjustment factor c.

            Args:
                size (int): Size of the node.

            Returns:
                float: The adjustment factor c.
        """

        if size > 2:
            return 2 * (math.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size
        if size == 2:
            return 1
        return 0

    def s(self, path_length):
        """
            Compute the anomaly score s.

            Args:
                path_length (int): Path length.

            Returns:
                float: The anomaly score s.
        """

        return math.pow(2, -path_length / self.c(self.subsampling_size))
