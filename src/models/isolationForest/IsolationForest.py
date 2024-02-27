from src.models.feature.Feature import Feature

class IsolationForest(object):

    def __init__(self, input_data, nb_tree, subsampling_size):
        self.input_data = input_data
        self.nb_tree = nb_tree
        self.subsampling_size = subsampling_size
        self.input_features = []

    def init(self):
        for column in self.input_data.columns:
            feature = Feature(column, self.input_data[column].tolist())
            feature.sort_value_set()
            self.input_features.append(feature)



    def create(self):
        pass