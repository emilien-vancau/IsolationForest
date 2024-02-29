from src.models.feature.Feature import Feature
from src.models.sample.Sample import Sample


class IsolationForest(object):

    def __init__(self, input_data, nb_tree, subsampling_size):
        self.input_data = input_data
        self.nb_tree = nb_tree
        self.subsampling_size = subsampling_size
        self.input_features = []

    def init(self):
        """convert input dataframe to features set"""
        #get df index values
        obs_indexes = self.input_data.index.values

        for column in self.input_data.columns:
            #get current feature set
            feature_column = self.input_data[column].tolist()
            feature = Feature(column)
            for obs in range(0, len(obs_indexes)):
                #create sample and add it to the corresponding feature set
                sample = Sample(obs_indexes[obs], feature_column[obs])
                feature.add_sample(sample)

            feature.sort_value_set()
            self.input_features.append(feature)


    def create(self):
        pass