class FeaturesSet(object):

    def __init__(self, features_size, features_set):
        self.size = features_size
        self.features_set = features_set

    # def max_size(self):
    #     #find the maximum size for each feature set, use for the IsolationLeaf node
    #     #search the maximum feature size to handle NaN values
    #     for feature in self.features_set:
    #         if len(feature.samples_set) > self.size:
    #             self.size = len(feature.samples_set)
    #     print(self.size)