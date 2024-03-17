class Feature(object):

    def __init__(self, name):
        self.name = name
        self.samples_set = []

    def add_sample(self, sample):
        self.samples_set.append(sample)

    def add_samples(self, sample_set):
        self.samples_set = sample_set

    def sort_value_set(self):
        self.samples_set.sort(key=lambda x: x.value)

    def split_samples(self, split_index):
        left_features = Feature(self.name)
        right_features = Feature(self.name)
        left_features.add_samples(self.samples_set[:split_index])
        right_features.add_samples(self.samples_set[split_index:])

        left_size = len(left_features.samples_set)
        right_size = len(right_features.samples_set)

        return left_features, right_features, left_size, right_size
