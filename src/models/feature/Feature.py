
class Feature(object):

    def __init__(self, name):
        self.name = name
        self.samples_set = []

    def add_sample(self, sample):
        self.samples_set.append(sample)

    def sort_value_set(self):
        self.samples_set.sort(key=lambda x: x.value)