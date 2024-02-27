
class Feature(object):

    def __init__(self, name, values_set):
        self.name = name
        self.values_set = values_set

    def sort_value_set(self):
        self.values_set.sort()