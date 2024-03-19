class IsolationNode(object):

    def __init__(self, feature_name, feature_index, split_value, observation_id):
        self.feature_name = feature_name #name of the selected feature
        self.feature_index = feature_index #index of the selected feature
        self.split_value = split_value #split index in the feature list
        self.observation_id = observation_id #id of sample attach to the value
        self.left_child = None #left subtree
        self.right_child = None #right subtree

    def set_left_child(self, inode):
        self.left_child = inode

    def set_right_child(self, inode):
        self.right_child = inode

    # def is_leaf(self):
    #     return self.left_child is None and self.right_child is None
