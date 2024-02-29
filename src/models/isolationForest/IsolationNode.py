
class IsolationNode(object):

    def __init__(self, value, split_value, observation_id, left_child, right_child):
        self.value = value #data value from de the feature list
        self.split_value #split index in the feature list
        self.observation_id = observation_id #id of sample attach to the value
        self.left_child = left_child #left subtree
        self.right_child = right_child #right subtree