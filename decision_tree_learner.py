import sys
import pandas as pd
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    data.columns = ["x1", "x2", "y"]
    print(data)
    return data


class Node:
    def __init(self):
        # Internal Nodes
        self.feature = None
        self.threshold = None
        self.left = None  # True branch
        self.right = None  # False branch
        # Leaf Nodes
        self.label = None


def entropy(data):
    total_count = data.shape[0]
    label_count_0 = (data["y"] == 0).sum()
    label_count_1 = (data["y"] == 1).sum()

    if label_count_0 == 0 or label_count_1 == 0:
        return 0

    label_prob_0 = label_count_0 / total_count
    label_prob_1 = label_count_0 / total_count
    entropy = -label_prob_0 * \
        np.log2(label_prob_0)-label_prob_1*np.log2(label_prob_1)
    return entropy


def info_gain(parent_entropy, left, right):
    left_entropy = entropy(left)
    right_entropy = entropy(right)


def find_best_split(data, c):
    max_info_gain = 0
    best_feature = None
    best_threshold = None

    parent_entropy = entropy(data)

    for candidate in c:
        feature, threshold, label = candidate
        left_data = data[data[feature] >= threshold]
        right_data = data[~(data[feature] >= threshold)]
        info_gain = info_gain(parent_entropy, left_data, right_data)

        # TODO: do i need to check data here?

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            best_threshold = threshold
    return best_feature, best_threshold


def determine_candidate_splits(data, feature):
    c = []
    # TODO: Loop features instead of taking the feature as arguement
    num_rows = data.shape[0]
    c_data = data[[feature, "y"]]
    c_data = c_data.sort_values(by=feature)
    print(c_data)
    for i in range(num_rows - 1):
        y = c_data.iloc[i]["y"]
        y_next = c_data.iloc[i+1]["y"]
        vj = c_data.iloc[i][feature]
        vj_next = c_data.iloc[i+1][feature]
        print(f"{i}:   {vj} - {y}")
        if y != y_next:
            c.append([feature, vj_next, y_next])
            print(f"split this{i}:   {vj_next} - {y_next}")

    print(c)
    return c


def make_subtree(data):
    c = determine_candidate_splits(data)
    # If Stopping criteria is met, make a leaf node
    # determine class/label probabilities for N
    # the node is empty
    if len(data) == 0:
        leaf_node = Node()
        leaf_node.label = 1
        return leaf_node
    # all splits have zero gain ratio (if the entropy of the split is non-zeroand/the entropy of any candidates split is zero
    # is handled when determining best split (true leaf nodes, are these handled by these two conditions - hw Q1)
    # TODO #1 see if this can be replaced or not once you call find_best splits or see if we need to add entropy/info gain to determine candidate split
    if len(c) == 0:
        leaf_node = Node()
        # TODO #2 there should be data and only one label i think related to TODO #1
        leaf_node.label = data.iloc[0]["y"]
        return leaf_node
    else:
        # make an internal node
        s = find_best_split(data, c)
        best_feature, best_threshold = s
        internal_node = Node()
        internal_node.feature = best_feature
        internal_node.threshold = best_threshold

        # for each outcome k of s
        left_data = data[data[best_feature] >= best_threshold]
        # ~ gives the opposite for a data frame
        right_data = data[~(data[best_feature] >= best_threshold)]
        internal_node.left = make_subtree(left_data)
        internal_node.right = make_subtree(right_data)
        return internal_node


def main():
    if len(sys.argv) < 2:
        print("Incorrect ")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_data(file_path)
    determine_candidate_splits(data, "x1")


if __name__ == "__main__":
    main()
