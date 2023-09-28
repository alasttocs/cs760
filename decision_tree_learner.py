import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    data.columns = ["x1", "x2", "y"]
    return data


class Node:
    def __init__(self):
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
    label_prob_1 = label_count_1 / total_count
    entropy = -label_prob_0 * \
        np.log2(label_prob_0)-label_prob_1*np.log2(label_prob_1)
    return entropy


def info_gain_ratio(parent_entropy, left, right):
    total = left.shape[0] + right.shape[0]

    left_entropy = entropy(left)
    right_entropy = entropy(right)

    e_prob_left = left.shape[0] / total
    e_prob_right = right.shape[0] / total

    info_gain = parent_entropy - \
        (e_prob_left * left_entropy + e_prob_right * right_entropy)
    data = pd.concat([left, right], ignore_index=True)
    # print(f"info_gain here: {info_gain} ")
    if entropy(data) == 0:
        return 0
    info_gain_ratio = info_gain / entropy(data)
    return info_gain_ratio


def find_best_split(data, c):
    max_info_gain_ratio = 0
    best_feature = None
    best_threshold = None

    parent_entropy = entropy(data)

    for candidate in c:
        feature, threshold = candidate
        left_data = data[data[feature] >= threshold]
        right_data = data[~(data[feature] >= threshold)]
        info_gain = info_gain_ratio(
            parent_entropy, left_data, right_data)

        # TODO: do i need to check data here?

        # print(f"{feature},{threshold},{info_gain}")

        if info_gain > max_info_gain_ratio:
            max_info_gain_ratio = info_gain
            best_feature = feature
            best_threshold = threshold
    return best_feature, best_threshold


def determine_candidate_splits(data):
    c = []
    num_rows = data.shape[0]
    for feature in data.columns[:-1]:
        data = data.sort_values(by=feature)
        for value in data[feature].unique():
            c.append([feature, value])
    # print(c)
    return c


def make_subtree(data):
    c = determine_candidate_splits(data)
    # print("here is the first return of determine candidate split ------")
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
        if best_feature is None or best_threshold is None:
            leaf_node = Node()
            leaf_node.label = data.iloc[0]["y"]
            return leaf_node
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


def print_tree(node, depth=0, prefix="Root: "):
    if node is not None:
        print("-----" * depth + prefix, end="")

        if node.label is not None:
            print(f"Label {node.label}")
        else:
            print(f"Feature {node.feature} >= {node.threshold}")

            print_tree(node.left, depth + 1, ">If True: ")
            print_tree(node.right, depth + 1, ">If False: ")


def plot_data(data, title):

    x1 = data['x1'].tolist()
    x2 = data['x2'].tolist()
    y = data['y'].tolist()
    # Create a scatter plot
    colors = ['blue' if val == 1 else 'red' for val in y]
    plt.scatter(x1, x2, c=colors, s=3, label='Data Points')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.title(title + " Plot")

    # Save the file
    plt.savefig('D2.png')
    # Show the plot
    plt.show()


def predict(node, instance):
    # for use in decision boundry
    if node.label is not None:
        return node.label
    feture_index = 0
    if node.feature == "x2":
        feture_index = 1
    if instance[feture_index] >= node.threshold:
        return predict(node.left, instance)
    else:
        return predict(node.right, instance)


def plot_decision_boundary(node, x1_range, x2_range, title):
    # code credit for printing decision boundary to https://stackoverflow.com/questions/62119880/classifier-predict-in-python
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    Z = np.array([predict(node, np.array([x1, x2]))
                 for x1, x2 in zip(xx1.ravel(), xx2.ravel())])
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.8, cmap='RdYlBu')

    # Add labels and title
    plt.xlabel('Feature 1 (x1)')
    plt.ylabel('Feature 2 (x2)')
    plt.title(title + ' Decision Boundary')

    plt.savefig(title + '.png')
    plt.show()


def test_model(data_test, root, count):
    node_count = count_nodes(root)
    incorrect = 0
    total = 0
    for index, row in data_test.iterrows():
        test_instance = np.array(row)
        predicted_label = predict(root, test_instance)
        if test_instance[2] != predicted_label:
            incorrect += 1
        total += 1
    print(f"{count}\t{node_count}\t{incorrect/total}")


def sk_model(test_data, training_data, n):
    x_test = test_data[['x1', 'x2']]
    y_test = test_data['y']
    x_train = training_data[['x1', 'x2']]
    y_train = training_data['y']  # Assuming 'y' is your target variable
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    num = dtc.tree_.node_count
    predictions = dtc.predict(x_test)
    error_rate = 1 - accuracy_score(y_test, predictions)

    print(f"{n}\t{num}\t{error_rate}")


def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def main():
    if len(sys.argv) < 2:
        print("Incorrect ")
        sys.exit(1)

    file_path = sys.argv[1]
    data = load_data(file_path)

    # root = make_subtree(data)
    # print_tree(root)

    # Q 6
    # plot_data(data, file_path)
    # x1_range = np.linspace(0.0, 1.0, 500)
    # x2_range = np.linspace(0.0, 1.0, 500)
    # plot_decision_boundary(root, x1_range, x2_range)

    # Q7/Q8 (random_state=42)
    random_data = data.sample(frac=1, random_state=42)

    data_test = random_data.iloc[8192:, :]
    x_test = data_test[['x1', 'x2']]
    y_test = data_test['y']
    data_train = random_data.iloc[:8192, :]

    # x1_range = np.linspace(-1.5, 1.5, 500)
    # x2_range = np.linspace(-1.5, 1.5, 500)

    dt_32 = data_train.iloc[:32, :]
    sk_model(data_test, dt_32, 32)
    # root = make_subtree(dt_32)
    # test_model(data_test, root, 32)
    # plot_decision_boundary(root, x1_range, x2_range, "n32")

    dt_128 = data_train.iloc[:128, :]
    sk_model(data_test, dt_128, 128)
    # root = make_subtree(dt_128)
    # test_model(data_test, root, 128)
    # plot_decision_boundary(root, x1_range, x2_range, "n128")

    dt_512 = data_train.iloc[:512, :]
    sk_model(data_test, dt_512, 512)
    # root = make_subtree(dt_512)
    # test_model(data_test, root, 512)
    # plot_decision_boundary(root, x1_range, x2_range, "n512")

    dt_2048 = data_train.iloc[:2048, :]
    sk_model(data_test, dt_2048, 2048)
    # root = make_subtree(dt_2048)
    # test_model(data_test, root, 2048)
    # plot_decision_boundary(root, x1_range, x2_range, "n2048")

    dt_8192 = data_train.iloc[:8192, :]
    sk_model(data_test, dt_8192, 8192)
    # root = make_subtree(dt_8192)
    # test_model(data_test, root, 8192)
    # plot_decision_boundary(root, x1_range, x2_range, "n8192")


if __name__ == "__main__":
    main()
