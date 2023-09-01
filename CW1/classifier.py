# classifier.py
# Lin Li/26-dec-2021
# Daniel Van Cuylenburg/24-feb-2022
#
# A node class and classifier class representing the ID3 algorithm (decision tree learning algorithm).

import pandas as pd
import numpy as np

# This simple Node class represents a node of a (decision) tree.
# 
# self.value = Holds the value of the node.
# self.children = A list of each of the nodes children.
# self.leaf = A boolean value, true if the node is a leaf, false otherwise.
# self.outcome = If the node is a leaf, stores an encoded action, otherwise an empty string;
# '0', '1', '2', or '3', representing a direction for the pacman to move in.
class Node:
    def __init__(self):
        self.value = ""
        self.children = []
        self.leaf = False
        self.outcome = ""

# This class represents an ID3 classifier based on the decision tree learning methodology.
class Classifier:
    
    # Runs the 'self.reset()' function when initialized.
    def __init__(self):
        self.reset()

    # Sets the variable 'columnLabels'. This function is run between each pacman game, once the pacman dies.
    # 
    # self.columnLabels = A list of 26 strings, used as column names when constructing a pandas dataframe.
    # Each string (column) represents a piece of data about the state of the map around the pacman, as well as its encoded action.
    def reset(self):
        self.columnLabels = ['wall1', 'wall2', 'wall3', 'wall4',
                             'food1', 'food2', 'food3', 'food4',
                             'ghost1.1', 'ghost1.2', 'ghost1.3', 'ghost1.4', 'ghost1.5', 'ghost1.6', 'ghost1.7', 'ghost1.8',
                             'ghost2.1', 'ghost2.2', 'ghost2.3', 'ghost2.4', 'ghost2.5', 'ghost2.6', 'ghost2.7', 'ghost2.8',
                             'visible',
                             'target']
    
    # Creates a pandas dataframe, inputs the training data, runs the 'self.ID3()' function,
    # stores the result in self.tree. Runs once before the start of the game.
    # 
    # data = A list of lists of integers: either '0' or '1'. Feature vectors. Represents the training data.
    # target = List of integers: either '0', '1', '2', or '3'. Encodes actions.
    # The ith elements of data and target go together.
    #
    # df = pandas dataframe to represent the training data.
    # dict = Dictionary used to match each columns label to its respective datapoint for each feature vector.
    # self.tree = The root node of the decision tree, returned by the 'self.ID3()' function.
    def fit(self, data, target):
        df = pd.DataFrame(columns = self.columnLabels)
        for row in range(len(data)):
            dict = {}
            for column in range(len(data[0])):
                dict[self.columnLabels[column]] = [data[row][column]]
            dict['target'] = target[row]
            df = pd.concat([df, pd.DataFrame(dict, index = [row])])

        self.columnLabels.remove('target')
        self.tree = self.ID3(df, self.columnLabels)
        
    # Calculates the entropy of a probability distribution.
    # 
    # examples = (Sub)set of the training data (feature vectors with encoded actions) represented by a pandas dataframe.
    # 
    # outcomes = List of four integers, each representing how many of each encoded action is present in the 'examples' set.
    # This list maps to the actions '0', '1', '2', and '3' respectively.
    # outcomeChance = Probability distribution for the actions in the 'outcomes' set, represented as a list of four floats.
    # total = Weighted average entropy.
    def Entropy(self, examples):
        # Counts how many of each encoded actions are present in the 'examples' set.
        outcomes = [0, 0, 0, 0]
        for _, row in examples.iterrows():
            for i in range(len(outcomes)):
                if row['target'] == i: outcomes[i] += 1
        
        # Each action's count is divided by the total number of actions.
        outcomeChance = []
        for outcome in outcomes:
            outcomeChance.append(outcome / sum(outcomes))
        
        # Applies the entropy formula to the probability distribution 'outcomeChance'.
        total = 0
        for chance in outcomeChance:
            if chance != 0:
                total += chance * np.log2(chance)
        return -(total)

    # Calculates the Information gain - that being the difference in entropy from before to after
    # the set 'examples' is split on the 'attribute'.
    # 
    # examples = (Sub)set of the training data (feature vectors and encoded actions) represented by a pandas dataframe.
    # attribute = An attribute, as a string, that the dataset will be split by.
    # 
    # uniquePossible = A list of all unique possible values present in the 'attribute' column of the dataframe.
    # Either [0], [1], or [0, 1].
    # gain = Information gain, as a float.
    # subset = Set split by the 'attribute'.
    # subsetEntropy = Entropy of the subset.
    def InformationGain(self, examples, attribute):
        uniquePossible = np.unique(examples[attribute])
        gain = self.Entropy(examples)
        for value in uniquePossible:
            subset = examples[examples[attribute] == value]
            subsetEntropy = self.Entropy(subset)
            gain -= (float(len(subset)) / float(len(examples))) * subsetEntropy
        return gain

    # Implementation of the ID3 algorithm. Recursive function that generates and returns a decision tree.
    # 
    # examples = (Sub)set of the training data (feature vectors and encoded actions) represented by a pandas dataframe.
    # attributes = A list of strings representing the data's attributes.
    # 
    # tree = Root node of the current tree.
    # maxGain = The maximum gain if we were to split the training data by the current 'best' attribute.
    # best = Best attribute to split the data by.
    # gain = The information gain of the 'examples' split by the current attribute.
    # uniquePossible = A list of all unique possible values present in the 'best' attribute column of the dataframe.
    # Either [0], [1], or [0, 1].
    # subset = Set split by the 'best' attribute.
    # subTree = For each 'uniquePossible', a branch of the tree is created. This is the root node of the branch.
    # remain = List of attributes not including the 'best' attribute.
    def ID3(self, examples, attributes):
        tree = Node()
        
        # If the 'attributes' list is empty, returns the leaf node with the 'tree.outcome' set using a plurality vote.
        # The most common encoded action is set as the 'tree.outcome'.
        if attributes == []:
            # Counts each unique encoded action, then selects the action with the the highest count.
            tree.outcome = [examples['target'].value_counts().idxmax()] 
            tree.leaf = True
            return tree
        
        # Calls the function 'self.InformationGain()' for each attribute. If the attribute has a higher information gain
        # than the current 'maxGain', then this attribute becomes the 'best' attribute. If there is no best attribute,
        # the first attribute from the 'attributes' list is used.
        maxGain = 0
        best = attributes[0]
        for feature in attributes:
            gain = self.InformationGain(examples, feature)
            if gain > maxGain:
                maxGain = gain
                best = feature
        tree.value = best
        
        # For each 'uniquePossible', the tree branches into subtrees.
        uniquePossible = np.unique(examples[best])
        for value in uniquePossible:
            subset = examples[examples[best] == value]
            subTree = Node()
            subTree.value = value
            # If the entropy of the 'subset' is 0, (i.e. if all examples in the subset have the same classification/encoded action)
            # sets the subtree as a leaf, and sets the subtree's 'outcome' as the encoded action.
            if self.Entropy(subset) == 0:
                subTree.leaf = True
                subTree.outcome = np.unique(subset['target'])
            # Otherwise, calls 'self.ID3()' with the 'subset' and remaining attributes,
            # and appends the result to the list of the subtrees children.
            else:
                remain = attributes[:]
                remain.remove(best)
                subTree.children.append(self.ID3(subset, remain))
                
            # Appends the 'subtree' to the list of the trees children.
            tree.children.append(subTree)
        return tree
    
    # Parses through the generated decision tree until an encoded action is reached and returned.
    # 
    # data = Feature vector for the current state of the map around pacman.
    # legal = A list of legal directions the pacman can move in.
    # 
    # dict = Dictionary used to match each columns label to its respective datapoint in the feature vector.
    # currentNode = The current node during parsing, initially set to the root node of the decision tree ('self.tree').
    def predict(self, data, legal=None):
        dict = {}
        for column in range(len(data)):
            dict[self.columnLabels[column]] = data[column]
        
        # Parses through each node in the tree until a leaf is reached.
        currentNode = self.tree
        while not currentNode.leaf:
            if len(currentNode.children) == 1:
                currentNode = currentNode.children[0]
            else:
                for child in currentNode.children:
                    if child.value == dict[currentNode.value]:
                        currentNode = child
                        break
        
        # Returns the encoded action of the reached leaf node.
        return currentNode.outcome[0]
