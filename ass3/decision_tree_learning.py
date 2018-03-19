import Importance


'''
examples = [ ((attributes), classification) ]     [((0, 0, 1, 1, 0, 0, 0) , 1)]
attributes = (attribute_indecies)                 [0,1,2,3,4,5,6]
parent_examples = [ ((attributes), classification) ]
importance, the choosen importance function
'''

class Attr:
    def __init__(self, id, val=-1):
        self.id = id
        self.val = val
        self.pos_values = [1,2]


class Case:
    def __init__(self, attributes, target):
        self.attributes = []
        for i in range(len(attributes)):
            self.attributes.append(Attr(i, attributes[i]))
        self.target = target


class Node:
    def __init__(self, attr):
        self.attr = attr
        self.child = {}
        self.classification = 0

    def __str__(self):
        # Formats a string to be used at http://mshang.ca/syntree/
        if self.child:
            children = "{:s}]".format("".join(
                        str(self.child[k]) for k in self.child.keys()))
            return ("[{:d} ".format(self.attr.id) + children)


        else:
            return "[{:d}]".format(self.attr.id)

    def addChild(self, vk, node):
        if(vk in self.attr.pos_values):
            self.child[vk] = node
        else:
            raise Exception("Trying to add an invalid node")


def decision_tree_learning(examples, attributes, parent_examples, importance):
    # If there are no examples return the plurality value
    if len( examples ) == 0:
        # print("No examples")
        return plurality_value(parent_examples)

    # If all examples have the same classification, return that
    for example in examples:
        if example.target != examples[0].target:
            break;
    else: # Will only run if the above for loop didn't break
        # print("All examples of same class")
        return examples[0].target

    # If there are no attributes
    for attr in attributes:
        if type(attr) != int:
            break
    else:
        # print("No attributes")
        return plurality_value(examples)

    attr = max([(importance(attr, examples), attr) for attr in attributes if type(attr) != int]
               , key=(lambda x: x[0]))[1]

    # tree[attr] has index of 0 and 1
    tree = Node(attr)
    for vk in attr.pos_values:  #values 1 and 2
        exs = [e for e in examples if e.attributes[attr.id].val == vk]
        exclude_attr = list(attributes)
        exclude_attr[attr.id] = 0 #
        sub_tree = decision_tree_learning(exs, exclude_attr, examples, importance)
        tree.addChild(vk, sub_tree)

    return tree


# Selects the most commom output value among a set of examples
def plurality_value(examples):
    sum_ones = 0
    sum_twos = 0
    for ex in examples:
        if ex.target == 1:
            sum_ones += 1
        else:
            sum_twos += 1
    if sum_ones < sum_twos:
        return 2
    else:
        return 1


def get_data(path):
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip().split("\t")
            line = [ int(element) for element in line ]
            cases.append( Case(line[:-1], line[-1]) )
    return cases


def test_tree(node, examples):
    correct_classification = 0
    for example in examples:
        current = node
        debug_counter = 0
        while type(current) != int:
            current = current.child[example.attributes[current.attr.id].val]
            if debug_counter > 100:
                break
        if current == example.target:
            correct_classification += 1
    print("Accuracy:", correct_classification/len(examples))
    print("Correct:", correct_classification,"Total:",len(examples))


if __name__ == "__main__":
    init_examples = get_data("data/training.txt")
    test_data = get_data("data/test.txt")

    attrs = []
    for i in range(len(init_examples[0].attributes)):
        attrs.append(Attr(i))

    ran_tree = decision_tree_learning( init_examples,
                                   attrs,
                                   [],
                                   Importance.importance_ran)

    print("random importance")
    print(ran_tree)
    test_tree(ran_tree, test_data)

    exp_tree = decision_tree_learning( init_examples,
                                       attrs,
                                       [],
                                       Importance.importance_exp)

    print("expected importance")
    print(exp_tree)
    test_tree(exp_tree, test_data)
