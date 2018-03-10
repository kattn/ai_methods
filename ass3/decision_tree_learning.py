import Importance


'''
examples = [ ((attributes), classification) ]     [((0, 0, 1, 1, 0, 0, 0) , 1)]
attributes = (attribute_indecies)                 [0,1,2,3,4,5,6]
parent_examples = [ ((attributes), classification) ]
importance, the choosen importance function
'''
def decision_tree_learning(examples, attributes, parent_examples, importance):

    # If there are no examples return the plurality value
    if len( examples ) == 0:
        # print("No examples")
        return plurality_value(parent_examples)

    # If all examples have the same classification, return that
    for example in examples:
        if example[1] != examples[0][1]:
            break;
    else: # Will only run if the above for loop didn't break
        # print("All examples of same class")
        return examples[0][1]

    # If there are no attributes
    if len( attributes ) == 0:
        # print("No attributes")
        return plurality_value(examples)

    attr = max([(importance(attributes[a], examples),a) for a in range(len(attributes))]
               , key=(lambda x: x[0]))[1]
    # tree[attr] has index of 0 and 1
    tree = {attr: [None, None]}
    for vk in range(attributes[attr]):
        exs = [e for e in examples if e[0][attr] == vk]
        exclude_attr = list(attributes)
        del exclude_attr[attr]
        sub_tree = decision_tree_learning(exs, exclude_attr, examples, importance)
        tree[attr][vk] = sub_tree

    return tree


# Selects the most commom output value among a set of examples
def plurality_value(examples):
    sum_zeros = 0
    sum_ones = 0
    for ex in examples:
        if ex[1] == 1:
            sum_ones += 1
        else:
            sum_zeros += 1
    return max(sum_ones, sum_zeros)


def get_data(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip().split("\t")
            line = [ int(element)-1 for element in line ]
            example = tuple(line[:-1]), line[-1]
            examples.append( (example[0], example[1]) )
    return examples


if __name__ == "__main__":
    init_examples = get_data("data/training.txt")

    tree = decision_tree_learning( init_examples,
                            [2]*len( init_examples[0][0] ),
                            [],
                            Importance.importance_ran)
    print(tree)
