import numpy as np

# Implemting eq 15.12
def forward(T, O, msg):
    return normalize(O*T.T*msg)

# Implemting eq 15.13
def backward(T, O, msg):
    return T*O*msg


def filtering(T, O_model, init_message, observations):
    # Initilize the fv_msg array with the init_message
    fv_messages = [ init_message ]

    # For every observation, calculate the next msg
    for e in observations:
        msg = forward(T, O_model[e], fv_messages[-1])
        fv_messages.append(msg)

    return fv_messages


def smoothing(T, O_model, init_message, observations):
    # Initilize the fv_msg array with the init_message
    fv_messages = [init_message]

    # For every observation, calculate the next msg
    for e in observations:
        msg = forward(T, O_model[e], fv_messages[-1])
        fv_messages.append(msg)

    # Initilize the b_msg array with 1's and the same shape as init_message
    b_msg = [ np.ones_like(init_message) ]*( len(observations)+1 )
    # Initiate the sv array
    sv = [None]*len(observations)

    # For every observation, calculate the smoothing estimate and next b_msg
    for i in range(len( observations )-1, -1, -1):
        sv[i] = normalize(np.multiply(fv_messages[i+1], b_msg[i+1]))
        b_msg[i] = backward(T, O_model[observations[i]], b_msg[i+1])
    return sv, b_msg


# Normalize the probabilities
def normalize(probs):
    total = probs.sum()
    alpha = 1/total
    return alpha*probs


if __name__ == '__main__':
    # Dynamic/transition model
    T = np.matrix('0.7 0.3; 0.3 0.7')

    # Observation/sensor model
    O = [np.matrix('0.1 0; 0 0.8'),  # P(Umberella = False)
         np.matrix('0.9 0; 0 0.2')]  # P(Umberella = True)

    # Problems, where 1 means the umberella is shown and 0 that it is not
    prob_1 = [1, 1]
    prob_2 = [1, 1, 0, 1, 1]

    # Printing
    print("Filtering prob 1")
    print(filtering(T, O, np.matrix('0.5; 0.5'), prob_1))
    print("Filtering prob 2")
    print(filtering(T, O, np.matrix('0.5; 0.5'), prob_2))
    print("Smoothing prob 1")
    print("sv: ", smoothing(T, O, np.matrix('0.5; 0.5'), prob_1)[0])
    print("b_msg: ", smoothing(T, O, np.matrix('0.5; 0.5'), prob_1)[1])
    print("Smoothing prob 1")
    print("sv: ", smoothing(T, O, np.matrix('0.5; 0.5'), prob_2)[0])
    print("b_msg: ", smoothing(T, O, np.matrix('0.5; 0.5'), prob_2)[1])
