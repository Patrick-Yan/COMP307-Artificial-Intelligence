import numpy as np
import matplotlib.pyplot as plt

# load the data into a array
Input = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1],
                  [1, 0, 0],
                  [0, 1, 1],
                  [0, 0, 0]])
# correct answer
expectedOutput = np.array([0, 1, 1, 0, 0, 1, 1, 0])
# initialize the weight and range is (-1,1)
W = (np.random.random(3) - 0.5) * 2
# W = np.array([0,0,0])
# learning rate
lr = 0.01
# initialize the output
Output = 0
# initialize the epochs
epochs = 0
# count the number of loops
i = 0
# count how many errors show
error = 0
# list for drawing image
List = []
# backup for accuracy
m = 0

backup = []

for steps in range(200 * 8):

    Output = np.dot(Input[i], W)
    #
    # if Output > 0:
    #     Output = 1
    # else:
    #     Output = 0

    # W = W + lr * ((expectedOutput[i] - Output) * Input[i])

    if Output < expectedOutput[i]:
        for d in range(3):
            if Input[i][d] == 1:
                W[d] = W[d] + lr * ((expectedOutput[i] - Output) * Input[i][d])
            else:
                W[d] = W[d] - lr * ((expectedOutput[i] - Output) * Input[i][d])
    if Output > expectedOutput[i]:
        for g in range(3):

            if Input[i][g] == 1:
                W[g] = W[g] - lr * ((Output - expectedOutput[i]) * Input[i][g])
            else:
                W[g] = W[g] + lr * ((Output - expectedOutput[i]) * Input[i][g])

    i = i + 1
    if i == 8:
        i = 0
        epochs = epochs + 1

        print(epochs)

        Output = np.dot(Input, W)
        for z in range(8):
            if Output[z] <= 0:
                Output[z] = 0
            else:
                Output[z] = 1

        j = 0
        for k in range(8):
            if Output[k] == expectedOutput[k]:
                j = j + 1
                m = j / 8
        List.append(m)
        print(m)
        print(W)
        print("------------------")

plt.plot(List)
plt.show()
