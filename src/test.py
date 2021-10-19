import numpy as np

test = np.array([[1, 2, 3], [4, 5, 6]])
print(test.shape)
for el in test:
    print(el)
test = test.reshape(-1)
for el in test:
    print(el)