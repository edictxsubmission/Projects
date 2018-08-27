import struct

import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as file:
        zero, data_type, dims = struct.unpack('>HBB', file.read(4))
        shape = tuple(struct.unpack('>I', file.read(4))[0] for d in range(dims))
        return np.fromstring(file.read(), dtype=np.uint8).reshape(shape)

def main(test_count, train_count):
    train_data = read_idx("train-images.idx3-ubyte")
    train_labels = read_idx("train-labels.idx1-ubyte")
    test_data = read_idx("t10k-images.idx3-ubyte")
    test_labels = read_idx("t10k-labels.idx1-ubyte")
    
    trainVectors = []
    trainLabels = []
    testVectors = []
    testLabels = []
    
    for i in range(0, 20000):
        newTrainVec = [1]
        for x in range(0, 28):
            for y in range(0, 28):
                newTrainVec.append(train_data[i][x][y])
        trainVectors.append(newTrainVec)
        trainLabels.append(train_labels[i])
        
    for i in range(8000, 10000):
        newTestVec = [1]
        for x in range(0, 28):
            for y in range(0, 28):
                testVectors.append(test_data[i][x][y])
        testVectors.append(newTestVec)
        testLabels.append(test_labels[i])
        
    return len(trainVectors[0])