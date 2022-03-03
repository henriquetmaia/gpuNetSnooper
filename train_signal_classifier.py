import os
import re
import sys
import time
import math
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # cleanup tf verbosity

from tqdm import tqdm
import tensorflow as tf
import torch
import compose_net

BATCH_SIZE = 128
EPOCHS = 15 
PERCENT_OF_DATASET = 100

N_FEATURES = 1
NORMALIZE_INPUTS = True
SHUFFLE_DATASET = True
REMOVE_REP_DATA = True

FEATURE_ID = 1

WINDOW_LENGTHS = [128]
WINDOW_STRIDE = 1
TARGET_INDICES = [-1] # -1 is all

ACTIVATIONS = [
    'tanh',
    # 'elu',
    # 'relu',
    # 'sigmoid',
    # 'leaky_relu'
]

# Classifiers to try:
LAYER_DEFS_KERAS = [
    ['BLL', 128, 'D', 0.2, 'BLL', 128, 'TNS', -1 ],
]

def checkValid( dataSet ):
    if (np.isinf(dataSet)).any() or (np.isnan(dataSet)).any():
        print( 'inf or nan found in dataSet, exiting')
        sys.exit()
    return

def normalizeDsetToFile(trainInput, normalizedPrefix):
    outFile = 'genTrainNormalizeMeanStd_' + normalizedPrefix + '.txt'
    with open(outFile, "w") as f:
        np.savetxt(f, trainInput.mean(0))
        f.write('+++\n')
        np.savetxt(f, trainInput.std(0))
    return

def normalizeDsetFromFile( trainInput, testInput, normalizedPrefix ):
    outFile = 'genTrainNormalizeMeanStd_' + normalizedPrefix + '.txt'
    with open(outFile, 'r') as fileHandle:
        delimited = fileHandle.read().split("+++") # last delimited should be avoided, since empty
        trainInputMean = np.reshape(np.fromstring(delimited[0], sep=" "),(-1,1)) 
        trainInputStd = np.reshape(np.fromstring(delimited[1], sep=" "),(-1,1)) 

    testInput = (testInput - trainInputMean) / trainInputStd
    trainInput = (trainInput - trainInputMean) / trainInputStd
    return trainInput, testInput   

def preprocSplitDatasetToParamWindow( trainFile, wL, targetIndex, fileTypeEnum, normalizedPrefix ):
    windowLength = abs(wL)

    if targetIndex == -1:
        indexOfTarget = 'ALL'
    else:
        indexOfTarget = targetIndex

    if fileTypeEnum == 1: # train
        tmpFile = "genTrainDataSetFromSeriesSplit_" + str(windowLength) + "_" + str(WINDOW_STRIDE) + "_" + str(FEATURE_ID) + "_" + str(indexOfTarget) + ".bin"
    elif fileTypeEnum == 0: # test
        tmpFile = "genTestDataSetFromSeriesSplit_" + str(windowLength) + "_" + str(WINDOW_STRIDE) + "_" + str(FEATURE_ID) + "_" + str(indexOfTarget) + ".bin"
    elif fileTypeEnum == 2: # both
        tmpFile = "genBothDataSetFromSeriesSplit_" + str(windowLength) + "_" + str(WINDOW_STRIDE) + "_" + str(FEATURE_ID) + "_" + str(indexOfTarget) + ".bin"

    if not os.path.isfile(tmpFile):
        #create it
        print('{} does not exist, creating it'.format(tmpFile))
        numSequenceSamples = 0
        numSequenceSamplesSkipped = 0
        with open(tmpFile, "wb") as f: 
            with open(trainFile, 'r') as fileHandle:
                next(fileHandle)
                delimited = fileHandle.read().split("+++") # last delimited should be avoided, since empty
                for inferenceID in tqdm(np.arange(len(delimited) - 1)):
                    scan_class_seg = np.reshape(np.fromstring(delimited[inferenceID], sep=" "),(-1,3)) 

                    for startIdx in np.arange(0, scan_class_seg.shape[0]- windowLength, WINDOW_STRIDE):
                        window = scan_class_seg[startIdx:startIdx+windowLength,0]
                        if targetIndex == -1:
                            target = scan_class_seg[startIdx:startIdx+windowLength, FEATURE_ID]
                            if REMOVE_REP_DATA and fileTypeEnum == 1: # cull only training data
                                if len(np.unique(target)) < 2 and np.random.randint(0, 2) > 0: # all same and 1/2 chance
                                    numSequenceSamplesSkipped += 1
                                    continue
                        else:
                            target = scan_class_seg[startIdx + indexOfTarget, FEATURE_ID]
                        sequenceSample = np.concatenate( (window, target), axis=None )
                        sequenceSample.tofile(f)
                        numSequenceSamples += 1
        print('numSequenceSamples {}, num skipped {}'.format(numSequenceSamples, numSequenceSamplesSkipped))
    else:
        print('{} already exists! Using it'.format(tmpFile))

    with open(tmpFile, 'rb') as f:
        dataset = np.fromfile(f)
        checkValid(dataset)

        if targetIndex == -1:
            dataset = dataset.reshape(-1, 2*windowLength)
        else:
            dataset = dataset.reshape(-1, windowLength + 1)

        if fileTypeEnum == 1:
            if NORMALIZE_INPUTS:
                print('NORMALIZE_INPUTS is ON, saving to file')
                normalizeDsetToFile(dataset[:, :windowLength], normalizedPrefix)
            else:
                print('NORMALIZE_INPUTS is OFF, not saving to file')

        endOfDataset = int(dataset.shape[0] * PERCENT_OF_DATASET/100.0)    
        if SHUFFLE_DATASET:
            print('SHUFFLE_DATASET is ON')
            np.random.shuffle(dataset)
        else:
            print('SHUFFLE_DATASET is OFF')

        inputData = dataset[:endOfDataset, :windowLength]
        targetData = dataset[:endOfDataset, windowLength:]

        print('num unique classes starting with: {} as: {}'.format( len(np.unique(targetData)), np.unique(targetData) ))

        if fileTypeEnum == 2:
            testInputData = dataset[endOfDataset:, :windowLength]
            testTargetData = dataset[endOfDataset:, windowLength:]

        inputData = inputData.reshape(inputData.shape[0],inputData.shape[1], N_FEATURES)
        if fileTypeEnum == 2:
            testInputData = testInputData.reshape(testInputData.shape[0], testInputData.shape[1], N_FEATURES)

        if targetIndex == -1: # if predicting multiple timesteps per sample, target must be 3D
            targetData = targetData.reshape(targetData.shape[0], targetData.shape[1], N_FEATURES)
            if fileTypeEnum == 2:
                testTargetData = testTargetData.reshape(testTargetData.shape[0], testTargetData.shape[1], N_FEATURES)
        else:
            targetData = targetData.reshape(targetData.shape[0], N_FEATURES)
            if fileTypeEnum == 2:
                testTargetData = testTargetData.reshape(testTargetData.shape[0], N_FEATURES)

        print('inputData shaped: {}, target shaped: {}, unique classes: {}|{}'.format(
            inputData.shape, targetData.shape, len(np.unique(targetData)), np.unique(targetData) ))
        assert(  inputData.shape[0] == targetData.shape[0] )
        if fileTypeEnum == 2:
            print('testInputData shaped: {}'.format(testInputData.shape))
            print('testTargetData shaped: {}'.format(testTargetData.shape))
            print('num unique classes: {}'.format( len(np.unique(testTargetData)) ))
            assert( testInputData.shape[0] == testTargetData.shape[0] )
            return inputData, targetData, testInputData, testTargetData
        else:
            return inputData, targetData

def run_timeSeries( BATCH_SIZE, EPOCHS, runPrefix, hidden_layers, act_fun, x_train, y_train, x_test, y_test, windowLength, targetIndex, normalizedPrefix ):

    checkpoint_dir = "training_ckpts"
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if NORMALIZE_INPUTS:
        print('NORMALIZE_INPUTS is ON')
        x_train, x_test = normalizeDsetFromFile(x_train, x_test, normalizedPrefix)
    else:
        print('NORMALIZE_INPUTS is OFF')

    n_classes = len(np.unique(y_train))

    print('windowLength ', windowLength )
    model = tf.keras.Sequential()
    if windowLength < 0: # and targetIndex < 0:
        input_shape = (None, N_FEATURES)
    else:
        input_shape = (windowLength, N_FEATURES)
    model.add( tf.keras.layers.InputLayer(input_shape))

    if FEATURE_ID == 1:
        model = compose_net.kerasHidden(model, hidden_layers, act_fun, n_classes)
    elif FEATURE_ID == 2:
        model = compose_net.kerasHidden(model, hidden_layers, act_fun, 1)

    # if softmax, from_logits=False, if no softmax but linear, from_logits=True
    if FEATURE_ID == 1:
        print('FEATURE_ID 1 model compile')
        model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['acc'])
        print('FEATURE_ID 1 model compiled')

    elif FEATURE_ID ==2:
        print('BinaryCrossentropy')
        model.compile(
                    # optimizer='adam',
                    optimizer='rmsprop',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=['acc'])
    print('model summary:\n',model.summary())

    # monitor_term = 'val_loss'
    monitor_term = 'val_acc'

    # Create a callback that saves the model's weights
    checkpoint_path = checkpoint_dir + "/" + runPrefix +".hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1, monitor=monitor_term)

    # train
    historyOp = model.fit( x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cp_callback], shuffle=SHUFFLE_DATASET,
                                validation_data=(x_test,y_test), class_weight=None, validation_freq=1, verbose=1)

    # save model
    history = historyOp.history
    lowestValLoss = np.amin(history[monitor_term])
    model = tf.keras.models.load_model(checkpoint_path)
    save_path = 'saved_{:.5f}_'.format( lowestValLoss )  + str(windowLength) + "_" + str(WINDOW_STRIDE) + '_' + str(BATCH_SIZE) \
                            + '_' + str(hidden_layers).replace(" ", "") + '.hdf5'
    model.save( save_path )
    print("Model saved in path: %s" % save_path)
    return

def trainClassifier():
    origStart = time.time()
    indexRun = 0
    timestring = time.strftime("%Y%m%d-%H%M%S") + '_'

    trainFile = 'genData_t00_FULL_train_timeseries_+++.txt'
    testFile = 'genData_t01_timeseries_+++.txt'
    numRuns = len(LAYER_DEFS_KERAS) * len(ACTIVATIONS) * len(TARGET_INDICES) * len(WINDOW_LENGTHS)
    for windowLength in WINDOW_LENGTHS:
        if windowLength < 0:
            wID = 'N' + str(-windowLength)
        else:
            wID = windowLength

        for tIdx in TARGET_INDICES:
            if tIdx == -1:
                tID = 'ALL'
            else:
                tID = tIdx

            normalizedPrefix = timestring + str(wID) + "_" + str(tID) \
                                            + "_" + str(PERCENT_OF_DATASET) + "_b_" + str(BATCH_SIZE) \
                                            + "_e_" + str(EPOCHS)
            testInput, testTarget = preprocSplitDatasetToParamWindow(testFile, windowLength, tIdx, 0, normalizedPrefix )
            trainInput, trainTarget = preprocSplitDatasetToParamWindow(trainFile, windowLength, tIdx, 1, normalizedPrefix )
            # trainInput, trainTarget, testInput, testTarget = preprocSplitDatasetToParamWindow(bothFile, windowLength, tIdx, 2, normalizedPrefix )
            assert( len(np.unique(trainTarget)) == len(np.unique(testTarget)) )

            for layer in range(len(LAYER_DEFS_KERAS)):
                specs = LAYER_DEFS_KERAS[layer]
                for activation in ACTIVATIONS:
                    tf.compat.v1.reset_default_graph() # clear graphs
                    torch.cuda.empty_cache() # clear GPU

                    start = time.time()
                    if activation == 'relu':
                        act = tf.nn.relu
                    if activation == 'elu':
                        act = tf.nn.elu
                    elif activation == 'tanh':
                        act = tf.math.tanh
                    elif activation == 'sigmoid':
                        act = tf.math.sigmoid
                    else:
                        act = tf.nn.leaky_relu

                    runPrefix = timestring + str(indexRun).zfill(3) + "_" + str(wID) + "_" + str(tID) \
                                            + "_" + str(PERCENT_OF_DATASET) + "_" + activation + "_b_" + str(BATCH_SIZE) \
                                            + "_e_" + str(EPOCHS) + "_" + str(specs).replace(" ", "") 
                    indexRun += 1
                    print('attempting {}/{} aka: {}'.format(indexRun, numRuns, runPrefix))
                    run_timeSeries( BATCH_SIZE, EPOCHS, runPrefix, specs, act, trainInput, trainTarget, 
                                                testInput, testTarget, windowLength, tIdx, normalizedPrefix)
                    print("Finished training {} in {:.2f}s\n".format(runPrefix, time.time() - start))

    print("Finished Comparing Networks in {:.2f}s".format(time.time() - origStart))
    return

def main():
    if len(sys.argv) == 1:
        trainClassifier()
    else:
        print('missing windowLength argument, or too many arguments')
        print('insufficient arguments %d' % len(sys.argv))
        for i in range(len(sys.argv)):
            print('%d %s' % (i, sys.argv[i]))

if __name__ == '__main__':
    main()
