import sys
import os
import ast
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from functools import reduce
import trace_tools

def clearFileWithName( filename ):
    try:
        os.remove(filename)
    except OSError:
        pass    

def decipherFilename( filename ):
    with open(filename, 'r') as fileHandle:
        delimited = fileHandle.read().split("+++")
    longFileName = delimited[0]

    groundTruth = re.search('\[(.+?)\]', longFileName).group(0)
    if groundTruth is None:
        print('the following file breaks the groundTruth regex: {}'.format(longFileName))
        return

    groundBatchSize = re.search('b(.+?)_', longFileName).group(1)
    if groundBatchSize is None:
        print('the following file breaks the groundBatchSize regex: {}'.format( longFileName))
        return
    else:
        groundBatchSize = int(groundBatchSize)

    inputNumChannels = re.search('i_(.+?)_b', longFileName).group(1)
    if inputNumChannels is None:
        print('the following file breaks the inputNumChannels regex: {}'.format( longFileName))
        return
    else:
       inputNumChannels = int(inputNumChannels)

    inputImgWidth = re.search('s_(.+?)_[kt]', longFileName).group(1)
    if inputImgWidth is None:
        print('the following file breaks the inputImgWidth regex: {}'.format( longFileName))
        return
    else:
       inputImgWidth = int(inputImgWidth)

    kerasOrTorch = re.search('_(.)_i_', longFileName).group(1)
    if kerasOrTorch is None:
        print('the following file breaks the kerasOrTorch regex: {}'.format( longFileName))
        return
    elif kerasOrTorch != 't':
        print('not equipped to handle keras instead of pyTorch yet, kerasOrTorch regex: {}'.format( longFileName))
        return        
    return groundTruth, groundBatchSize, inputNumChannels, inputImgWidth

def getLayerWidthDataset( folder_name, numUniqueBatchSizes ):
    if folder_name[-1] == '/': folder_name = folder_name[:-1]
    outFile = folder_name + '/genData_areaDurationTotal.txt'

    if numUniqueBatchSizes != 0:
        clearFileWithName(outFile)
        filesToGenFrom = []
        for (dirpath, dirnames, filenames) in os.walk( folder_name ):
            filesToGenFrom.extend(filenames)
            break
        filesToGenFrom.sort()

        netCounter = 1
        sameNetDiffBatchIdx = 0
        for timing_scan_trace_file in tqdm(filesToGenFrom):
            if timing_scan_trace_file == '.DS_Store'  or timing_scan_trace_file[0:3] == 'gen': continue # ignore files we generate
            groundTruth, groundBatchSize, inputNumChannels, inputImgWidth = decipherFilename( folder_name + '/' + timing_scan_trace_file)
            layerWidthsList = ast.literal_eval(groundTruth)
            currDims = [inputImgWidth, inputImgWidth, inputNumChannels]

            convOpName = 'pt_cudnn_convolution'
            biasAddOpName = 'pt_add'
            fcOpName = 'pt_addmm'
            batchNormOpName = ['pt_batch_norm']
            activationOpNames = ['pt_leaky_relu', 'pt_relu', 'pt_sigmoid', 'pt_softmax', 'pt_tanh', 'pt_elu']
            biasAddOpNames = [ biasAddOpName ] + activationOpNames + batchNormOpName
            poolOpNames = ['pt_avg_pool2d', 'pt_max_pool2d']
            linearLayers = ['Ar', 'At', 'As', 'Al', 'Ae', 'Asm', 'b', 'add']

            layerInput = []
            layerOutput = []
            layerType = [] # distinguish layer types from others
            layerMoniker = [] # distinguish this type of layer from others of same type
            layerIndex = [] # position in net order, for combining layers across batchSizes
            layerHalved = []

            cKernel = []
            cPad = []
            cStride = []
            def ingnoreConvFeats():
                cKernel.append( -1 )
                cPad.append( -1 )
                cStride.append( -1 )
                layerHalved.append( 0 )                

            lidx = 0
            layerIdx = 0
            RsNt = -1
            layerIsFlat = False
            strPlaceHolder = 1
            maxMonikerLength = 16
            maxTypeLength = 1
            while lidx < len(layerWidthsList):
                layer = layerWidthsList[lidx]

                nodesIn = reduce(lambda x, y: x*y, currDims)
                if layer in linearLayers: # 1
                    ltype = 1
                    if layer == 'b': # 3
                        ltype = 3
                    elif layer == 'add': # 7
                        ltype = 7                       

                    layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    layerOutput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    layerType.append( int(str(strPlaceHolder) + str(ltype).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(ltype) + str( linearLayers.index(layer) ) +  str(nodesIn).zfill(4) ) )
                    lidx -= 1
                    ingnoreConvFeats()
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1

                elif layer == 'N': # 2
                    nodesOut = int( layerWidthsList[lidx + 1] )
                    layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    layerOutput.append( [nodesOut, 1, 1, nodesOut] )
                    layerType.append( int(str(strPlaceHolder) + str(2).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(2) + str(nodesIn).zfill(4) + str(nodesOut).zfill(4) ) )
                    currDims = [1, 1, nodesOut]
                    layerIsFlat = True
                    ingnoreConvFeats()
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1

                elif layer in ['L', 'R', 'BL']: # 8
                    nodesOut = int( layerWidthsList[lidx + 1] )
                    layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    layerOutput.append( [nodesOut, 1, currDims[2], nodesOut] )
                    layerType.append( int(str(strPlaceHolder) + str(8).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(8) + str(nodesIn).zfill(4) + str(nodesOut).zfill(4) ) )
                    currDims = [1, currDims[2], nodesOut]
                    ingnoreConvFeats()
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1

                elif layer in ['C2V', 'C2S', 'C2P']: # 4
                    if layerIsFlat:
                        nodesOut = 784
                        layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                        layerOutput.append( [nodesOut, 1, 1, nodesOut] )
                        layerType.append( int(str(strPlaceHolder) + str(2).zfill(maxTypeLength) ))
                        layerMoniker.append( int( str(2) + str(nodesIn).zfill(4) + str(nodesOut).zfill(4) ) )
                        currDims = [1, 1, nodesOut]
                        ingnoreConvFeats()
                        layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                        layerIdx += 1

                        layerInput.append( [nodesOut, currDims[0], currDims[1], currDims[2]] )
                        layerOutput.append( [nodesOut, currDims[0], currDims[1], currDims[2]] )
                        layerType.append( int(str(strPlaceHolder) + str(1).zfill(maxTypeLength) ) )
                        layerMoniker.append( int( str(1) + str(0) + str(nodesOut).zfill(4) ) )
                        ingnoreConvFeats()
                        layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                        layerIdx += 1                        

                        currDims = [28,28,1]
                        nodesIn = reduce(lambda x, y: x*y, currDims)
                        layerIsFlat = False

                    layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    dilation = 1
                    channel_out = int(layerWidthsList[lidx + 1])
                    kernel = int(layerWidthsList[lidx + 2])
                    stride = int(layerWidthsList[lidx + 3])
                    if layer == 'C2V':
                        pad = 0
                        layerHalved.append(0)                        
                    elif layer == 'C2S':
                        pad = int((kernel - 1)/2)
                        layerHalved.append(0)                        
                        if kernel % 2 == 0: # not sampled
                            pad = kernel - 1
                            dilation = 2 # not supported...
                            return
                    elif layer == 'C2P': # lookback to 2nd previous for resnet residual connection
                        pad = 0
                        if RsNt < 50:
                            layerInput[-1] = layerInput[-6]
                        else:
                            layerInput[-1] = layerInput[-8]
                        currDims = layerInput[-1][1:]
                        layerHalved.append(2)
                    else:
                        print('undefined pooling layer lidx {} for layer {} in {}'.format(lidx, layer, layerWidthsList))
                        sys.exit()

                    h_out = int((currDims[0] + 2 * pad - dilation * (kernel - 1) - 1)/stride + 1)
                    currDims = [h_out, h_out, channel_out]
                    nodesOut = reduce(lambda x, y: x*y, currDims)
                    layerOutput.append( [nodesOut, currDims[0], currDims[1], currDims[2]] )
                    layerType.append( int(str(strPlaceHolder) + str(4).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(kernel) + str(pad) + str(layerInput[-1][3]).zfill(4) + str(layerOutput[-1][3]).zfill(4) 
                                             + str(layerInput[-1][1]).zfill(3) + str(layerInput[-1][1]).zfill(3) ) )
                    cKernel.append( kernel )
                    cPad.append( pad )
                    cStride.append( stride )
                    lidx += 2
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1
                    if len(layerType) > 1 and str(layerType[-2])[1] == '5':
                        layerHalved[-1] = 1

                elif layer == 'P2': # 5
                    imgHalf = int(currDims[1]/2)
                    nodesOut = imgHalf * imgHalf * currDims[2]
                    layerInput.append( [nodesIn, currDims[0], currDims[1], currDims[2]] )
                    layerOutput.append( [nodesOut, imgHalf, imgHalf, currDims[2]] ) # flatten applied after pool
                    layerType.append( int(str(strPlaceHolder) + str(5).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(5) + str(nodesIn).zfill(4) + str(imgHalf).zfill(3) + str(nodesOut).zfill(4) ) )
                    ingnoreConvFeats()
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1                    
                    currDims = [imgHalf, imgHalf, currDims[2]]

                elif layer == 'PS': # 9
                    imgHalf = int(currDims[1]/2)
                    nodesOut = currDims[2]
                    layerInput.append( [nodesIn, currDims[0], currDims[1], nodesOut] )
                    layerOutput.append( [nodesOut, 1, 1, nodesOut] ) # flatten applied after pool
                    layerType.append( int(str(strPlaceHolder) + str(9).zfill(maxTypeLength) ) )
                    layerMoniker.append( int( str(5) + str(nodesIn).zfill(4) + str(1).zfill(3) + str(nodesOut).zfill(4) ) )
                    ingnoreConvFeats()
                    layerIndex.append( int( str(1) + str(layerIdx).zfill(3) + str(sameNetDiffBatchIdx).zfill(4) ) )
                    layerIdx += 1                    
                    currDims = [1, 1, nodesOut]

                elif layer == 'F': # 8
                    currDims = [1, 1, nodesIn]
                    layerIsFlat = True
                    lidx -= 1

                elif layer in ['RsN', 'RsNc']: # 6
                    RsNt = int(layerWidthsList[lidx + 1])
                    lidx += 1

                elif layer in ['VGG']: # 6
                    RsNt = int(layerWidthsList[lidx + 1])
                    # lidx += 1
                elif layer in ['Mn', 'Anet', 'Tn', 'mnC1', 'mnC2', 'mnD1', 'mnD2']: # 6
                    lidx -= 1
                else:
                    print('file {} has no identifable layer marker {}'.format(timing_scan_trace_file, layer) )
                    return
                lidx += 2
            # END WHILE

            if netCounter % numUniqueBatchSizes == 0:
                sameNetDiffBatchIdx += 1
            netCounter += 1

            numLayers = len(layerInput)
            assert( len(layerOutput) == numLayers )
            assert( len(layerType) == numLayers )
            assert( len(layerMoniker) == numLayers )
            assert( len(cKernel) == numLayers )
            assert( len(cPad) == numLayers )
            assert( len(cStride) == numLayers )
            assert( len(layerIndex) == numLayers )
            assert( len(layerHalved) == numLayers )

            layerInput = np.array( layerInput )
            layerOutput = np.array( layerOutput )
            layerType = np.array( layerType )
            layerMoniker = np.array( layerMoniker )
            cKernel = np.array( cKernel )
            cPad = np.array( cPad )
            cStride = np.array( cStride )
            layerBatchSize = np.full( layerType.shape, groundBatchSize )
            layerIndex = np.array( layerIndex )
            layerHalved = np.array( layerHalved )

            scannedData, eventTimeline, increment, shift, leftRightTimeSample = trace_tools.preprocessFile( folder_name + '/' + timing_scan_trace_file )
            increment, shift, leftBoundSample, rightBoundSample = trace_tools.segmentBounds( leftRightTimeSample[0], eventTimeline, scannedData, increment, shift )
            if leftBoundSample == -1: continue
            segIdx = 0

            avgBase = np.mean( scannedData[leftBoundSample - 20 : leftBoundSample] )
            op_time_dur_name = trace_tools.filterOpsInsideSegment( increment, shift, scannedData, eventTimeline, leftBoundSample, rightBoundSample, False )

            if len(op_time_dur_name[0]) != numLayers:
                print( len(op_time_dur_name[0]), numLayers, 'different layers from filename and op-trace')
                return
            ops_time = np.array( op_time_dur_name[0] )
            ops_dur = np.array( op_time_dur_name[1] )
            ops_name = np.array( op_time_dur_name[2] )
            ops_dur = ops_dur[ops_time.argsort()]
            ops_name = ops_name[ops_time.argsort()]
            ops_time.sort()
            ops_heights = trace_tools.sampleHeight( ops_time, ops_dur, scannedData, avgBase )

            assert( len(ops_time) == numLayers )
            assert( len(ops_dur) == numLayers )
            assert( len(ops_name) == numLayers )
            assert( len(ops_heights) == numLayers )
            dataset = np.stack((layerType, layerMoniker, cKernel, cPad, cStride, layerBatchSize, ops_time, ops_dur, layerIndex, layerHalved), axis=-1)
            dataset = np.hstack((layerInput, layerOutput, dataset, ops_heights))
            with open(outFile, "a") as f:
                np.savetxt(f, dataset)
    return

def main():
    if len(sys.argv) == 3:
        getLayerWidthDataset(sys.argv[1], int(sys.argv[2]))     
    else:
        print('insufficient arguments %d' % len(sys.argv))
        for i in range(len(sys.argv)):
            print('%d %s' % (i, sys.argv[i]))

if __name__ == '__main__':
    main()