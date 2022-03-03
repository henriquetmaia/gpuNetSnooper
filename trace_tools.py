import sys
import os
import re
import json
import numpy as np
from scipy.signal import find_peaks
from tensorflow.python.client import timeline
from tqdm import tqdm
import tensorflow as tf

def sampleHeight( ops_time, ops_dur, scannedData, avgBase ):
    width = 1
    numSamples = 3
    opHeights = np.zeros( ( len(ops_time), numSamples ) )

    for opIdx in np.arange(len(ops_time)):
        start = int( ops_time[opIdx] )
        end = int( ops_dur[opIdx] + start )

        samplePoints = np.linspace( start, end, numSamples+2 )
        for sampleIdx, samplePoint in enumerate( samplePoints ):

            if sampleIdx == 0 or sampleIdx > numSamples:
                continue
            sampleCenter = int(samplePoint)
            opHeights[opIdx, sampleIdx - 1] = np.mean( scannedData[sampleCenter - width : sampleCenter + width + 1] ) - avgBase

    return opHeights

def parseScanFile( timing_scan_trace_file ):
    with open(timing_scan_trace_file, 'r') as fileHandle:
        delimited = fileHandle.read().split("+++")

    if len(delimited) != 4:
        print('ERROR splitting {} into 4 portions'.format(timing_scan_trace_file))
        quit()

    fullArchName = delimited[0]
    scanTimingString = delimited[1]
    scannedDataString = delimited[2]
    traceJSONString = delimited[3]

    # raw voltage values for numSamples; [0,5.0] Volts:
    scannedData = np.fromstring(scannedDataString, sep=" ")

    # timing of select samples, rough indication of timing every 512:
    scanTiming = np.reshape(np.fromstring(scanTimingString, sep=" "),(-1,2)) 
    # switching to microsecond representation to match chrome-trace:
    scanTiming[:,0] = scanTiming[:,0] * 1000000 

    eventTimeline = []
    chr_trace = json.loads(traceJSONString)
    for event in chr_trace['traceEvents']:
        if 'dur' in event:
            eventTimeline.append(event)
    return scannedData, scanTiming, eventTimeline

def estimateAlignmentFromScanTiming( scannedData, scanTiming, eventTimeline ):
    timeOfFistSpike = np.inf
    durationOfFirstSpike = np.inf
    timeOfSecondSpike = np.inf
    durationOfSecondSpike = np.inf
    for event in eventTimeline:
        if event['name'] == 'normal_':
            if event['ts'] < timeOfFistSpike:
                timeOfFistSpike = event['ts']
                durationOfFirstSpike = event['dur']            
                timeOfFistSpike = timeOfFistSpike + durationOfFirstSpike/2
        if event['name'] == 'cauchy_':
            if event['ts'] < timeOfSecondSpike and event['ts'] > timeOfFistSpike:
                timeOfSecondSpike = event['ts']
                durationOfSecondSpike = event['dur']
                timeOfSecondSpike = timeOfSecondSpike + durationOfSecondSpike/2

        if event['name'] != 'unknown':
            opName = event['name'] 
        else:
            opName = (event['args'])['name'] + '_unknown'
        event['name'] = 'pt_' + opName

    if timeOfFistSpike == np.inf or durationOfFirstSpike == np.inf:
        print('ERROR finding time of first spike {} {}'.format(timeOfFistSpike, durationOfFirstSpike))
        quit()
    if timeOfSecondSpike == np.inf or durationOfSecondSpike == np.inf:
        print('ERROR finding time of second spike {} {}'.format(timeOfSecondSpike, durationOfSecondSpike))
        quit()

    #index of fist relevant spike in sensor data
    scanSpikes = find_peaks(scannedData, prominence=(0.2))[0]
    sampleOfFirstSpike = scanSpikes[0]

    sampleOfSecondSpike = scanSpikes[-1]
    increment = (timeOfSecondSpike - timeOfFistSpike) / (sampleOfSecondSpike - sampleOfFirstSpike)
    shiftRight = (timeOfSecondSpike / increment) - sampleOfSecondSpike
    shiftLeft = (timeOfFistSpike / increment) - sampleOfFirstSpike
    shift = (shiftLeft + shiftRight) / 2.0
    return increment, shift

def findBoundEvents( eventTimeline, increment, shift, sampleLimit ):
    bound_time_dur_estimate_name = [] # np.zeros((numEvents,3))
    rightIndices = []
    index = 0

    for event in eventTimeline:
        if event['name'] == 'pt_normal_':
            bound_time_dur_estimate_name.append( [event['ts'], event['dur'], (event['ts'] / increment) - shift, event['name'] ] )
            index += 1
        elif event['name'] == 'pt_cauchy_':
            bound_time_dur_estimate_name.append( [event['ts'], event['dur'], (event['ts'] / increment) - shift, event['name'] ] )
            rightIndices.append(index)
            index += 1

    return bound_time_dur_estimate_name, rightIndices

def findSegmentBounds( bound_time_dur_estimate_name, rightIndices, scannedData ):
    windowHalf = 40
    lookRight = 30
    avgLookLeft = 50
    threshold = 0.3
    boundIdx = 0
    leftRightTimeSample = np.zeros( (len(rightIndices), 4) )

    firstOnly = np.inf
    for rightBound in rightIndices:

        rightTime = bound_time_dur_estimate_name[rightBound][0]
        rightEstimate = int( bound_time_dur_estimate_name[rightBound][2] )
        nearestTime = np.NINF
        nearestOp = -1
        for op in np.arange( len(bound_time_dur_estimate_name) ):
            if bound_time_dur_estimate_name[op][3] == 'pt_normal_':
                leftTime = bound_time_dur_estimate_name[op][0]
                if leftTime < rightTime and leftTime > nearestTime:
                    nearestTime = leftTime
                    nearestOp = op

        leftTime = nearestTime
        leftEstimate = int( bound_time_dur_estimate_name[nearestOp][2] )
        leftRightTimeSample[boundIdx] = [leftTime, rightTime, leftEstimate, rightEstimate]

        boundIdx += 1
    leftRightTimeSample = leftRightTimeSample[:boundIdx]
    return leftRightTimeSample 

def preprocessFile( filepath ):
    filename = os.path.basename(filepath)    
    scannedData, scanTiming, eventTimeline = parseScanFile( filepath )
    increment, shift = estimateAlignmentFromScanTiming( scannedData, scanTiming, eventTimeline )
    bound_time_dur_estimate_name, rightIndices = findBoundEvents( eventTimeline, increment, shift, len(scannedData) )
    leftRightTimeSample = findSegmentBounds( bound_time_dur_estimate_name, rightIndices, scannedData )
    return scannedData, eventTimeline, increment, shift, leftRightTimeSample

def findTightBounds( eventTimeline, leftTime, rightTime ):
    rightBound = np.NINF
    leftBound = np.NINF

    ptLeftTightBoundOptions = ['pt_addmm', 'pt_cudnn_convolution', 'pt__cudnn_rnn']
    ptRightTightBoundOptions = ['pt__softmax', 'pt_softmax', 'pt_reshape', 'pt_addmm', 'pt_matmul']

    for event in eventTimeline:
        if event['ts'] >= leftTime and event['ts'] <= rightTime: # within segment bounds
            if event['name'] != 'unknown':
                opName = event['name']
            else:
                opName = (event['args'])['name'] + '_unknown'

            opTime = event['ts']
            opDur = event['dur']
            if opName in ptRightTightBoundOptions:
                if opTime + opDur < rightTime and opTime + opDur > rightBound:
                    rightBound = opTime + event['dur'] + 185 # microseconds bump

            if opName in ptLeftTightBoundOptions:
                if leftBound == np.NINF or (opTime < rightTime and opTime < leftBound):
                    leftBound = opTime - 100
                    if opName == 'pt__cudnn_rnn':
                        leftBound = opTime - 40
    return leftBound, rightBound

def computeAdjustedSample( x, scannedData ):
    segStart = int( round( x ) )
    lookLeft = 1
    lookRight = 1
    x = segStart - lookLeft + np.argmin( scannedData[segStart - lookLeft : segStart + lookRight + 1]  )
    return x

def segmentBounds( segLeftRightTimeSample, eventTimeline, scannedData, increment=None, shift=None ):
    leftTime = segLeftRightTimeSample[0]
    rightTime = segLeftRightTimeSample[1]
    leftTimeBound, rightTimeBound = findTightBounds( eventTimeline, leftTime, rightTime )
    if leftTimeBound == np.NINF or rightTimeBound == np.NINF:
        return -1, -1,

    leftBoundSample = int( round ( (leftTimeBound / increment) - shift ) )
    rightBoundSample = int( round ( (rightTimeBound / increment) - shift ) )

    leftBoundSample = computeAdjustedSample(leftBoundSample, scannedData)
    rightBoundSample = computeAdjustedSample(rightBoundSample, scannedData)
    return increment, shift, leftBoundSample, rightBoundSample

def computeAdjustedStartLength( x, x_length, scannedData ):
    segStart = int( round( x ) )
    segEnd = int( round( x + x_length ) )
    lookLeft = 1
    lookRight = 1
    x = segStart - lookLeft + np.argmin( scannedData[segStart - lookLeft : segStart + lookRight + 1]  )
    x_length = segEnd - lookLeft + np.argmin( scannedData[segEnd - lookLeft : segEnd + lookRight + 1]) - x

    if x_length == 0:
        x_length = 1
        x -= 1
    return x, x_length

def filterOpsInsideSegment( increment, shift, scannedData, eventTimeline, leftBoundSample, rightBoundSample, debugPrint=False, singleOpOnly=[] ):
    
    # the following trace operations are either repeats of other always-present and coincident trace ops, 
    # or nullops that do not involve a network architecture element of interest
    ptIgnoreOpts = ['pt_t', 'pt__softmax', 'pt_reshape', 'pt_view', 'pt_contiguous', 'pt_set_', 'pt_clone',
                    'pt_flatten', 'pt__unsafe_view', 'pt_empty', 'pt_set', 'pt_cudnn_is_acceptable', 'pt_rnn_tanh', 'pt_lstm',
                    'pt_adaptive_avg_pool2d', 'pt_thnn_conv_depthwise2d_forward', 'pt__convolution', 'pt_convolution',
                    'pt_dropout', 'pt_conv2d', 'pt_mm']

    filterSmallEvents = 0 
    seg_time_dur_name = [ [],[],[] ]
    for event in eventTimeline:
        eSample = (event['ts'] / increment) - shift
        if eSample >= leftBoundSample and eSample <= rightBoundSample: # within segment bounds roughly
            eSample, eDur = computeAdjustedStartLength( eSample, event['dur'] / increment, scannedData)
            if eSample >= leftBoundSample and eSample <= rightBoundSample: # within segment bounds exactly
                opName = event['name'] 
                if opName[-1] == '_': # cleanup copies of ops
                    opName = opName[:-1]
                if len(singleOpOnly) and opName not in singleOpOnly:
                    continue          

                if 'batch_norm' in opName and opName != 'pt_batch_norm':
                    continue # copy of 'pt_batch_norm'
                if 'pt_max_pool2d' in opName and opName != 'pt_max_pool2d':
                    continue # theres a 'pt_max_pool2d_with_indices' which collides perfectly
                if opName in ptIgnoreOpts:
                    continue
                if eDur == 0:
                    continue # these ops are too quick to be detected by the sensor

                seg_time_dur_name[0].append( eSample )
                seg_time_dur_name[1].append( eDur )
                seg_time_dur_name[2].append( opName )
    return seg_time_dur_name
