import sys
import ast
import random
import numpy as np
import torch as pt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from functools import reduce

## off the shelf networks:
import build_net

class ptView(pt.nn.Module):
    def __init__(self, shape):
        super(ptView, self).__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class ptSelectItem(pt.nn.Module):
    def __init__(self, item_index):
        super(ptSelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, x):
        return x[self.item_index]

def configure_problem_torch( startingDims, hidden_layers  ):
    layers = []
    currDims = startingDims

    lidx = 0
    appendClassifier = True
    skipAct = False
    actVariants = ['Ar', 'At', 'As', 'Al', 'Ae', 'Asm']
    act_funcs = [ pt.nn.ReLU(), pt.nn.Tanh(), pt.nn.Sigmoid(), pt.nn.LeakyReLU(), pt.nn.ELU(), pt.nn.Softmax(dim=1) ]
    layerIsFlat = False
    while lidx < len(hidden_layers):
        layer = hidden_layers[lidx]

        if layer in ['mnC1', 'mnC2', 'mnD1', 'mnD2']:
            lidx += 1
            continue

        if layer == '_': # or layer in actVariants:
            lidx += 1
            continue
        elif layer in actVariants:
            act_fun = act_funcs[ actVariants.index( layer ) ]
            layers.append( act_fun )
            lidx += 1
            continue

        if layer == 'VGG':
            vggConfig = hidden_layers[lidx+1]
            if vggConfig == 11:
                model = build_net.VGG11()
            elif vggConfig == 13:
                model = build_net.VGG13()
            elif vggConfig == 16:
                model = build_net.VGG16()
            elif vggConfig == 19:
                model = build_net.VGG19()
            else:
                print('undefined VGG config lidx {} for layer {} in {}'.format(lidx, layer, hidden_layers))
                sys.exit()
            return model

        if layer == 'Anet':
            model = build_net.AlexNet()
            return model

        if layer == 'RsN':
            resConfig = hidden_layers[lidx + 1]
            resAct = hidden_layers[lidx + 2]
            if resConfig == 18:
                model = build_net.resnet18( resAct, pretrained=True) 
            elif resConfig == 34:
                model = build_net.resnet34( resAct, pretrained=True)
            elif resConfig == 50:
                model = build_net.resnet50( resAct, pretrained=True)
            elif resConfig == 101:
                model = build_net.resnet101( resAct, pretrained=True)
            elif resConfig == 152:
                model = build_net.resnet152( resAct, pretrained=True)
            else:
                print('undefined build_net config lidx {} for layer {} in {}'.format(lidx, layer, hidden_layers))
                sys.exit()
            return model

        if layer == 'RsNc':
            resConfig = hidden_layers[lidx + 1]
            resAct = hidden_layers[lidx + 2]            
            if resConfig == 18:
                model = build_net.cResNet18( resAct )
            elif resConfig == 34:
                model = build_net.cResNet34( resAct )
            elif resConfig == 50:
                model = build_net.cResNet50( resAct )
            elif resConfig == 101:
                model = build_net.cResNet101( resAct )
            elif resConfig == 152:
                model = build_net.cResNet152( resAct )
            else:
                print('undefined c-resnet config lidx {} for layer {} in {}'.format(lidx, layer, hidden_layers))
                sys.exit()
            return model            

        if layer[0:2] == 'C2' and layerIsFlat:
            layers.append( pt.nn.Linear( currDims[0], 784 ) )
            currDims = [784]
            layers.append( pt.nn.ReLU() )
            layers.append( ptView( (-1,1,28,28)  ) )
            currDims = [1, 28, 28]
            layerIsFlat = False

        if layer == 'b':
            if layerIsFlat:
                layers.append( pt.nn.BatchNorm1d( currDims[0] ) )
            else:
                layers.append( pt.nn.BatchNorm2d( currDims[0] ) )
            lidx += 1
            continue
        elif layer in ['R', 'L', 'BL']:

            if layer == 'R':
                layers.append( pt.nn.RNN( currDims[1], hidden_layers[lidx + 1], batch_first=True, bidirectional=False, num_layers=1 ) )
                currDims[1] = hidden_layers[lidx + 1]
            elif layer == 'L':
                layers.append( pt.nn.LSTM( currDims[1], hidden_layers[lidx + 1], batch_first=True, bidirectional=False, num_layers=1 ) )
                currDims[1] = hidden_layers[lidx + 1]
            elif layer == 'BL':
                layers.append( pt.nn.LSTM( currDims[1], hidden_layers[lidx + 1], batch_first=True, bidirectional=True, num_layers=1 ) )
                currDims[1] = 2 * hidden_layers[lidx + 1]

            layers.append( ptSelectItem(0) )
            skipAct = True
            appendClassifier = False
        elif layer == 'P2':
            if hidden_layers[lidx + 1] == 'Pm':
                layers.append( pt.nn.MaxPool2d( 2 ) )
            elif hidden_layers[lidx + 1] == 'Pa':
                layers.append( pt.nn.AvgPool2d( 2 ) )
            else:
                print('undefined pooling layer lidx {} for layer {} in {}'.format(lidx, layer, hidden_layers))
                sys.exit()
            currDims[1:] = [int(currDims[1] / 2)] * 2
            lidx += 2
            continue
        elif layer == 'F':
            layers.append( pt.nn.Flatten() )
            layerIsFlat = True
            currDims = [reduce(lambda x, y: x*y, currDims)]
            lidx += 1
            continue
        elif layer == 'C2S':
            pad = int((hidden_layers[lidx + 2] - 1)/2)
            dilation = 1
            stride = hidden_layers[lidx+3]
            if hidden_layers[lidx + 2] % 2 == 0: #even kernels
                pad = hidden_layers[lidx + 2] - 1
                dilation = 2
            layers.append( pt.nn.Conv2d( currDims[0], hidden_layers[lidx + 1], hidden_layers[lidx + 2], stride=stride, padding=pad, dilation=dilation, bias=False ) )
            currDims[0] = hidden_layers[lidx + 1]
            h_out = int((currDims[1] + 2 * pad - dilation * (hidden_layers[lidx + 2] - 1) - 1)/stride + 1)
            currDims[1:] = [h_out] * 2            
            lidx += 2            
        elif layer == 'C2V':
            pad = 0
            dilation = 1
            stride = hidden_layers[lidx+3]
            layers.append( pt.nn.Conv2d( currDims[0], hidden_layers[lidx + 1], hidden_layers[lidx + 2], stride=stride, bias=False ) )
            currDims[0] = hidden_layers[lidx + 1]
            h_out = int((currDims[1] + 2 * pad - dilation * (hidden_layers[lidx + 2] - 1) - 1)/stride + 1)
            currDims[1:] = [h_out] * 2
            lidx += 2
        elif layer == 'N':
            if not layerIsFlat:
                layers.append( pt.nn.Flatten() )
                layerIsFlat = True
                currDims = [reduce(lambda x, y: x*y, currDims)]
            layers.append( pt.nn.Linear( currDims[0], hidden_layers[lidx + 1] ) )
            currDims = [hidden_layers[lidx + 1]]
        else:
            print('undefined layer structure reached lidx {} for layer {} in {}'.format(lidx, layer, hidden_layers))
            sys.exit()
        lidx += 2

        if lidx < len(hidden_layers) and hidden_layers[lidx] in actVariants:
            act_op = hidden_layers[lidx]
            act_fun = act_funcs[ actVariants.index( act_op ) ]
            lidx += 1
        else:
            act_fun = random.choice( act_funcs )

        if not skipAct:
            layers.append( act_fun )
            skipAct = False

    model = pt.nn.Sequential(*layers)
    return model

def kerasHidden( model, hidden_layers, act_fun, num_classes, stateful=False ):
    lidx = 0
    while lidx < len(hidden_layers):
        layer = hidden_layers[lidx]
        if layer == 'I':
            print( "skipping inputLayer shape")
        elif layer == 'b':
            model.add(tf.keras.layers.BatchNormalization())
            lidx -= 1
        elif layer == 'F':
            model.add(tf.keras.layers.Flatten())
            lidx -= 1
        elif layer == 'TF':
            model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Flatten() ) )
            lidx -= 1
        elif layer == 'D':
            model.add(tf.keras.layers.Dropout( hidden_layers[lidx + 1] ))
        elif layer == 'R':
            model.add(tf.keras.layers.SimpleRNN( hidden_layers[lidx + 1] ))
        elif layer == 'L':
            model.add(tf.keras.layers.LSTM( hidden_layers[lidx + 1], activation=act_fun, stateful=stateful ))
            # model.add( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1] ))
        elif layer == 'LL':
            model.add(tf.keras.layers.LSTM( hidden_layers[lidx + 1], return_sequences=True, activation=act_fun, stateful=stateful ))
            # model.add( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1], return_sequences=True ))
        elif layer == 'BL':
            model.add( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_layers[lidx + 1], activation=act_fun, stateful=stateful ) ) )
            # model.add( tf.keras.layers.Bidirectional( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1] ) ) )
        elif layer == 'BLL':
            model.add( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_layers[lidx + 1], return_sequences=True, activation=act_fun, stateful=stateful ), merge_mode='concat' ) )
            # model.add( tf.keras.layers.Bidirectional( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1], return_sequences=True ), merge_mode='concat' ) )
        elif layer == 'BLLS':
            model.add( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_layers[lidx + 1], return_sequences=True, activation=act_fun, stateful=stateful ), merge_mode='sum' ) )
            # model.add( tf.keras.layers.Bidirectional( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1], return_sequences=True ), merge_mode='sum' ) )
        elif layer == 'BLLA':
            model.add( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_layers[lidx + 1], return_sequences=True, activation=act_fun, stateful=stateful ), merge_mode='ave' ) )
            # model.add( tf.keras.layers.Bidirectional( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1], return_sequences=True ), merge_mode='ave' ) )
        elif layer == 'BLLM':
            model.add( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( hidden_layers[lidx + 1], return_sequences=True, activation=act_fun, stateful=stateful ), merge_mode='mul' ) )
            # model.add( tf.keras.layers.Bidirectional( tf.compat.v1.keras.layers.CuDNNLSTM( hidden_layers[lidx + 1], return_sequences=True ), merge_mode='mul' ) )
        elif layer == 'C1S':
            model.add(tf.keras.layers.Conv1D(hidden_layers[lidx + 1], hidden_layers[lidx + 2], padding='same', activation=act_fun, strides=1))
            lidx += 1
        elif layer == 'TC1S':
            model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Conv1D(hidden_layers[lidx + 1], hidden_layers[lidx + 2], padding='same', activation=act_fun, strides=1) ) )
            lidx += 1            
        elif layer == 'C1V':
            model.add(tf.keras.layers.Conv1D(hidden_layers[lidx + 1], hidden_layers[lidx + 2], padding='valid', activation=act_fun, strides=1))
            lidx += 1            
        elif layer == 'C2V':
            model.add(tf.keras.layers.Conv2D(hidden_layers[lidx + 1], hidden_layers[lidx + 2], padding='valid', activation=act_fun, strides=1))
            lidx += 1
        elif layer == 'C2S':
            model.add(tf.keras.layers.Conv2D(hidden_layers[lidx + 1], hidden_layers[lidx + 2], padding='same', activation=act_fun, strides=1))
            lidx += 1            
        elif layer == 'RsN1':
            model.add( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation=act_fun) )
        elif layer == 'RsN2':
            model.add( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation=act_fun) )
        elif layer == 'P1':
            if hidden_layers[lidx + 1] == 'Pm':
                model.add( tf.keras.layers.MaxPooling1D() )
            if hidden_layers[lidx + 1] == 'TPm':
                model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.MaxPooling1D() ) )
            elif hidden_layers[lidx + 1] == 'Pgm':
                model.add( tf.keras.layers.GlobalMaxPool1D() )
            elif hidden_layers[lidx + 1] == 'Pa':
                model.add( tf.keras.layers.AveragePooling1D() )
            elif hidden_layers[lidx + 1] == 'Pga':
                model.add( tf.keras.layers.GlobalAveragePooling1D() )
        elif layer == 'P2':
            if hidden_layers[lidx + 1] == 'Pm':
                model.add( tf.keras.layers.MaxPooling2D() )
            elif hidden_layers[lidx + 1] == 'Pgm':
                model.add( tf.keras.layers.GlobalMaxPool2D() )
            elif hidden_layers[lidx + 1] == 'Pa':
                model.add( tf.keras.layers.AveragePooling2D() )
            elif hidden_layers[lidx + 1] == 'Pga':
                model.add( tf.keras.layers.GlobalAveragePooling2D() )
        elif layer == 'RV':
            model.add( tf.keras.layers.RepeatVector(hidden_layers[lidx + 1]) )
        elif layer == 'N':
            model.add( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation=act_fun) )
        elif layer == 'NL':
            model.add( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation=None) )            
        elif layer == 'NS':
            if hidden_layers[lidx + 1] == -1:
                model.add( tf.keras.layers.Dense(num_classes, activation='softmax') )
            else:
                model.add( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation='softmax') )
        elif layer == 'TNS':
            if hidden_layers[lidx + 1] == -1:            
                model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(num_classes, activation='softmax') ) )
            else:
                model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation='softmax') ) )
        elif layer == 'TNSG':
            if hidden_layers[lidx + 1] == -1:            
                model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(num_classes, activation='sigmoid') ) )
            else:
                model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation='sigmoid') ) )                
        elif layer == 'TN':
            model.add( tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(hidden_layers[lidx + 1], activation=act_fun) ) )
        else:
            print('undefined layer structure reached {} {} {}'.format(hidden_layers, lidx, layer))
            sys.exit()
        lidx += 2
    return model
