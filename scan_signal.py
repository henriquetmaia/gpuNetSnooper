import os
import sys
import time
import tempfile
import numpy as np
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # helps find GPU 

import torch
from DAQSession import DAQSession
import compose_net

REPEAT_INF = 1 # how many times to repeat network in one scan session
# INF_CENT = range(90,151,30) # batch-sizes to sample at
INF_CENT = [110]

# Here we can determine which network types we would like represented in the dataset
# missing layer definitions, e.g. activations, are randomly assigned where appropriate
LAYERS_DEFS_TORCH = [

        ### Randomly generated networks
        # ['N',1024,'N',1024,'N',1024], # simple Fully-connected net, with sizes prescribed, activations to be randomly chosen
        # ['C2S', 64, 3, 1, 'At', 'C2S', 64, 3, 1, 'Ar', 'N', 256, 'Ar' ], # CNN with filters, kernel, stride, and activations prescribed
        # ['C2', 'C2', 'C2', 'C2', 'P2' ], # 4 Convolutions followed by a pooling layer
        # ['CB', 'CB', 'CB', 'CB', 'NB' ], # block-structured Convolutions and Fully-Connected sequences
        # ['L','N'], # LSTM followed by Fully-connected
        # ['R','N'], # simple RNN network

        ### Popular network architectures
        ['VGG', 16], # CIFAR-10 VGG
        ['RsNc', 18], # CIFAR-10 ResNet
        # ["Anet"], # CIFAR-10 AlexNet
        # ['RsN', 50], # ImageNet ResNet

        ### hand crafted networks for testing and debugging
        # ['mnC1'],
        # ['mnC2'],
        # ['mnD1'],
        # ['mnD2'],

        ['VGG', 13], # CIFAR-10 VGG
        ['VGG', 19], # CIFAR-10 VGG
        ['RsN', 18],
        # ['RsN', 50],

    ]

class TimeLiner:
    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, stepStats):
        fetched_timeline = timeline.Timeline(stepStats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        if self._timeline_dict is None: # for first run store full trace
            self._timeline_dict = chrome_trace_dict
        else: # for other - update only time consumption, not definitions
            for event in chrome_trace_dict['traceEvents']:
                if 'ts' in event and 'dur' in event: # events time consumption started with 'ts' prefix
                    self._timeline_dict['traceEvents'].append(event)

    def update_timeline_with_json(self, chrome_trace_json_file):
        # convert crome trace to python dict
        chrome_trace_dict = json.load(chrome_trace_json_file)

        if self._timeline_dict is None: # for first run store full trace
            self._timeline_dict = dict.fromkeys(['traceEvents'], [])
        for event in chrome_trace_dict:
            if 'ts' in event and 'dur' in event: 
                if 'CUDA' in event['pid']: # has both CPU and CUDA calls, just care about CUDA/GPU
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'a') as f:
            json.dump(self._timeline_dict, f, indent=4)

def enableDAQEvents( ds, outFile ):
    user_data = { 
        "status" : {'complete': False, 'error': False},
        "f_name" : outFile,
        "curr" : 0,
        "timeIdx" : np.zeros((300,2)) }
    ds.enableEvent(user_data)
    return

def genNetSpecs( specBluePrint ):
    # Make layer definitions explicit to create high-level labeled dataset

    # Relu, Tanh, Sigmoid, LeakyRelu, Elu for now, other can be added
    actVariants = ['Ar', 'At', 'As', 'Al', 'Ae']
    specs = []
    if specBluePrint[0] in ['RsN', 'RsNc']:

        specs += specBluePrint
        # specs.append( np.random.choice(actVariants) )
        specs.append( 'Ar' )

        if specBluePrint[0] == 'RsN':
            specs += [ 'C2S', 64, 7, 2, 'b', 'Ar', 'P2', 'Pm' ]
        elif specBluePrint[0] == 'RsNc':
            specs += [ 'C2S', 64, 3, 1, 'b', 'Ar' ]

        if specBluePrint[1] == 18:
            specs += [ 'C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'add', 'Ar' ] * 2
            specs += [ 'C2S', 128, 3, 2, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'C2P', 128, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 256, 3, 2, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'C2P', 256, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 512, 3, 2, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'C2P', 512, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'add', 'Ar' ] 
        elif specBluePrint[1] == 34:
            specs += [ 'C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'add', 'Ar' ] * 3
            specs += [ 'C2S', 128, 3, 2, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'C2P', 128, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'add', 'Ar' ] * 3
            specs += [ 'C2S', 256, 3, 2, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'C2P', 256, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'add', 'Ar' ] * 5
            specs += [ 'C2S', 512, 3, 2, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'C2P', 512, 1, 2, 'b', 'add', 'Ar' ] 
            specs += [ 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'add', 'Ar' ] * 2
        else:
            block_1 = [ 'C2S', 64, 1, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 256, 1, 1, 'b', 'C2S', 256, 1, 1, 'b', 'add', 'Ar' ] 
            block_1 += [ 'C2S', 64, 1, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 256, 1, 1, 'b', 'add', 'Ar' ] * 2

            block_2A = [ 'C2S', 128, 1, 1, 'b', 'Ar', 'C2S', 128, 3, 2, 'b', 'Ar', 'C2S', 512, 1, 1, 'b', 'C2P', 512, 1, 2, 'b', 'add', 'Ar' ] 
            block_2B = [ 'C2S', 128, 1, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 512, 1, 1, 'b', 'add', 'Ar' ]

            block_3A = [ 'C2S', 256, 1, 1, 'b', 'Ar', 'C2S', 256, 3, 2, 'b', 'Ar', 'C2S', 1024, 1, 1, 'b', 'C2P', 1024, 1, 2, 'b', 'add', 'Ar' ] 
            block_3B = [ 'C2S', 256, 1, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 1024, 1, 1, 'b', 'add', 'Ar' ]

            block_4 = [ 'C2S', 512, 1, 1, 'b', 'Ar', 'C2S', 512, 3, 2, 'b', 'Ar', 'C2S', 2048, 1, 1, 'b', 'C2P', 2048, 1, 2, 'b', 'add', 'Ar' ] 
            block_4 += [ 'C2S', 512, 1, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 2048, 1, 1, 'b', 'add', 'Ar' ] * 2

            if specBluePrint[1] == 50:
                specs += block_1
                specs += block_2A
                specs += block_2B * 3
                specs += block_3A
                specs += block_3B * 5
                specs += block_4
            elif specBluePrint[1] == 101:
                specs += block_1
                specs += block_2A
                specs += block_2B * 3
                specs += block_3A
                specs += block_3B * 22
                specs += block_4                
            elif specBluePrint[1] == 152:
                specs += block_1
                specs += block_2A
                specs += block_2B * 7
                specs += block_3A
                specs += block_3B * 35
                specs += block_4

        if specBluePrint[0] == 'RsN':
            specs += [ 'PS', 'Pa', 'N', 1000 ]
        else:
            specs += [ 'P2', 'Pa', 'N', 10 ]
        return specs

    if specBluePrint[0] == 'VGG':
        specs += specBluePrint
        # specs.append( np.random.choice(actVariants) )
        specs.append( 'Ar' )

        if specBluePrint[1] == 11: # 32
            specs += ['C2S', 64, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm'] # 16
            specs += ['C2S', 128, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm'] # 8
            specs += ['C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar' ]
            specs += ['P2', 'Pm'] # 4
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar' ]
            specs += ['P2', 'Pm'] # 2
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar' ]

        elif specBluePrint[1] == 13:
            specs += ['C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar' ]
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar' ]
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar' ]

        elif specBluePrint[1] == 16:
            specs += ['C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar']
        elif specBluePrint[1] == 19:
            specs += ['C2S', 64, 3, 1, 'b', 'Ar', 'C2S', 64, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 128, 3, 1, 'b', 'Ar', 'C2S', 128, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar', 'C2S', 256, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar']
            specs += ['P2', 'Pm']
            specs += ['C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar', 'C2S', 512, 3, 1, 'b', 'Ar']
        else:
            print('unknown VGG def', specBluePrint)
            sys.exit()

        specs += ['P2', 'Pm', 'N', 10]
        return specs

    if specBluePrint[0] == 'Anet':
        specs += specBluePrint
        specs += ['C2S', 64, 3, 2, 'Ar', 'P2', 'Pm', 'C2S', 192, 3, 1, 'Ar', 'P2', 'Pm', 'C2S', 384, 3, 1, 'Ar']
        specs += ['C2S', 256, 3, 1, 'Ar', 'C2S', 256, 3, 1, 'Ar', 'P2', 'Pm', 'N', 4096, 'Ar', 'N', 4096, 'Ar', 'N', 10]
        return specs

    if specBluePrint[0] == 'mnC1':
        specs += specBluePrint
        specs += ['C2S', 10, 3, 1, 'Ar', 'C2S', 20, 3, 2, 'Ar', 'C2S', 32, 3, 1, 'Ar','C2S', 10, 3, 2, 'Ar', 'N', 2048, 'Ar', 'N', 10]
        return specs
    if specBluePrint[0] == 'mnC2':
        specs += specBluePrint
        specs += ['C2S', 16, 5, 1, 'b', 'Ar', 'P2', 'Pm', 'C2S', 32, 5, 1, 'b', 'Ar', 'P2', 'Pm', 'N', 10]
        return specs
    if specBluePrint[0] == 'mnD1':
        specs += specBluePrint
        specs += ['N', 1024, 'Ar', 'N', 1024, 'Ar', 'N', 10]
        return specs
    if specBluePrint[0] == 'mnD2':
        specs += specBluePrint
        specs += ['N', 128, 'Ar', 'b', 'N', 128, 'Ar', 'b', 'N', 128, 'Ar', 'b', 'N', 128, 'Ar', 'b', 'N', 10]
        return specs

    minNodes = 512
    maxNodes = 4096

    minGates = 512
    maxGates = 2048

    minBlockLayers = 6
    maxBlockLayers = 10

    convVariants = ['C2S', 'C2V'] # same size or valid region, tensorflow conventions
    poolVariants = ['Pm', 'Pa'] # max pool or average 
    postLayerOps = ['_', 'b' ] # nothing or batchNorm

    filterVariants = [4,8,16,32,64,128,256,512,1024]
    kernelVariants = [1,3,5,7] # removing even numbers gets rid of dilation != 1
    strideVariants = [1,2]

    # to make ranges inclusive, need to add 1
    maxNodes += 1
    maxGates += 1
    maxBlockLayers += 1

    lidx = 0
    isPrescribed = False
    while lidx < len(specBluePrint):
        layer = specBluePrint[lidx]
        skipAct = False

        if layer in ['R', 'L', 'BL']:
            specs.append( layer )
            lidx += 1
            if lidx < len(specBluePrint) and isinstance(specBluePrint[lidx], int):
                specs.append( specBluePrint[lidx] )
                lidx += 1
                isPrescribed = True
            else:
                specs.append( np.random.randint( minGates, maxGates ) )
            isPrescribed = True
            continue
        elif layer == 'P2':
            specs.append( layer )
            lidx += 1
            if lidx < len(specBluePrint) and specBluePrint[lidx] in poolVariants:
                specs.append( specBluePrint[lidx] )
                lidx += 1
            else:
                specs.append( np.random.choice(poolVariants) )
                specs.append('F')
            continue
        elif layer == 'b':
            specs.append( layer )
            lidx += 1
            continue
        elif layer == 'D':
            specs.append( layer ) # D and dropout fraction come together
            lidx += 1
            specs.append( layer ) # if D is listed, should follow with float
            lidx += 1
            continue

        if layer == 'N':
            specs.append(layer)
            lidx += 1
            if lidx < len(specBluePrint) and isinstance(specBluePrint[lidx], int):
                specs.append( specBluePrint[lidx] )
                lidx += 1
                isPrescribed = True
            else:
                specs.append( np.random.randint( minNodes, maxNodes ) )

        elif layer == 'NB':

            specs.append( 'N' )
            specs.append( 4096 )
            chosenAct = np.random.choice(actVariants)
            specs.append( chosenAct )

            nblockVariants = np.random.randint( minNodes, maxNodes )
            layersInBlock = np.random.randint( minBlockLayers, maxBlockLayers ) - 1
            for i in np.arange(layersInBlock):
                specs.append( 'N' )
                specs.append( nblockVariants  )
                # specs.append( 'b' )
                specs.append( chosenAct )
            lidx += 1
            isPrescribed = True

        elif layer == 'CB':
            layersInBlock = np.random.randint( minBlockLayers, maxBlockLayers )

            chosenNumFilters = np.random.choice( filterVariants )
            supposedKernel = np.random.choice( kernelVariants )
            # attempt to fit in memory for older processors:
            while supposedKernel * chosenNumFilters > np.max(filterVariants) * 1.5: 
                supposedKernel = np.random.choice( kernelVariants )
                chosenNumFilters = np.random.choice( filterVariants )

            chosenAct = np.random.choice(actVariants)
            for i in np.arange(layersInBlock):
                specs.append( 'C2S' )
                specs.append( chosenNumFilters )
                specs.append( supposedKernel )
                specs.append( 1 ) # keep stride at 1 for blocks
                if chosenNumFilters >= 512:
                    specs.append( 'b' )
                specs.append( chosenAct )

            if np.random.choice( [2,1,2] ) == 2:
                specs.append( 'P2' )
                specs.append( np.random.choice(poolVariants) )
            lidx += 1
            isPrescribed = True
            skipAct = True

        elif layer[0:2] == 'C2':
            if layer in ['C2V', 'C2S']:
                specs.append( layer )
            else:
                specs.append( np.random.choice(convVariants) )                

            lidx += 1
            if lidx < len(specBluePrint) and isinstance(specBluePrint[lidx], int): # filter and kernel come together
                specs.append( specBluePrint[lidx] )
                lidx += 1
                specs.append( specBluePrint[lidx] )
                lidx += 1
                specs.append( specBluePrint[lidx] )
                lidx += 1                
                isPrescribed = True
            else:

                chosenNumFilters = np.random.choice( filterVariants )
                supposedKernel = np.random.choice( kernelVariants )
                while supposedKernel * chosenNumFilters > np.max(filterVariants) * 1.5:
                    supposedKernel = np.random.choice( kernelVariants )
                    chosenNumFilters = np.random.choice( filterVariants )
                
                specs.append( chosenNumFilters )                    
                specs.append( supposedKernel )
                specs.append( np.random.choice( strideVariants ) )
        else:
            print('{} undefined layer structure reached {} from {}'.format(layer, specs, specBluePrint))
            sys.exit()

        #post layer options
        postOp = np.random.choice(postLayerOps)
        if not isPrescribed and postOp != '_':
            # follow with choice of postlayerOps
            specs.append( postOp )

        # follow with activation function
        if lidx < len(specBluePrint) and specBluePrint[lidx] == 'Asm':
            specs.append( specBluePrint[lidx] )
            lidx += 1            
        elif lidx < len(specBluePrint) and specBluePrint[lidx] in actVariants:
            specs.append( specBluePrint[lidx] )
            lidx += 1
        elif not skipAct:
            specs.append( np.random.choice(actVariants) )

    return specs

def performScan_torch(ds, startingDims, model, outFile, infBatch ):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('torch could not find cuda device, exiting')
        sys.exit()
    model.eval() # set to inference mode
    model.to(device)

    fullBatchSize = 100
    endOfBatch = int(fullBatchSize * infBatch/100.0)
    if startingDims[1] == 224:
        import urllib
        # url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
        url, filename = ("https://raw.githubusercontent.com/pytorch/vision/master/gallery/assets/dog2.jpg", "dog2.jpg")
        try: 
            urllib.URLopener().retrieve(url, filename)
        except: 
            urllib.request.urlretrieve(url, filename)        
        from PIL import Image
        from torchvision import transforms
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).repeat(endOfBatch,1,1,1)
        data = input_batch.to(device)
    elif len(startingDims) == 2:
        data = torch.randn(endOfBatch, startingDims[0], startingDims[1] ).to(device)
    else:
        data = torch.randn(endOfBatch, startingDims[0], startingDims[1], startingDims[1]).to(device)

    many_runs_timeline = TimeLiner()
    enableDAQEvents(ds, outFile)

    ################# Run once to remove model upload delay: ################v
    with torch.no_grad():
        model(data)
    torch.cuda.synchronize()
    time.sleep(0.15)

    ################# Start the acquisition: ################################^
    acqTime_0 = time.time()
    ds.startScan()  

    largeNumber = 7654321

    iter_num = 0
    traceRun = True
    keepLooping = True
    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            while keepLooping:
                torch.cuda.synchronize()
                if traceRun:
                    torch.cuda.FloatTensor(largeNumber).normal_() # leftBound to simplify processing
                    torch.cuda.synchronize()
                    time.sleep(0.001)
                    with torch.no_grad():
                        model(data)
                torch.cuda.synchronize()
                time.sleep(0.0045)
                if traceRun:
                    torch.cuda.FloatTensor(largeNumber).cauchy_() # rightBound
                    torch.cuda.synchronize()
                    time.sleep(0.001)                    
                    iter_num += 1
                    if iter_num == REPEAT_INF:
                        traceRun = False                    
                if ds.isIdle():
                    keepLooping = False
                    time.sleep(0.05)

        with tempfile.NamedTemporaryFile(mode="w+") as f:
            prof.export_chrome_trace(f.name)
            many_runs_timeline.update_timeline_with_json(f)
            ds.write_scanned_buffer(outFile)
            many_runs_timeline.save(outFile)
            ds.disableEvent()
    torch.cuda.empty_cache()
    return

def run_net_torch(daq_scanner, problem, runPrefix, layers, infBatch):

    if problem[0] == 'imageNet':
        currDims = [problem[1], problem[2], problem[2]]
    elif problem[0] == 'imdb':
        num_features = problem[2]
        seq_max_length = problem[1]
        currDims = [seq_max_length, num_features]
    elif len(problem) == 2:
        currDims = [problem[0], problem[1], problem[1]]
    else:
        print('unsupported PROBLEM %s' % problem )
        raise ValueError

    problemDims = currDims[:]
    model = compose_net.configure_problem_torch( currDims, layers )

    charLimit = 60
    filenameFromRunPrefix = runPrefix[0:charLimit]
    if len(filenameFromRunPrefix) == charLimit:
        outFile = filenameFromRunPrefix + ']_ts_scan_trace.txt'
    else:
        outFile = filenameFromRunPrefix + '_ts_scan_trace.txt'
    with open(outFile, 'w') as f:
        f.write('{}\n+++\n'.format(runPrefix))

    performScan_torch(daq_scanner, problemDims, model, outFile, infBatch )
    return

def main():
    print('torch', torch.__version__)
    origStart = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for scanSet in np.arange(2):
        timestring = timestamp + '_' + str(scanSet) + '/'
        if not os.path.isdir(timestring):
            os.mkdir(timestring)

        numScanSecs = 1.0
        daq_scanner = DAQSession( numScanSecs )

        moreData = 1 # test
        if scanSet == 0: 
            moreData = 10 # train

        netList =[]
        # imageSizes = [28,32,64,96]
        imageSizes = [128]
        for md in np.arange(moreData):
            for ci in [1,3]: # grayscale & 3D images
                for ii in imageSizes:  # this doesnt account for image size 32 vs 28
                    for spec_blue in LAYERS_DEFS_TORCH:
                        spec = genNetSpecs( spec_blue ) # keep same layer-widths for different batch sizes
                        maxlen = -1
                        if spec[0] in ['L', 'R', 'BL']:
                            maxlen = np.random.randint(16, 32)
                            num_feats = np.random.randint(3, 8)
                            for inf_size in INF_CENT:
                                netList.append([spec, inf_size, maxlen, num_feats] )
                        else:
                            for inf_size in INF_CENT:
                                netList.append([spec, inf_size, ci, ii] )

        runIdx = 0
        numRuns = len(netList)
        numRunDigits = len(str(numRuns))
        for net in netList:
            
            if scanSet == 0: 
                break # skip for test sets

            specs = net[0]
            inf_size = net[1]

            if 'RsN' in specs:
                problem = ['imageNet', 3, 224]
                problemName = str(3)
                imgSide = str(224)
            elif specs[0] in  ['VGG', 'Anet', 'RsNc']:
                problem = [3, 32]
                problemName = str(3)
                imgSide = str(32)
            elif specs[0] in ['L', 'R', 'BL']:
                    maxlen = net[2]
                    num_feats = net[3]
                    problemName = str(num_feats)
                    imgSide = str(maxlen)
                    problem = ['imdb', maxlen, num_feats]
            else:
                problem = [net[2], net[3]]
                problemName = str(net[2])
                imgSide = str(net[3]).zfill(3)

            runPrefix = timestring + str(runIdx).zfill(numRunDigits) + "_s_" + imgSide + "_t_i_" + problemName
            runPrefix = runPrefix + "_b" + str(inf_size).zfill(3) + "_" + str(specs).replace(" ", "")

            if specs[0] not in ['RsN', 'RsNc', 'L', 'VGG', 'Anet']:
                runPrefix = runPrefix[:-1] + ',\'N\',10,\'Asm\']' # make 10-class classifier

            print('attempting: {}'.format(runPrefix))
            runStart = time.time()
            with torch.no_grad():
                run_net_torch(daq_scanner, problem, runPrefix, specs, inf_size) 
                torch.cuda.empty_cache()
                if runIdx == 0:
                    run_net_torch(daq_scanner, problem, runPrefix, specs, inf_size) # run twice since scanner & profiler always slow to capture the first
            torch.cuda.empty_cache()
            print("{} Finished training {}/{}, {} in {:.2f}s\n".format(scanSet, str(runIdx+1).zfill(numRunDigits), numRuns, runPrefix, time.time() - runStart))
            runIdx += 1

        daq_scanner.close()
        print("Finished Comparing {} Networks in {:.2f}s".format(scanSet, time.time() - origStart))
    return

if __name__ == '__main__':
    main()