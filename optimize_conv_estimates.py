import sys
import numpy as np

def checkFile( filename ):
    if not os.path.isfile(filename):
        print('exiting because cannot find {}'.format(filename))
        sys.exit() 

def checkValid( dataSet ):
    if (np.isinf(dataSet)).any() or (np.isnan(dataSet)).any():
        print( 'inf or nan found in dataSet, exiting')
        sys.exit()

def warmStartWithFile( inputImgSize, predictedLayerVarFile ):
    folderPrefix = predictedLayerVarFile.split("/")[0]
    outFile = folderPrefix + '/genOptimizedLayerVars.txt'

    useLog = False
    if useLog:
        import docplex.cp.model as cpx
    else:
        import docplex.mp.advmodel as cpx

    checkFile(predictedLayerVarFile)
    layerEstimates = np.genfromtxt(predictedLayerVarFile)
    checkValid( layerEstimates )

    # ii, io, k, pad, stride, type (reg, post-pool, residual)
    layerIi = layerEstimates[:,0]
    layerIo = layerEstimates[:,1]
    layerK = layerEstimates[:,2]
    layerP = layerEstimates[:,3]
    layerS = layerEstimates[:,4]
    layerType = layerEstimates[:,5]

    layerIi_compare = np.copy(layerIi)
    layerIo_compare = np.copy(layerIo)
    layerK_compare = np.copy(layerK)
    layerP_compare = np.copy(layerP)

    # Variable boundaries
    kl = 1
    ku = 7
    pl = 0
    pu = 5
    il = 1
    sl = 1
    su = 2
    wl = 0
    dilation = 1
    numLayers = len(layerK)
    layers = range(0,numLayers)

    if useLog: 
        opt_model = cpx.CpoModel(name='cnn_layer_params')
    else:
        opt_model = cpx.Model(name='cnn_layer_params')

    kernel_vars = {
        i: opt_model.integer_var(
            name="kernel_{}".format(i),
            lb=kl, 
            ub=ku)
        for i in layers }

    kextra_vars = {
        i: opt_model.integer_var(
            name="kextra_{}".format(i),
            lb=wl)
        for i in layers }

    padding_vars = {
        i: opt_model.integer_var(
            name="pad_{}".format(i),
            lb=pl, 
            ub=pu)
        for i in layers }

    img_in_vars = {
        i: opt_model.integer_var(
            name="img_in_{}".format(i),
            lb=il)
        for i in layers }

    img_out_vars = {
        i: opt_model.integer_var(
            name="img_out_{}".format(i),
            lb=il)
        for i in layers }

    # images cannot get bigger ## this is handled by CNN params contraint, but not with bad stride and dilation
    constraints_img_nonincr = { 
        i: opt_model.add_constraint(
            ct=img_in_vars[i] >= img_out_vars[i],
            ctname="constraint_img_nonincr_{0}".format(i))
        for i in layers}

    # odd kernels only
    constraints_kernel_odd = { 
        i: opt_model.add_constraint(
            ct=kernel_vars[i] - 2 * kextra_vars[i] == 1,
            ctname="constraint_kernel_odd_{0}".format(i))
        for i in layers}

    # doesnt end too small, positive
    opt_model.add_constraint(
            ct=img_out_vars[len(img_out_vars)-1] >= 1,
            ctname="constraint_img_last_{0}".format(len(img_out_vars) - 1))    

    # image_out matches image_in
    constraints_img_out_is_in = {}
    constraints_branch_matches_A = {}
    constraints_branch_matches_B = {}
    constraints_branch_matches_C = {}
    for i, layer in enumerate(layerType):
        if i == 0: # know input size
            constraints_img_out_is_in[i] = opt_model.add_constraint(
                    ct=img_in_vars[i] == inputImgSize,
                    ctname="constraint_img_out_is_in_{0}".format(i))

        #image_out matches image_in
        elif layer == 1: # pool layer, halved input from output
            constraints_img_out_is_in[i] = opt_model.add_constraint(
                    ct=img_in_vars[i] * 2 == img_out_vars[i-1],
                    ctname="constraint_img_out_is_in_{0}".format(i))
        elif layer == 2: # skip connect, doubled input from output
            layerS[i] = 2 # stride is 2 on skip connects
            layerS[i-2] = layerS[i]
            constraints_branch_matches_A[i] = opt_model.add_constraint(
                    ct=img_in_vars[i] == img_in_vars[i-2],
                    ctname="constraint_branch_matches_A{0}".format(i))
            constraints_branch_matches_B[i] = opt_model.add_constraint(
                    ct=img_out_vars[i] == img_out_vars[i-2],
                    ctname="constraint_branch_matches_B{0}".format(i))
            constraints_branch_matches_C[i] = opt_model.add_constraint(
                    ct=img_in_vars[i] == img_out_vars[i] * 2,
                    ctname="constraint_branch_matches_C{0}".format(i))
        else:
            constraints_img_out_is_in[i] = opt_model.add_constraint(
                    ct=img_in_vars[i] == img_out_vars[i-1],
                    ctname="constraint_img_out_is_in_{0}".format(i))

    # layer params constraint
    constraints_cnn_params = { 
        i: opt_model.add_constraint(
            ct=img_in_vars[i] + 2 * padding_vars[i] - dilation * kernel_vars[i] + dilation == layerS[i] * img_out_vars[i],
            ctname="constraint_cnn_params_{0}".format(i))
        for i in layers}


    k_diff_sqd = opt_model.sum_squares( layerK_compare[i] - kernel_vars[i] for i in layers)

    p_diff_sqd = opt_model.sum_squares( layerP_compare[i] - padding_vars[i] for i in layers)

    if useLog:
        ii_diff_sqd = opt_model.sum_squares( opt_model.log( opt_model.abs( layerIi_compare[i] - img_in_vars[i] for i in layers) ) )
        io_diff_sqd = opt_model.sum_squares( opt_model.log( opt_model.abs( layerIo_compare[i] - img_out_vars[i] for i in layers) ) )
    else:
        io_diff_sqd = opt_model.sum_squares( layerIo_compare[i] - img_out_vars[i] for i in layers)
        ii_diff_sqd = opt_model.sum_squares( layerIi_compare[i] - img_in_vars[i] for i in layers)

    # even weights, though these could be tuned with heuristics
    objectiveWeights = np.array([ 1, 1, 1, 1 ])
    oW = objectiveWeights/objectiveWeights.sum()

    opt_model.minimize( oW[0] * k_diff_sqd + oW[1] * p_diff_sqd + oW[2] * ii_diff_sqd + oW[3] * io_diff_sqd )
    solution = opt_model.solve(log_output=False); # silent

    layerSolutions = np.full((layerEstimates.shape[0], layerEstimates.shape[1]), np.inf)
    for i in layers:
        layerSolutions[i,0] = solution.get_value( "img_in_{}".format(i) )
        layerSolutions[i,1] = solution.get_value( "img_out_{}".format(i) )
        layerSolutions[i,2] = solution.get_value( "kernel_{}".format(i) )
        layerSolutions[i,3] = solution.get_value( "pad_{}".format(i) )
        layerSolutions[i,4] = layerS[i]
        layerSolutions[i,5] = solution.get_objective_value()
    checkValid(layerSolutions)
    with open(outFile, "w") as f:
        np.savetxt(f, layerSolutions)
    return

def main():
    if len(sys.argv) == 3:
        warmStartWithFile( int(sys.argv[1]), sys.argv[2], '0' )
    else:
        print('missing windowLength argument, or too many arguments')
        print('insufficient arguments %d' % len(sys.argv))
        for i in range(len(sys.argv)):
            print('%d %s' % (i, sys.argv[i]))

if __name__ == '__main__':
    main()
