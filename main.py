import src.sshiba as sshiba
import argparse
import numpy as np

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Options for create shyntehtic data")

    parser.add_argument('-p', '--path', help='Directory with data files', required=True)

    args = parser.parse_args()

    # Generate data
    data = sshiba.dataModel()
    data.loadData(args.path)
    data.normalizeData()

    #trainData
    myKc = 20            # number of latent features
    max_it = int(5*1e4)  # maximum number of iterations
    tol = 1e-6           # tolerance of the stopping condition (abs(1 - L[-2]/L[-1]) < tol)
    prune = 1            # whether to prune the irrelevant latent features

    X_tr = data.getXtr()
    X_tst = data.getXtst()
    Y_tr = data.getYtr()
    Y_tst = data.getYtst()
        
    X = np.vstack((X_tr,X_tst))

    myModel = sshiba.SSHIBA(myKc, prune)
    X0 = myModel.struct_data(X, 0, 0)
    X1_tr = myModel.struct_data(Y_tr, 0, 0)
    X1_tst = myModel.struct_data(Y_tst, 0, 0)
    myModel.fit(X0, X1_tr, max_iter = max_it, tol = tol, Y_tst = X1_tst, mse = 1)
    print('Final MSE %.3f' %(myModel.mse[-1]))