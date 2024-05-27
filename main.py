import src.sshiba as sshiba
import src.fileOperations as fop
import argparse
import numpy as np
import os

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Options for create shyntehtic data")

    parser.add_argument('-p', '--path', help='Directory with data files', required=True)
    parser.add_argument('-s', '--saveDir', help='directory to save model and predict', required=True)

    args = parser.parse_args()

    # Generate data
    data = sshiba.dataModel()
    data.loadData(args.path)
    data.normalizeData()

    #trainData
    X_tr = data.getXtr()
    X_tst = data.getXtst()
    Y_tr = data.getYtr()
    Y_tst = data.getYtst()
        
    myKc = 20            # number of latent features
    max_it = int(5*1e4)  # maximum number of iterations
    tol = 1e-6           # tolerance of the stopping condition (abs(1 - L[-2]/L[-1]) < tol)

    myModel = sshiba.SSHIBA(myKc)
    X0_tr = myModel.struct_data(X_tr, 'reg', 0)
    X1_tr = myModel.struct_data(Y_tr, 'reg', 0)
    X0_tst = myModel.struct_data(X_tst, 'reg', 0)
    X1_tst = myModel.struct_data(Y_tst, 'reg', 0)
    myModel.fit(X0_tr, X1_tr, max_iter = max_it, tol = tol, Y_tst = X1_tst, X_tst = X0_tst, mse = 1)
    print('Final MSE %.3f' %(myModel.mse[-1]))

    # Predict
    print('Predicting...')
    Y_pred_tst,var = myModel.predict([0],1,X0_tst)
    print('Saving model...')    
    fop.savingPickelFile (os.path.join(args.saveDir,'model.pkl'), myModel)
    fop.savingPickelFile (os.path.join(args.saveDir,'Y_pred_tst.pkl'), Y_pred_tst)

