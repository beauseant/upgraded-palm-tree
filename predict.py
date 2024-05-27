import src.sshiba as sshiba
import argparse
import numpy as np
import pickle

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser(description="Options for create shyntehtic data")

    parser.add_argument('-f', '--file', help='Directory with data files', required=True)
    args = parser.parse_args()

    #load pickle
    try:
        with open(args.file, 'rb') as f:
            myModel = pickle.load(f)
    except Exception as e:
        print ('Error loading model %s' % e)
        exit()  
X0_tst = myModel.struct_data(X_tst, 0, 0)
Y_pred_tst,var = myModel.predict([0],1,0,X0_tst)

