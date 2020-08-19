#!/usr/bin/env python3

from sklearn.svm import SVC
import sys
import pickle


if __name__ == '__main__':
    # Create an array of train data.
    X = []
    y = []

    # Read training data from STDIN, and append them to the train data array.
    for line in sys.stdin:
        line = line.strip('\n')
        if line:
            # Split the line with space characters.
            fields = line.split('|')
            if fields[3] == '1':      # 2nd stage model is fed only with positive drug-drug interactions
                # Append the item features to the item sequence.
                # fields are:  0=sid, 1=id_e1, 2=id_e2, 3=ddi, 4=ddi_type, 5=features
                #features = (fields[5].split(' ')).append(fields[6].split)
                item = [v.split("=")[1] for v in fields[5:]]
                X.append(item)

                # Append the label to the label sequence.
                y.append(fields[4])

   # Create instance of SVM classifier
    model = SVC(
          C=1.2,
          kernel='rbf',
          gamma=0.001,
          class_weight='balanced')
    model.fit(X, y)

    # save the model to disk
    filename = 'model_2st_stage.sav'
    pickle.dump(model, open(filename, 'wb'))