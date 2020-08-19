#!/usr/bin/env python3

import sys
import pickle


if __name__ == '__main__':

    # Load model from file
    filename = 'model_1st_stage.sav'
    model = pickle.load(open(filename, 'rb'))

    # Create an array of train data.
    X = []
    info = []

    # Read training data from STDIN, and append them to the train data array.
    for line in sys.stdin:
        line = line.strip('\n')
        if line:
            # Split the line with space characters.
            fields = line.split('|')

            # Get the numerical item features for prediction.
            # fields are:  0=sid, 1=id_e1, 2=id_e2, 3=ddi, 4=ddi_type, 5...N = features
            X = [v.split("=")[1] for v in fields[5:]]

            # Get sentence, entity information and features (passed on to 2nd stage SVM if 1st stage prediciton positive)
            sid,id_e1,id_e2,ddi,ddi_type = fields[0],fields[1],fields[2],fields[3],fields[4]
            features = fields[5:]

            prediction = model.predict([X])

            if prediction[0] == "0":
                print("|".join([sid, id_e1, id_e2, "0", "null"]))
            else:
                if ddi_type == "null": # If prediction is false positive, interaction type will be changed to "int"
                    print(sid, id_e1, id_e2, "1", "int","|".join(features), sep="|")
                else:
                    print(sid, id_e1, id_e2, "1", ddi_type,"|".join(features), sep="|")