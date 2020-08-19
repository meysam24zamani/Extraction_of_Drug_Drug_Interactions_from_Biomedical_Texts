#!/usr/bin/env python3

import sys
import pickle


if __name__ == '__main__':

    # Load model from file
    filename = 'model_2st_stage.sav'
    model = pickle.load(open(filename, 'rb'))

    # Create an array of train data.
    X = []
    info = []
    result = []

    # Read training data from STDIN, and append them to the train data array.
    for line in sys.stdin:
        line = line.strip('\n')
        if line:
            # Split the line with space characters.
            fields = line.split('|')
            if fields[3] == '1':    # Predict only positive interactions
                # Append the item features to the item sequence.
                # fields are:  0=sid, 1=id_e1, 2=id_e2, 3=ddi, 4=ddi_type, 5=features
                #features = fields[5].split('|')
                X = [v.split("=")[1] for v in fields[5:]]

                # Append sentence and entity information
                sid,id_e1,id_e2,ddi,ddi_type = fields[0],fields[1],fields[2],fields[3],fields[4]

                prediction = model.predict([X])
                result.append("|".join([sid, id_e1, id_e2, ddi, prediction[0]]))
            else:
                result.append(line)

    print(*result, sep="\n")