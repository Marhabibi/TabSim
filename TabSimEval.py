from inputvar import *
md1,output,inp = inputvarsEval(sys.argv[1:],sys.argv[0])
from input_transformation import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import set_random_seed
set_random_seed(2)

import random
np.random.seed(813306)	

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import load_model
from keras_self_attention import SeqSelfAttention

def siamese_loss(y_true,y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(0.5*(1-y_true) * square_pred + 0.5*( y_true) * margin_square)

if __name__ == "__main__":
	
	# variable initialization
	MAX_COL = 9
	MAX_COL_LENGTH = 9
	MAX_CELL_LENGTH = 4
	MAX_CAP_LENGTH = 12

	# read input tables for measuring similarity
	tab1_test,tab2_test,cap1_test,cap2_test,y_test,n1_test,n2_test,rs_test,vocab = transform_tables(inp, "test")	
	
	# load model
	model = load_model(md1,custom_objects={'SeqSelfAttention': SeqSelfAttention,'siamese_loss':siamese_loss})
	# assign label 0 to similar tables and 1 to dissimilar ones
	y_test=[0 if item == 1 else 1 for item in y_test]
	
	# predict similarity scores
	y_prec=model.predict([cap1_test,cap2_test,tab1_test,tab2_test])
	
	# binarized similarity scores to similar (0) and dissimilar (1) based on a threshold 0.5
	y_prediction = [int(pred>0.5) for pred in list(y_prec)]
	
	# write predictions in a csv file
	refs_preds = pd.DataFrame([(n1,n2,r,p) for n1,n2,r,p in zip(n1_test,n2_test,y_test,y_prediction)], columns = ["t1","t2","reference","prediction"])
	refs_preds.to_csv(output+".csv",index=False)
	
	# report the performance scores
	print(classification_report(y_test,y_prediction,digits = 5))
	print(confusion_matrix(y_test,y_prediction))

	