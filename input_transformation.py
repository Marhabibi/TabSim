import pickle
import numpy as np

import keras
from keras import backend as K
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.text import text_to_word_sequence,Tokenizer

def table_vocab(X,tokenizer):
	MAX_COL=9
	MAX_COL_LENGTH=9
	MAX_CELL_LENGTH=4
	data = np.zeros((len(X), MAX_COL, MAX_COL_LENGTH,MAX_CELL_LENGTH), dtype='int32')
	
	for i, table in enumerate(X):
		for j, col in enumerate(table):
			if j< MAX_COL:
				for k, cell in enumerate(col):
					if k<MAX_COL_LENGTH:
						z=0
						for _,word in enumerate(text_to_word_sequence(cell,lower=True)):
							if z<MAX_CELL_LENGTH and tokenizer.word_index.get(word) is not None:
								data[i,j,k,z] = tokenizer.word_index[word]
								z=z+1

	return data

def caption_vocab(X,tokenizer):
	
	MAX_CAP_LENGTH=12
	data = np.zeros((len(X), MAX_CAP_LENGTH), dtype='int32')
	
	for i, caption in enumerate(X):
		z=0
		caplen = len(text_to_word_sequence(caption))
		for _,word in enumerate(text_to_word_sequence(caption,lower=True)):
			if z<MAX_CAP_LENGTH and tokenizer.word_index.get(word) is not None:
				data[i,z] = tokenizer.word_index[word]
				z=z+1                    
	
	
	return data

def transform_tables(inp,config):
	
	with open(inp,'rb') as f:
		tables = pickle.load(f)

	ns1,ns2,y,rs,cs1,cs2,ts1,ts2=[],[],[],[],[],[],[],[]
	
	for i in range(0,5):
		cs1 = cs1 + [c1 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		cs2 = cs2 + [c2 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		ts1 = ts1 + [t1 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		ts2 = ts2 + [t2 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		y = y + [l for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		ns1 = ns1 + [n1 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		ns2 = ns2 + [n2 for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
		rs = rs + [r for (c1,c2,t1,t2,l,n1,n2,r) in tables[i]]
	
	
	texts = ["XXX"] + [' '.join(text_to_word_sequence(' '.join(sum(t1,[])),lower=True)) for t1 in ts1] + [' '.join(text_to_word_sequence(' '.join(sum(t2,[])),lower=True)) for t2 in ts2] + [' '.join(text_to_word_sequence(c1,lower=True)) for c1 in cs1] + [' '.join(text_to_word_sequence(c2,lower=True)) for c2 in cs2]
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(texts)


	if config == "train":
		
		cs1 = cs1[0:int(len(cs1)*0.75)]
		cs2 = cs2[0:int(len(cs2)*0.75)]
		ts1 = ts1[0:int(len(ts1)*0.75)]
		ts2 = ts2[0:int(len(ts2)*0.75)]
		ns1 = ns1[0:int(len(ns1)*0.75)]
		ns2 = ns2[0:int(len(ns2)*0.75)]
		rs = rs[0:int(len(rs)*0.75)]
		y = y[0:int(len(y)*0.75)]

	else:
		cs1 = cs1[int(len(cs1)*0.75):]
		cs2 = cs2[int(len(cs2)*0.75):]
		ts1 = ts1[int(len(ts1)*0.75):]
		ts2 = ts2[int(len(ts2)*0.75):]
		ns1 = ns1[int(len(ns1)*0.75):]
		ns2 = ns2[int(len(ns2)*0.75):]
		rs = rs[int(len(rs)*0.75):]
		y = y[int(len(y)*0.75):]

	data1 = table_vocab(ts1,tokenizer)
	data2 = table_vocab(ts2,tokenizer)
	cap1 = caption_vocab(cs1,tokenizer)
	cap2 = caption_vocab(cs2,tokenizer)
	
	return data1,data2,cap1,cap2,np.array(y),ns1,ns2,rs,tokenizer.word_index
	
