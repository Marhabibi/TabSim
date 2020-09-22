from inputvar import *
epochs, md1,lrs,inp,emb_loc = inputvarsTrain(sys.argv[1:],sys.argv[0])
from input_transformation import *

import gensim
from gensim.models.keyedvectors import KeyedVectors

from keras_self_attention import SeqSelfAttention
from keras.layers import Input,LSTM,Bidirectional,Concatenate,Embedding,Reshape,TimeDistributed,Flatten,Permute,Dense,Lambda,Conv1D,MaxPooling1D,Dropout,Conv2D,BatchNormalization,Activation

from keras.callbacks import ModelCheckpoint
from keras.models import Model

from keras.optimizers import RMSprop


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(813306)	
from tensorflow import set_random_seed
set_random_seed(2)

# compute similarity accuracy (as similar or dissimilar) with a fixed threshold on distances
def accuracy(y_true, y_pred):
    
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))	

# compute distance between two table representation
def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))
	
def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

# compute Siamese loss
def siamese_loss(y_true,y_pred):
	
	margin = 1
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(0.5*(1-y_true) * square_pred + 0.5*( y_true) * margin_square)

# read embedding vectors
def read_emb(dictionary,file_emb):
	
	embedding_matrix = np.random.random((len(dictionary) + 1, 200))
	modelemb = gensim.models.KeyedVectors.load_word2vec_format(file_emb, binary=True)
	w2v = dict(zip(modelemb.wv.index2word, modelemb.wv.syn0))

	for j, i in dictionary.items():
		if w2v.get(j) is not None:
			embedding_matrix[i] = w2v[j]
			
	return embedding_matrix	

# a model for sequential input representation (caption and cell content)
def model_seq(input_shape,vocab_size,emb_vec,embedding_flag):
	
	X_in = Input(input_shape)
	
	if int(embedding_flag) == 1:
		X_emb = Embedding(vocab_size,200,weights = [emb_vec], input_length = input_shape[-1], trainable = True,mask_zero = True)(X_in)#,mask_zero = True)(X_in)	
	else:
		X_emb = Embedding(vocab_size,200, input_length = input_shape[-1], trainable = True,mask_zero = True)(X_in)
	X_lstm = Bidirectional(LSTM(50, return_sequences = False))(X_emb)
	
	return Model(X_in,X_lstm)

# attention layers for column or row representations
def model_attention(CL,input_shape):

	X_in = Input(input_shape)
	
	X_att = SeqSelfAttention(attention_width=CL,attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation=None)(X_in)

	return Model(X_in,X_att)


# tabular model	
def model_table(in_shape,vocab_size,emb_matrix,embedding_flag,COL_L, ROW_L):
	
	X_in = Input(in_shape)
	# cell representation
	X_in_reshape = Reshape((in_shape[-3]*in_shape[-2],in_shape[-1]),input_shape = (in_shape[-3],in_shape[-2],in_shape[-1]))(X_in)	
	X_in_cells = TimeDistributed(model_seq((in_shape[-1],),vocab_size,emb_matrix,embedding_flag))(X_in_reshape)
	X_in_reshape = Reshape((in_shape[-3],in_shape[-2],100), input_shape=(in_shape[-3]*in_shape[-2],100))(X_in_cells)
	
	# attention layer on rows and columns
	X_in_att = TimeDistributed(model_attention(COL_L,(in_shape[-2],100)))(X_in_reshape)
	X_in_in_att = Reshape((in_shape[-3],in_shape[-2]*100,),input_shape=(in_shape[-3],in_shape[-2],100,))(X_in_att)
	X_in_in_dense = Dense(100)(X_in_in_att)
	X_in_batchnorm = BatchNormalization()(X_in_in_dense)
	X_in_activ = Activation('relu')(X_in_batchnorm)
	X_in_in_in_att = SeqSelfAttention(attention_width=ROW_L,attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,attention_activation=None)(X_in_activ)
	X_in_att_reshape = Reshape((in_shape[-3]*100,),input_shape=(in_shape[-3],100,))(X_in_in_in_att)
	
	# dense layer on the final representation
	dense_out = Dense(100,activation = 'relu')(X_in_att_reshape)
	dense_out_drop = Dropout(0.1)(dense_out)

	return Model(X_in, dense_out_drop) 
	
# similarity model
def final_model(CAP_L,COL_L, ROW_L, CELL_L,embedding_flag,emb_matrix_cap,emb_matrix_tab,vocab):
	
	# caption representation initialization
	m_c = model_seq((CAP_L,),len(vocab)+1,emb_matrix_cap,embedding_flag)
	# table representation initialization
	m_t_v = model_table((COL_L,ROW_L,CELL_L,),len(vocab)+1,emb_matrix_cap,embedding_flag, COL_L, ROW_L)
		
	# captions and tabular inputs
	input_c1 = Input((CAP_L,))
	input_c2 = Input((CAP_L,))

	input_t1 = Input((COL_L,ROW_L,CELL_L,))
	input_t2 = Input((COL_L,ROW_L,CELL_L,))
		
	# represent captions
	in_c1 = m_c(input_c1)
	in_c2 = m_c(input_c2)

	# represent tabular
	in_t1 = m_t_v(input_t1)
	in_t2 = m_t_v(input_t2)
	
	# concatenate tabular and caption representations
	merged_results_t1 = Concatenate(axis = -1)([in_c1,in_t1])
	merged_results_t2 = Concatenate(axis = -1)([in_c2,in_t2])
		
	# measure distance between two table representations
	distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([merged_results_t1, merged_results_t2])
	
	return Model([input_c1,input_c2,input_t1,input_t2],distance)

if __name__ == "__main__":
	
	# variable initialization
	MAX_COL = 9
	MAX_COL_LENGTH = 9
	MAX_CELL_LENGTH = 4
	MAX_CAP_LENGTH = 12
	embedding_flag = 1
	learning_r = float(lrs)
	epoch_s = int(epochs)
	opt = keras.optimizers.RMSprop(lr=learning_r, rho=0.9)
	filepath= md1+"/model-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}-{loss:.4f}-{accuracy:.4f}.hdf5"
	
	# read input tables
	tab1_train,tab2_train,cap1_train,cap2_train,y_train,_,_,_,vocab = transform_tables(inp, "train")
	y_train = [0 if item == 1 else 1 for item in y_train]
	
	# read embedding vectors
	emb_matrix_cap = read_emb(vocab,emb_loc)
	emb_matrix_tab = read_emb(vocab,emb_loc)
	
	# model initialization and training
	final_m = final_model(MAX_CAP_LENGTH, MAX_COL, MAX_COL_LENGTH, MAX_CELL_LENGTH,embedding_flag,emb_matrix_cap,emb_matrix_tab,vocab)
	final_m.compile(loss = siamese_loss, optimizer = opt,metrics=[accuracy])
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min',period = 1)
	hist=final_m.fit([cap1_train,cap2_train,tab1_train,tab2_train],y_train,validation_split =0.25, shuffle = True,epochs = epoch_s,callbacks=[checkpoint])
