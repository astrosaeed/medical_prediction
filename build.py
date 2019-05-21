import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split 
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from keras import backend as K
import sys


def coeff_determination(y_true, y_pred):
	SS_res =  K.sum(K.square( y_true-y_pred )) 
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class Train:
	def __init__(self):
		self.file_name = 'insurance.csv'

	def get_data(self):

		 df = pd.read_csv(self.file_name)

		 return df


	def oneHotEncode(self,df):
		for col in list(df):
			if( df[col].dtype == np.dtype('object')):
				dummies = pd.get_dummies(df[col],prefix=col)
				df = pd.concat([df,dummies],axis=1)

				 #drop the encoded column
				df.drop([col],axis = 1 , inplace=True)
		return df

	def normalize(self,df):

		x = df.values #returns a numpy array
		min_max_scaler = preprocessing.MinMaxScaler()
		x_scaled = min_max_scaler.fit_transform(x)
		df = pd.DataFrame(x_scaled)

		return df



	def split_data(self,df):
		X_train, X_test=train_test_split(df, test_size=0.33, random_state=42 )
		return X_train, X_test


	def extract_label(self,df, labelindex):
		label = df[labelindex]
		data = df.drop([labelindex], axis=1)
		#data = df

		return data, label


	def build_model(self,train_data, train_labels, test_data,test_labels,layer_nodes):


		model = Sequential()
		# The Input Layer :
		model.add(Dense(layer_nodes[0], kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

		# The Hidden Layers :
		model.add(Dense(layer_nodes[1], kernel_initializer='normal',activation='relu'))
		model.add(Dense(layer_nodes[2], kernel_initializer='normal',activation='relu'))
	#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))#
	#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

		# The Output Layer :
		model.add(Dense(1, kernel_initializer='normal',activation='linear'))

		#Compile the network :
		model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error',coeff_determination])
		model.summary()
		print layer_nodes[0]
		os.makedirs(str(layer_nodes[0])+'_'+str(layer_nodes[1])+'_'+str(layer_nodes[2]))
		checkpoint_name = str(layer_nodes[0])+'_'+str(layer_nodes[1])+'_'+str(layer_nodes[2])+'/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
		checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
		callbacks_list = [checkpoint]

		history=model.fit(train_data, train_labels, epochs=500, batch_size=64, validation_data=(test_data,test_labels), callbacks=callbacks_list)
		print (history.history.keys())
		#sys.exit(0)
		# summarize history for loss
		plt.plot(history.history['mean_absolute_error'])
		plt.plot(history.history['val_mean_absolute_error'])
		plt.title('mean absolute error')
		plt.ylabel('loss')
		plt.ylim(0, 0.1)
		plt.xlabel('epoch')
		plt.legend(['test', 'train'], loc='upper left')
		plt.savefig(str(layer_nodes[0])+'_'+str(layer_nodes[1])+'_'+str(layer_nodes[2])+'/loss.png')
		plt.close()

		plt.plot(history.history['coeff_determination'])
		plt.plot(history.history['val_coeff_determination'])
		plt.title('R2')
		plt.ylabel('R2')
		#plt.ylim(0, 0.1)
		plt.xlabel('epoch')
		plt.legend(['test', 'train'], loc='upper left')
		plt.savefig(str(layer_nodes[0])+'_'+str(layer_nodes[1])+'_'+str(layer_nodes[2])+'/R2.png')
		plt.close()


def main():

	instant= Train()
	train_set , test_set =instant.split_data(instant.normalize(instant.oneHotEncode(instant.get_data())))
	print train_set.shape
	print test_set.shape
	#print train.iloc[:,3]  # 3rd colum is the label
	data_train,label_train = instant.extract_label(train_set,3)
	data_test,label_test = instant.extract_label(test_set,3)

	#npdata = np.array(data_train) 
	#nplabel =np.array (label_train)
	#print type(nplabel)
	#nplabel = nplabel.reshape(-1,1) 
	#print nplabel.shape
	#print npdata.shape
	for nodes in [(8,16,32),(8,32,16), (12,16,12),(12,20,8)]:
		instant.build_model(np.array(data_train), np.array (label_train).reshape(-1,1), np.array(data_test), np.array (label_test).reshape(-1,1),nodes)

	#model_test=load_model('Weights-994--0.02692.hdf5')
	#build_model(data, label)
	#print np.array(test)[1,:].shape
	#print 'true label'
	#print np.array(label_test).reshape(-1,1).shape
	#print 'data_test shape'
	#print data_test.shape
	#print np.array(data_test)[1,:].reshape(1,-1).shape
	#predictions = model_test.predict(np.array(data_test)[1,:].reshape(1,-1), verbose=1)

	#print predictions
	#print np.array(label_test).reshape(-1,1)[1,0]

if __name__ =="__main__":
	main()
