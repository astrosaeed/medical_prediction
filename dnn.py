import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split 

file_name = 'insurance.csv'

def get_data():

	 df = pd.read_csv(file_name)

	 return df


def oneHotEncode(df):
    for col in list(df):
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

             #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

def normalize(df):

	x = df.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled)

	return df



def split_data(df):
	X_train, X_test=train_test_split(df, test_size=0.33, random_state=42 )
	return X_train, X_test


def extract_label(df, labelindex):
	label = df[labelindex]
	data = df.drop([labelindex], axis=1)
	#data = df

	return data, label


def build_model(train_data, train_labels):


	model = Sequential()
	# The Input Layer :
	model.add(Dense(16, kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

	# The Hidden Layers :
	model.add(Dense(32, kernel_initializer='normal',activation='relu'))
#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))#
#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

	# The Output Layer :
	model.add(Dense(1, kernel_initializer='normal',activation='linear'))

	#Compile the network :
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
	model.summary()

	checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
	checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
	callbacks_list = [checkpoint]

	model.fit(train_data, train_labels, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


def main():

	train , test =split_data(normalize(oneHotEncode(get_data())))
	print train.shape
	print test.shape
	#print train.iloc[:,3]  # 3rd colum is the label
	data,label = extract_label(train,3)


	npdata = np.array(data) 
	nplabel =np.array (label)
	print type(nplabel)
	#nplabel = nplabel.reshape(-1,1) 
	print nplabel.shape
	print npdata.shape
	build_model(npdata , nplabel)

	#build_model(data, label)




if __name__ =="__main__":
	main()
