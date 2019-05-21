import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split 
import seaborn as sb
import matplotlib.pyplot as plt

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
def extract_label(df, labelindex):
	label = df[labelindex]
	data = df.drop([labelindex], axis=1)
		#data = df

	return data, label

def main():

	train  =oneHotEncode(get_data())
	del train['charges']
	#data_train,label_train = extract_label(train,0)
	
	C_mat = train.corr()
	fig = plt.figure(figsize = (15,15))

	sb.heatmap(C_mat, vmax = .8, square = True)
	plt.title('The heatmap showing the correlation between the features')
	plt.show()
	#print train.head()

if __name__ =="__main__":
	main()
