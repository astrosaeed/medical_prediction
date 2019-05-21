from build import *

class TrainTwo(Train):

	def build_model(self,train_data, train_labels, test_data,test_labels,layer_nodes):


		model = Sequential()
		# The Input Layer :
		model.add(Dense(layer_nodes[0], kernel_initializer='normal',input_dim = train_data.shape[1], activation='relu'))

		# The Hidden Layers :
		model.add(Dense(layer_nodes[1], kernel_initializer='normal',activation='relu'))
	#	model.add(Dense(layer_nodes[2], kernel_initializer='normal',activation='relu'))
	#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))#
	#	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

		# The Output Layer :
		model.add(Dense(1, kernel_initializer='normal',activation='linear'))

		#Compile the network :
		model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
		model.summary()
		print layer_nodes[0]
		os.makedirs(str(layer_nodes[0])+'_'+str(layer_nodes[1]))
		checkpoint_name = str(layer_nodes[0])+'_'+str(layer_nodes[1])+'/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
		checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
		callbacks_list = [checkpoint]

		history=model.fit(train_data, train_labels, epochs=1000, batch_size=16, validation_data=(test_data,test_labels), callbacks=callbacks_list)

		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('mean absolute error')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(str(layer_nodes[0])+'_'+str(layer_nodes[1])+'/loss.png')
		plt.close()



def main():

	instant= TrainTwo()
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
	for nodes in [(8,16),(8,32), (12,16),(12,20)]:
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
