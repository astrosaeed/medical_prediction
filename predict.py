import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split 
from keras.models import load_model

mymodel=load_model('Weights-994--0.02692.hdf5')