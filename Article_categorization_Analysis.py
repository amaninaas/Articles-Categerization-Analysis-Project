# -*- coding: utf-8 -*-
"""Neural_Network_predictions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z__A9ZQC0sYyidOEGsk45g4RWleF4w7i
"""

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive"

!ls "/content/drive/MyDrive/PU_NLP"

# Module
import os
import json 
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Input,Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

#%% Constants
LOGS_PATH = os.path.join(os.getcwd(),'logs',
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
BEST_MODEL_PATH = os.path.join(os.getcwd(),'Saved_models','model_mcd.h5')
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'Saved_models','tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'Saved_models,'ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_models','model.h5')

#%% Step 1) Data Loading
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(CSV_URL)

#%% Step 2) Data Inspection
df.head()
df.tail()

# can check duplicated in NLP
# There is 99 duplicated text
df.duplicated().sum()

#%% Step 3) Data Cleaning 

# Remove the duplicated data
df = df.drop_duplicates()

# Assign variable to the dataset columns
article = df['text'].values  # features of X
category = df['category'].values # target, y

# To backup the dataset
article_backup = article.copy()
category_backup = category.copy()

#%% Step 4) Features Selection
#-no features to select

#%% Step 5) Data Preprocessing (Tokenizer to change the text to number)
# 1) Convert into lower case (no upper case been detected in text)

# 2) Tokenization
# must not contain empty list
# need to convert the text to numbers
vocab_size = 10000 
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(article) # Learning all the words

word_index = tokenizer.word_index

# To show 10 to 20 only put the slice after the list
print(dict(list(word_index.items())[10:20]))

# to convert into numbers
article_int = tokenizer.texts_to_sequences(article)

# to check length of every sentence in review
for i in range(len(article_int)):
  print(len(article_int[i]))
  
# 3)Padding & Trunctation
# to decide the length of the padding, use <median> to pick the padding number
length_article = []
for i in range(len(article_int)):
  length_article.append(len(article_int[i]))
  #print(len(article_int[i]))

np.median(length_article)

# comprehension
max_len = np.median([len(article_int[i])for i in range(len(article_int))])
max_len # need to convert to integer

padded_article = pad_sequences(article_int,
                              maxlen=int(max_len),
                              padding='post',
                              truncating='post')

# 4)OneHotEncoding for the Target(y)
# Y target
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

# 5)Train test split
X_train,X_test,y_train,y_test = train_test_split(padded_article, 
                                                 category,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Model Development
input_shape = np.shape(X_train)[1:]
out_dim = 128

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,out_dim)) 
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['acc'])
			  
# Visualization the model
plot_model(model,show_shapes=True,show_layer_names=True)

#%% Model Training
# Tensorboard Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

# ModelCheckpoint
mdc = ModelCheckpoint(BEST_MODEL_PATH,monitor='val_acc',
                      save_best_only=True,
                      modes='max',verbose=1)
# EarlyStopping
early_callback = EarlyStopping(monitor='val_loss',patience=3)

hist =model.fit(X_train,y_train,
                epochs=5,
                validation_data=(X_test,y_test),
                callbacks=[mdc,tensorboard_callback,early_callback])

print(hist.history.keys())

#%% Plot Graph
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['Training acc','Validation acc'])
plt.show()

print(model.evaluate(X_test,y_test))


#%%Model Evaluation
y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test,axis=1)
cm = confusion_matrix(y_actual,y_pred)
cr = classification_report(y_actual,y_pred)

print(cm)
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Model Saving
#TOKENIZER
token_json = tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH,'w') as file:
  json.dump(token_json,file)

# OHE
with open(OHE_SAVE_PATH,'wb') as file:
  pickle.dump(ohe,file)

# MODEL
model.save(MODEL_SAVE_PATH)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs


