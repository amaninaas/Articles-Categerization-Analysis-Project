from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import matplotlib.pyplot as plt



class ModelDevelopment:
    def simple_dl_model(self,input_shape,nb_class,vocab_size,out_dim,
                        nb_node=128,dropout_rate=0.3):
        '''
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        nb_class : TYPE
            DESCRIPTION.
        vocab_size : TYPE
            DESCRIPTION.
        out_dims : TYPE
            DESCRIPTION.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 128.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Embedding(vocab_size,out_dim))
        model.add(Bidirectional(LSTM(nb_node,return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(nb_node,return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(nb_node)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()

        return model

class ModelEvaluation:
    def plot_loss_grapy(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['Training loss','Validation loss'])
        plt.show()
        hist.history['loss']
    def plot_acc_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.xlabel('epoch')
        plt.legend(['Training Acc','Validation Acc'])
        plt.show()
        hist.history['loss']