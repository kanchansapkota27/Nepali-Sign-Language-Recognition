#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers import Dense,Dropout,BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import image as Img
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import constants
import cv2


base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)
inception_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)
class Predictor:
    def __init__(self,frames_list):
        self.raw_frames_list=frames_list
        self.model_path=constants.modelPath
        #Explicit Classes=["father", "food", "promise", "tea", "wife"]
        self.classes=constants.classes

    def get_model(self):
            model = Sequential()
            model.add(LSTM(256,return_sequences=True, input_shape=(40,2048), dropout=0.4))
            model.add(BatchNormalization())
            # model.add(Dropout(0.6))
            model.add(LSTM(128))
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(self.classes), activation='softmax'))
            model.load_weights(self.model_path)
            return model

    def get_rescaled_list(self,input_list,size):
        assert len(input_list) >= size,"Not Enough Frames"
        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]

    def get_inception_features(self,rescaled_list):
        feature_sequence=[]
        for image in rescaled_list:
            #img = Img.load_img(image, target_size=(299, 299))
            #x = Img.img_to_array(img)
            x=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            x=cv2.resize(image,(299,299),cv2.INTER_AREA)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = inception_model.predict(x)
            feature_sequence.append(features[0])
        return feature_sequence

    def get_prediction(self,model,feature_sequence):
        sequence=np.asarray(feature_sequence).astype(np.float32)
        sequence=np.expand_dims(sequence,axis=0)
        prediction = model.predict(sequence)
        maxm = prediction[0][0]
        maxid = 0
        for i in range(len(prediction[0])):
          if(maxm<prediction[0][i]):
            maxm = prediction[0][i]
            maxid = i
        prediction_label=self.classes[maxid]
        return prediction_label

    def predict(self):
        model=self.get_model()
        rescaled_list=self.get_rescaled_list(self.raw_frames_list,constants.nb_frames)
        feature_sequence=self.get_inception_features(rescaled_list)
        predicted_label=self.get_prediction(model,feature_sequence)
        return predicted_label

