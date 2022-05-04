import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFBertModel
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import classification_report
from google.colab import drive
'''
save_model takes in a model and saves it to 
a monted google drive to save training progress
'''
def save_model(model):
    drive.mount('/content/gdrive')
    model.save('/content/drive/MyDrive/model/csc439model')
'''
build data cut dtat into labels and text
retrns array of 2 elements
'''
def build_data(file):
    fp = open(file,errors='ignore')
    data_text = []
    data_labels = []
    for i in fp:
        o = i.split(" ")[0]
        data_labels.append(o)
        data_text.append(" ".join(i.split()[1:]))
    return [data_labels,data_text]
'''
takes in a label text array and adds it to a np array
'''
def to_np_array(label,text,fp):
    np_array = np.array([" "," "])
    for i in fp:
        label = i.split(" ")[0]
        text = " ".join(i.split()[1:])
        np_array.append([label,text])
'''
takes in label and text array and builds
a pandas data frame
'''
def label_to_data_frame(labels,text):
    df = pd.DataFrame()

    df["label"] = labels
    df["text"] = text
    return df
'''
builds model sets up dense layers as well as both input layers
as well as seting max sencence size and mask size
'''
def build_model():
    token = AutoTokenizer.from_pretrained('bert-base-cased')
    bert = TFBertModel.from_pretrained('bert-base-cased')#this is slow
    max_len = 150
    
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")


    embeddings = bert(input_ids,attention_mask = mask)[0] 
    X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    #X = tf.keras.layers.BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    #X = Dense(32,activation = 'relu')(X)
    y = Dense(3,activation = 'sigmoid')(X)

    model = tf.keras.Model(inputs=[input_ids,mask], outputs=y)
    model.layers[2].trainable = True

    #optimizer = tf.keras.optimizers.Adam(0.01)
    optimizer = Adam(learning_rate=5e-05,epsilon=1e-08,decay=0.01,clipnorm=1.0)
    #from_logits =True
    loss =CategoricalCrossentropy(from_logits =True)
    metric = CategoricalAccuracy('balanced_accuracy')
    model.compile(optimizer = optimizer,loss = loss,metrics = metric)
    return model
'''
reads in data builds a data frame and ses model to make prediction
'''
def test_data_set(file,model,token):
    fp3 = open(file,errors='ignore')
    test_text = []
    test_labels = []
    for p in fp3:
        o = p.split(" ")[0]
        test_labels.append(o)
        test_text.append(" ".join(p.split()[1:]))

    test_dev = pd.DataFrame()
    test_dev["label"] = test_labels
    test_dev["text"] = test_text

    labels_to_int_test = {'good':0,'ok':1,'bad':2}
    test_dev['label'] = test_dev.label.map(labels_to_int_test)
    x_test = token(text = test_dev.text.tolist(),add_special_tokens=True,max_length=150,truncation=True,padding=True, return_tensors='tf',return_token_type_ids = False,return_attention_mask = True,verbose = True)

    predicted_test = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})

    y_predict_test = np.argmax(predicted_test,axis = 1)
    y_pass_test = test_dev.label
    print(classification_report(y_pass_test,y_predict_test))
'''
takes in a model and as well as train conditions and trains the model
this function is very slow
'''
def train_model(model,epochs,step_size,token,df_train,df_test):
    x_train = token(text = df_train.text.tolist(),add_special_tokens=True,max_length=150,truncation=True,padding=True, return_tensors='tf',return_token_type_ids = False,return_attention_mask = True,verbose = True)
    x_test = token(text = df_test.text.tolist(),add_special_tokens=True,max_length=150,truncation=True,padding=True, return_tensors='tf',return_token_type_ids = False,return_attention_mask = True,verbose = True)
    labels_to_int = {'good':0,'ok':1,'bad':2}
    df_train['label'] = df_train.label.map(labels_to_int)
    df_test['label'] = df_test.label.map(labels_to_int)
    y_train = to_categorical(df_train.label)
    y_test = to_categorical(df_test.label)
    history = model.fit(x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,y = y_train,validation_data = ({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test),epochs=epochs,batch_size=step_size)
    return history
def main():
    token = AutoTokenizer.from_pretrained('bert-base-cased')
    #bert = TFBertModel.from_pretrained('bert-base-cased')
    file_train = "movie_train.txt"
    file_test = "movie_test.txt"
    file_dev = "movie_dev.txt"   
    data_train = build_data(file_train)
    data_test = build_data(file_test)
    data_dev = build_data(file_dev)
    df_train = label_to_data_frame(data_train[0],data_train[1])
    df_test = label_to_data_frame(data_test[0],data_test[1])
    df_dev = label_to_data_frame(data_dev[0],data_dev[1])
    model = build_model()
    train_model(model,2,36,token,df_train,df_test)
    test_data_set(file_test,model,token)
    test_data_set(file_dev,model,token)

if __name__=="__main__":
    main()
'''
df_tarin = df.iloc[:cut,:].copy()
df_test = df.iloc[cut:,:].copy()

labels_to_int = {'good':0,'ok':1,'bad':2}
df_tarin['label'] = df_tarin.label.map(labels_to_int)
df_test['label'] = df_test.label.map(labels_to_int)
df_tarin.head(20)

y_train = to_categorical(df_tarin.label)
y_test = to_categorical(df_test.label)
'''
    