import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.layers import GRU, Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding, Concatenate
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from sklearn.pipeline import Pipeline
from keras_preprocessing.sequence import pad_sequences
import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os, re, csv, math, codecs
from tensorflow.data import Dataset
import process_data as process
import pymongo
import asyncio
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import threading
client = pymongo.MongoClient("mongodb+srv://thiendihill181:A0YZHAJ9L4kxZfhb@cluster0.ys2zvmm.mongodb.net/")
db = client["test"]
collection = db["phones"]


def convert_date(date_str):
    if 'giờ trước' in date_str:
        hours = int(date_str.split()[0])
        date = datetime.now() - timedelta(hours=hours)
    elif 'ngày trước' in date_str:
        days = int(date_str.split()[0])
        date = datetime.now() - timedelta(days=days)
    else:
        try:
            date = datetime.strptime(date_str, 'Ngày %d/%m/%Y')
        except ValueError:
            return None
    formatted_date = date.strftime('%m')
    return formatted_date

def process_overview():
    query = {"$and": [
                {"_id": {"$exists": True}},
                {"name": {"$exists": True}}
            ]}
    record = collection.find_one(query)
    
    fpt = record["overview_fpt"]
    if "overview_tgdd" in record:
        tgdd = record["overview_tgdd"]
        total_rating = round(((float(fpt["total_rating"].split('/')[0]) + float(tgdd["total_rating"])) / 2), 1)
        total_cmt = float(re.search(r'\d+', fpt["total_cmt"]).group()) + float(tgdd["total_cmt"])
        five_rating = float(fpt["five_rating"]) + round((float(re.search(r'\d+', tgdd["five_rating"]).group())*float(tgdd["total_cmt"])/100), 0)
    else:
        total_rating = fpt["total_rating"].split('/')[0]
        total_cmt = float(re.search(r'\d+', fpt["total_cmt"]).group())
        five_rating = float(fpt["five_rating"])
    data_overview = {
        "total_rating": str(total_rating) + "/5",
        "total_cmt": total_cmt,
        "five_rating": five_rating
    }
    collection.update_one({"_id": {"$exists": True}}, 
                        {"$set": {
                            "overview": data_overview
                        }})
    data_comment = record["data"]
    date = ["date"]
    df = pd.DataFrame(data_comment, columns=date)
    df["count_date"] = df["date"].apply(convert_date)
    months = [str(month).zfill(2) for month in range(1, 13)]
    count_by_month = df["count_date"].value_counts()
    count_by_month = count_by_month.reindex(months, fill_value=0)
    print(count_by_month)
    collection.update_one({"_id": {"$exists": True}}, 
                        {"$set": {
                            "status": "4",
                            "count_month": count_by_month.to_dict()
                        }})


  

# train = pd.read_csv('../UIT_ViSFD/Train.csv')
# dev = pd.read_csv('../UIT_ViSFD/Dev.csv')
# test = pd.read_csv('../UIT_ViSFD/Test.csv')

# aa=['SCREEN','CAMERA','FEATURES','BATTERY','PERFORMANCE','STORAGE','DESIGN','PRICE','GENERAL','SER&ACC']
# a =['SCREEN','CAMERA','FEATURES','BATTERY','PERFORMANCE','STORAGE','DESIGN','PRICE','GENERAL','SER&ACC','OTHERS']
# s=['Positive','Negative','Neutral']
# label=[]
# aspect=[]
# for i in range(0,10):
#     x=[]
#     for j in range(0,3):
#         x.append(aa[i]+'#'+s[j])
#     aspect.append(x)
# for i in range (0,10):
#     for j in range(0,3):
#         label.append(aa[i]+'#'+s[j])
# slabel=label.copy()
# saspect=aspect.copy()
# label.append('OTHERS')
# aspect.append(['OTHERS'])

# def ag_matrix(df_name,r1,r2,c1=5,c2=37,l=label):
#     df=df_name.copy()
#     for i in range(0,len(l)):
#         df[l[i]]=0
#     for i in range(r1,r2):
#         for j in range(0,len(l)):
#             if l[j] in str(df['label'][i]):
#                 df[l[j]][i]=1
#     m=df.iloc[r1:r2,c1:c2]
#     return m


# dd_train=ag_matrix(train,0,len(train))
# dd_test=ag_matrix(test,0,len(test))

# def text_process(train):
#     processed_text = re.sub(r'\W', ' ', train)  # Ví dụ: loại bỏ ký tự không phải chữ cái và số
#     processed_text = processed_text.lower()# Ví dụ: chuyển thành chữ thường
#     a = process.normalize_acronyms(processed_text)
#     b = process.tokenize(a)
#     c = process.remove_stop_words(b)
#     d = ' '.join(c)
#     return d

# X_train = train['comment'].apply(text_process)
# X_test = test['comment'].apply(text_process)

# EMBEDDING_FILE= '../process/cc.vi.300.vec'
# max_features=2489
# maxlen=150
# embed_size=300


# Y_train = dd_train
# Y_test = dd_test

# tokenizer = text.Tokenizer(num_words=max_features, lower=True)
# tokenizer.fit_on_texts(list(X_train))
# word_index = tokenizer.word_index

# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)

# X_train = pad_sequences(X_train, maxlen=maxlen)
# X_test = pad_sequences(X_test, maxlen=maxlen)
# print("create vector")

# embeddings_index = {}
# with open('../process/cc.vi.300.vec', encoding='utf-8') as f:
#     for line in f:
#         values = line.strip().split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
        
# embedding_dim = 300
# embedding_matrix = np.zeros((max_features, embedding_dim))

# for word, i in word_index.items():
#     if i >= max_features:
#         continue

#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector[:embedding_dim]

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# #training params
# batch_size = 32
# epochs =20

# #model parameters
# num_filters = 64
# embed_dim = 300
# weight_decay = 1e-4

# inp = Input(shape=(maxlen, ))
# x = Embedding(max_features, embed_size)(inp)
# x = SpatialDropout1D(0.35)(x)
# x = Bidirectional(GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
# x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
# avg_pool = GlobalAveragePooling1D()(x)
# max_pool = GlobalMaxPooling1D()(x)
# conc = Concatenate()([avg_pool, max_pool])
# x = Dense(31, activation="sigmoid")(conc)
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
# model.summary()

# batch_size = 32
# epochs =20

# model.load_weights(r'E:\TGDD-Spark-NLP\Crawl-data\BiGRU_68.h5')

# predictions = model.predict(X_test, batch_size=batch_size, verbose=1)


# print(predictions)
# pre=pd.DataFrame(predictions)
# pre.columns=label

# score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
# print('Test accuracy lstm:', score)


# def main():
#     query = {"$and": [
#                 {"_id": {"$exists": True}},
#                 {"name": {"$exists": True}}
#             ]}
#     record = collection.find_one(query)
#     data_comment = record["data"]
#     text = ["comment"]
#     df = pd.DataFrame(data_comment, columns=text)
#     X_test = df['comment'].apply(text_process)
#     X_test = tokenizer.texts_to_sequences(X_test)
#     X_test = pad_sequences(X_test, maxlen=maxlen)

#     predictions_x = model.predict(X_test, batch_size=batch_size, verbose=1)
#     result = pd.DataFrame(predictions_x, columns=label)

#     count_pos = result.filter(regex='Positive').gt(0.5).sum().sum()
#     count_neg = result.filter(regex='Negative').gt(0.5).sum().sum()
#     count_neu = result.filter(regex='Neutral').gt(0.5).sum().sum()
#     total_senti = count_pos + count_neg + count_neu
#     pos = (count_pos*100/total_senti).round(2)
#     neg = (count_neg*100/total_senti).round(2)
#     neu = (count_neu*100/total_senti).round(2)
#     result_senti = pd.DataFrame({"positive": [pos], "negative": [neg], "neutral": [neu]})
#     print(result_senti)

#     def count_condition(value):
#         if value > 0.5:
#             return 1
#         else:
#             return 0
#     count_result = result.apply(lambda x: x.apply(count_condition))
#     positive_data = []
#     negative_data = []
#     for index, row in count_result.iterrows():
#         for column, value in row.iteritems():
#             if value == 1:
#                 if '#' in column:
#                     emotion = column.split('#')[1]
#                 else:
#                     emotion = 'Unknown'
                    
#                 comment = df.iloc[index, df.columns.get_loc('comment')]
                
#                 if emotion == 'Positive' and comment not in positive_data:
#                     positive_data.append(comment)
#                 elif emotion == 'Negative' and comment not in negative_data:
#                     negative_data.append(comment)

#     print(count_result)
#     column_names = {
#         col: col.split("#")[0] + "_" + col.split("#")[1].lower() if col.find("#") != -1 else col
#         for col in count_result.columns
#     }
#     df = count_result.rename(columns=column_names)
#     print(df)
#     column_sum = df.sum()
#     column_sum = column_sum.to_dict()
#     print(column_sum)

#     data_senti = {
#         "positive": float(result_senti["positive"].iloc[0]),
#         "negative": float(result_senti["negative"].iloc[0]),
#         "neutral": float(result_senti["neutral"].iloc[0])
#     }

#     collection.update_one({"_id": {"$exists": True}}, 
#                         {"$set": 
#                             {"status": "4",
#                             "percent_senti": data_senti, 
#                             "aspect_senti": column_sum, 
#                             "positive_list":  [{"comment": comment} for comment in positive_data],
#                             "negative_list":  [{"comment": comment} for comment in negative_data]
#     }})
        

if __name__ == '__main__':
    process_overview()
   
    # main()