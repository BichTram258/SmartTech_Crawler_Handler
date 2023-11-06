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
import time
from flask import Flask, request, Response
import asyncio
from bson.objectid import ObjectId
app = Flask(__name__)
client = pymongo.MongoClient("mongodb+srv://thiendihill181:A0YZHAJ9L4kxZfhb@cluster0.ys2zvmm.mongodb.net/")
db = client["test"]
collection = db["phones"]
start_time = time.time()

def convert_date(date_str):
    iso_format_1 = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$')
    iso_format_2 = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{2}$')
    iso_format_3 = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1}$')
    iso_format_4 = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')
    alt_format = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
    if not isinstance(date_str, str):
        return None
    if iso_format_1.match(date_str) or iso_format_2.match(date_str) or iso_format_3.match(date_str):
        date_format = '%Y-%m-%dT%H:%M:%S.%f'
        date_object = datetime.strptime(date_str, date_format)
        return date_object.month   
    elif iso_format_4.match(date_str):
        date_format = '%Y-%m-%dT%H:%M:%S'
        date_object = datetime.strptime(date_str, date_format)
        return date_object.month
    elif alt_format.match(date_str):
        date_format = '%Y-%m-%d %H:%M:%S'
        date_object = datetime.strptime(date_str, date_format)
        return date_object.month
    elif 'giờ trước' in date_str:
        hours = int(date_str.split()[0])
        date = datetime.now() - timedelta(hours=hours)
        return date
    elif 'ngày trước' in date_str:
        days = int(date_str.split()[0])
        date = datetime.now() - timedelta(days=days)
        return date
    elif re.search(r'(\d+)\s*(tuần|ngày|tháng)', date_str):
        match = re.search(r'(\d+)\s*(tuần|ngày|tháng)', date_str)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            current_date = datetime.now()

            if unit == 'tuần':
                duration = timedelta(weeks=value)
            elif unit == 'ngày':
                duration = timedelta(days=value)
            elif unit == 'tháng':
                duration = timedelta(days=value * 30)

            date = current_date - duration
            return date
        else:
            return None
    # else: 
    #     date_object = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    #     date = date_object.month
    # if date:
    #     return date.strftime('%m')
    # else:
    #     return None



train = pd.read_csv(r'E:\TGDD-Spark-NLP\UIT_ViSFD\Train.csv')
dev = pd.read_csv(r'E:\TGDD-Spark-NLP\UIT_ViSFD\Dev.csv')
test = pd.read_csv(r'E:\TGDD-Spark-NLP\UIT_ViSFD\Test.csv')

aa=['SCREEN','CAMERA','FEATURES','BATTERY','PERFORMANCE','STORAGE','DESIGN','PRICE','GENERAL','SER&ACC']
a =['SCREEN','CAMERA','FEATURES','BATTERY','PERFORMANCE','STORAGE','DESIGN','PRICE','GENERAL','SER&ACC','OTHERS']
s=['Positive','Negative','Neutral']
label=[]
aspect=[]
for i in range(0,10):
    x=[]
    for j in range(0,3):
        x.append(aa[i]+'#'+s[j])
    aspect.append(x)
for i in range (0,10):
    for j in range(0,3):
        label.append(aa[i]+'#'+s[j])
slabel=label.copy()
saspect=aspect.copy()
label.append('OTHERS')
aspect.append(['OTHERS'])

def ag_matrix(df_name,r1,r2,c1=5,c2=37,l=label):
    df=df_name.copy()
    for i in range(0,len(l)):
        df[l[i]]=0
    for i in range(r1,r2):
        for j in range(0,len(l)):
            if l[j] in str(df['label'][i]):
                df[l[j]][i]=1
    m=df.iloc[r1:r2,c1:c2]
    return m

def aspect_matrix(df_name,r1,r2):
    if r1==0:
      kk=pd.DataFrame(np.zeros((11,r2-r1))).T
      kk.columns=a
    else:
      kk=pd.DataFrame(np.zeros((11,r2-r1+1))).T
      kk.columns=a
    w=0
    for i in range(r1,r2):
        kk['SCREEN'][w]=max(df_name['SCREEN#Positive'][i], df_name['SCREEN#Negative'][i], df_name['SCREEN#Neutral'][i])
        kk['CAMERA'][w]=max(df_name['CAMERA#Positive'][i], df_name['CAMERA#Negative'][i], df_name['CAMERA#Neutral'][i])
        kk['FEATURES'][w]=max(df_name['FEATURES#Positive'][i], df_name['FEATURES#Negative'][i], df_name['FEATURES#Neutral'][i])
        kk['BATTERY'][w]=max(df_name['BATTERY#Positive'][i], df_name['BATTERY#Negative'][i], df_name['BATTERY#Neutral'][i])
        kk['PERFORMANCE'][w]=max(df_name['PERFORMANCE#Positive'][i], df_name['PERFORMANCE#Negative'][i], df_name['PERFORMANCE#Neutral'][i])
        kk['STORAGE'][w]=max(df_name['STORAGE#Positive'][i], df_name['STORAGE#Negative'][i], df_name['STORAGE#Neutral'][i])
        kk['DESIGN'][w]=max(df_name['DESIGN#Positive'][i], df_name['DESIGN#Negative'][i], df_name['DESIGN#Neutral'][i])
        kk['PRICE'][w]=max(df_name['PRICE#Positive'][i], df_name['PRICE#Negative'][i], df_name['PRICE#Neutral'][i])
        kk['GENERAL'][w]=max(df_name['GENERAL#Positive'][i], df_name['GENERAL#Negative'][i], df_name['GENERAL#Neutral'][i])
        kk['SER&ACC'][w]=max(df_name['SER&ACC#Positive'][i], df_name['SER&ACC#Negative'][i], df_name['SER&ACC#Neutral'][i])
        kk['OTHERS'][w]=df_name['OTHERS'][i]
        w+=1
    return kk


dd_train=ag_matrix(train,0,len(train))
ddd_train=aspect_matrix(dd_train,0,len(train))
dd_test=ag_matrix(test,0,len(test))
ddd_test=aspect_matrix(dd_test,0,len(test))

def text_process(train):
    processed_text = re.sub(r'\W', ' ', train)  # Ví dụ: loại bỏ ký tự không phải chữ cái và số
    processed_text = processed_text.lower()# Ví dụ: chuyển thành chữ thường
    a = process.normalize_acronyms(processed_text)
    b = process.tokenize(a)
    c = process.remove_stop_words(b)
    d = ' '.join(c)
    return d

X_train = train['comment'].apply(text_process)
X_test = test['comment'].apply(text_process)

EMBEDDING_FILE= '../process/cc.vi.300.vec'
max_features=2489
maxlen=150
embed_size=300


Y_train = dd_train
Y_test = dd_test

tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(X_train))
word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
print("create vector")

embeddings_index = {}
with open('../process/cc.vi.300.vec', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
embedding_dim = 300
embedding_matrix = np.zeros((max_features, embedding_dim))

for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector[:embedding_dim]

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#training params
batch_size = 32
epochs =20

#model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4

inp = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size)(inp)
x = SpatialDropout1D(0.35)(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = Concatenate()([avg_pool, max_pool])
x = Dense(31, activation="sigmoid")(conc)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])
model.summary()

batch_size = 32
epochs =20

model.load_weights(r'E:\TGDD-Spark-NLP\Crawl-data\BiGRU_68.h5')

predictions = model.predict(X_test, batch_size=batch_size, verbose=1)


print(predictions)
pre=pd.DataFrame(predictions)
pre.columns=label

score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
print('Test accuracy bigru:', score)

def process_overview(record):
    fpt = record["overview_fpt"]
    data_comment = record["data"]
    rateStar = ["rateStar"]
    comment = ["comment"]
    df_cmt = pd.DataFrame(data_comment, columns=comment)
    df_fpt = pd.DataFrame(data_comment, columns=rateStar)
    df_fpt = df_fpt.dropna(subset=['rateStar'])
    total_fpt_tiki = ((df_fpt["rateStar"].sum())/(len(df_fpt["rateStar"]))).round(1)
    filtered_rows = df_fpt[df_fpt['rateStar'] == 5]
    count_5_star = len(filtered_rows)
    
    if "overview_tgdd" in record:
        tgdd = record["overview_tgdd"]
        total_rating = round(((total_fpt_tiki + float(tgdd["total_rating"])) / 2), 1)
        total_cmt = len(df_cmt["comment"]) 
        five_rating = count_5_star + round((float(re.search(r'\d+', tgdd["five_rating"]).group())*float(tgdd["total_cmt"])/100), 0)
    else:
        total_rating = total_fpt_tiki
        total_cmt = len(df_cmt["comment"])
        five_rating = float(fpt["five_rating"])
    data_overview = {
        "total_rating": str(total_rating) + "/5",
        "total_cmt": total_cmt,
        "five_rating": five_rating
    }
    print(data_overview)
    collection.update_one({"_id": ObjectId("6522447b2fd11e736b620e2c")}, 
                        {"$set": {
                            "overview": data_overview
                        }})
    date = ["date"]
    df = pd.DataFrame(data_comment, columns=date)
    df["count_date"] = df["date"].apply(convert_date)
    df["count_date"] = df["count_date"].apply(lambda x: str(x).zfill(2))
    print(df["count_date"].head(20))
    months = [str(month).zfill(2) for month in range(1, 13)]
    count_by_month = df["count_date"].value_counts()
    count_by_month = count_by_month.reindex(months, fill_value=0)
    print(count_by_month)
    collection.update_one({"_id": ObjectId("6522447b2fd11e736b620e2c")}, 
                        {"$set": {
                            "count_month": count_by_month.to_dict()
                        }})
def main(record):
    name = record["name"]
    data_comment = record["data"]
    text = ["comment"]
    df = pd.DataFrame(data_comment, columns=text)
    df = df.dropna(subset=['comment'])
    X_test = df['comment'].apply(text_process)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    predictions_x = model.predict(X_test, batch_size=batch_size, verbose=1)
    print(predictions_x)
    result = pd.DataFrame(predictions_x, columns=label)

    count_pos = result.filter(regex='Positive').gt(0.5).sum().sum()
    count_neg = result.filter(regex='Negative').gt(0.5).sum().sum()
    count_neu = result.filter(regex='Neutral').gt(0.5).sum().sum()
    total_senti = count_pos + count_neg + count_neu
    pos = (count_pos*100/total_senti).round(2)
    neg = (count_neg*100/total_senti).round(2)
    neu = (count_neu*100/total_senti).round(2)
    result_senti = pd.DataFrame({"positive": [pos], "negative": [neg], "neutral": [neu]})
    print(result_senti)

    def count_condition(value):
        if value > 0.5:
            return 1
        else:
            return 0
    count_result = result.apply(lambda x: x.apply(count_condition))
    score = model.evaluate(X_test, count_result, batch_size=batch_size, verbose=1)
    print('Test accuracy test:', score)
    positive_data = []
    negative_data = []
    for index, row in count_result.iterrows():
        for column, value in row.items():
            if value == 1:
                if '#' in column:
                    emotion = column.split('#')[1]
                else:
                    emotion = 'Unknown'
                    
                comment = df.iloc[index, df.columns.get_loc('comment')]
                
                if emotion == 'Positive' and comment not in positive_data:
                    positive_data.append(comment)
                elif emotion == 'Negative' and comment not in negative_data:
                    negative_data.append(comment)

    print(count_result)
    column_names = {
        col: col.split("#")[0] + "_" + col.split("#")[1].lower() if col.find("#") != -1 else col
        for col in count_result.columns
    }
    df = count_result.rename(columns=column_names)
    print(df)
    column_sum = df.sum()
    column_sum = column_sum.to_dict()
    print(column_sum)

    data_senti = {
        "positive": float(result_senti["positive"].iloc[0]),
        "negative": float(result_senti["negative"].iloc[0]),
        "neutral": float(result_senti["neutral"].iloc[0])
    }

    collection.update_one({"_id": ObjectId("6522447b2fd11e736b620e2c")}, 
                        {"$set": 
                            {"status": "4",
                            "percent_senti": data_senti, 
                            "aspect_senti": column_sum, 
                            "positive_list":  [{"comment": comment} for comment in positive_data],
                            "negative_list":  [{"comment": comment} for comment in negative_data]
    }})
def main_work():
    start_time_2 = time.time()
    while True:
        query = {
                "_id": ObjectId("6522447b2fd11e736b620e2c")
                # {"name": "iPhone 13 128GB"},
                # {"status_tgdd": "2"},
                # {"status_fpt": "2"},
                # {"status": "1"}
            }
        record = collection.find_one(query)
        if record:
            process_overview(record)
            main(record)  
            end_time = time.time()

            # Tính thời gian thực thi
            execution_time1 = end_time - start_time_2

            # In thời gian thực thi
            print(f"Thời gian thực thi: {execution_time1} giây") 
            
            break
        else:
            continue

# def start_task(dataId):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(main_work(dataId))
# @app.route('/handle_request/<string:dataId>', methods=['GET'])  
# def handle_request(dataId):
#     if dataId:
#         print(dataId)
#         time.sleep(1)
#         start_task(dataId)
#         return Response('Request handled successfully', status=200)

if __name__ == '__main__':
    main_work()
    
    