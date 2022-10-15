from datetime import date, datetime
import numpy as np
from fastapi import FastAPI, Form, responses
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

app = FastAPI()

@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
   return '''<form method="post"> 
    <input type="number" name="fromDate"/>  
    <input type="submit"/> 
    </form>'''   

@app.get("/graph")
def image():
    return responses.FileResponse("graph.jpg")
        
data = pd.read_csv('us-counties-2020.csv')
# tokenizer = Tokenizer(num_words=2000, split=' ')
# tokenizer.fit_on_texts(data['text'].values)
pd.to_datetime(data['date'], format = '%Y-%m-%d')

def preProcess_data(date):
    new_date = pd.to_datetime(date, format = '%Y-%m-%d')
    return new_date

def my_pipeline(start_date):
    startDate_new = preProcess_data(start_date)
    #endDate_new = preProcess_data(end_date)
    # X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    # X = pad_sequences(X, maxlen=28)
    return startDate_new

@app.post('/predict')
def predict(fromDate:int = Form(...)):
    #clean_fDate,clean_eDate = my_pipeline(fromDate) #clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('Forecast.h5') #load the saved model 
    predictions = loaded_model.forecast(fromDate) #predict the text
    #probability = max(predictions.tolist()[0]) #calulate the probability
    return { #return predicted covid cases
         "PREDICTED Covid cases for next ": fromDate,
         "days are": predictions
    }

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO '/predict' for forescast page or '/graph' for visualizing actual and predicted values graph"}
