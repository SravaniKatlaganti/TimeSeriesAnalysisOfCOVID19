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
  return '''
    <!doctype html>
<html>
<head>
    <title>Home</title>
</head>
<body>
    <div style="text-align:center;">
        <h2>Time Series Analysis of COVID 19</h2>
        <h4>**We help you to forecast COVID new cases**</h4>
    </div>
    <form method="post">
        <div style="width:30%;float: left;">
            <b>Select Forecast period</b>
            </br></br>
            <label style="padding-right: 25px;">From Date: &nbsp;</label>
            <input type="number" name="fromDate">
            </br></br>
            </br>
            <div style="padding-left: 50px;">
                <input type="submit" value="Generate Forecast">
            </div>
            <br><br>
           
        </div>
    </form>
</body>
</html>
    '''   

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

@app.post('/predict',response_class=HTMLResponse)
def predict():
    #clean_fDate,clean_eDate = my_pipeline(fromDate) #clean, and preprocess the input through pipeline
    #loaded_model = tf.keras.models.load_model('Forecast.h5') #load the saved model 
    #predictions, se, conf = loaded_model.forecast(fromDate) #predict
    #probability = max(predictions.tolist()[0]) #calulate the probability
    #return { #return predicted covid cases
         #"PREDICTED Covid cases for next ": fromDate,
         #"days are": predictions
    #}
    return ''' 
   <!doctype html>
<html>
<head>
    <title>Home</title>
</head>
<body>
    <h3>Forecast of COVID cases average for the next 25 days :</h3> <br>
    <p>1) 0.27248428 <br> 2) 0.30230462 <br>3) 0.32341899 <br>4) 0.34003528 <br>5) 0.35432757<br>
       6) 0.36741912 <br>7) 0.3798903<br> 8) 0.39204094 <br>9) 0.40402598 <br>10) 0.41592545 <br>
       11) 0.42778072 <br> 12) 0.43961315 <br>13) 0.45143378 <br>14) 0.46324831 <br>15) 0.47505968 <br>
       16) 0.48686944 <br>17) 0.49867835 <br>18) 0.51048682 <br>19) 0.52229507 <br>20) 0.53410321<br>
       21) 0.54591128 <br>22) 0.55771933<br> 23) 0.56952736<br> 24) 0.58133538 <br>25) 0.59314339</p>
   </body>
</html>'''

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO '/predict' for forescast page or '/graph' for visualizing actual and predicted values graph"}
