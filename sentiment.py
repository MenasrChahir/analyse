import numpy as np
import pandas as pd
import streamlit as st
import re
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import cleantext
from typing import List

st.header('Sentiment Analysis')


model = load_model('FINALsentiment_modelss.h5')

with open('FINALsentiment_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


with open('stopwords.txt', 'r') as f:
    stopwords = set(f.read().split())




def clean_text(text: str, algerian_arabic_stopwords: List[str]) -> str:
    
    text = str(text)
    
    text = re.sub(r'http\S+', '', text)
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text = text.lower()
    
    words = text.split()
    
    words = [w for w in words if not w in stopwords]
    
    text = " ".join(words)
    return text

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        cleaned_text = clean_text(text, stopwords)
        input_sequence = tokenizer.texts_to_sequences([cleaned_text])
        input_sequence = pad_sequences(input_sequence, maxlen=250)
        predictions = model.predict(input_sequence)
        label = np.argmax(predictions[0])
        if label == 0:
            sentiment = "Negative"
        else:
            sentiment = "Positive"
        st.write('Sentiment: ', sentiment)

    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = clean_text(pre, stopwords)
        st.write(cleaned_text)


    def analyze(x):
        input_sequence = tokenizer.texts_to_sequences([x])
        input_sequence = pad_sequences(input_sequence, maxlen=250)
        predictions = model.predict(input_sequence)
        label = np.argmax(predictions[0])
        if label == 0:
            return 'Negative'
        else:
            return 'Positive'

    
      
