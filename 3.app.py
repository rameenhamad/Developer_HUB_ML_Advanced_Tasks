import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

model = TFBertForSequenceClassification.from_pretrained("news_classifier_tf")
tokenizer = BertTokenizer.from_pretrained("news_classifier_tf")

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("News Headline Classifier (BERT - TensorFlow)")
st.write("Enter a news headline to predict its category:")

headline = st.text_area("Headline")

if st.button("Classify"):
    if headline.strip():
        inputs = tokenizer(headline, return_tensors="tf", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        prediction = int(tf.argmax(outputs.logits, axis=1).numpy()[0])
        st.success(f"**Predicted Category:** {labels[prediction]}")
    else:
        st.warning("Please enter a headline first!")
