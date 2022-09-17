import transformers
import streamlit as st

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import pipeline

sentiment_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-imdb-sentiment")
    
def load_text_gen_model():
    generator = pipeline("text-generation", model="gpt2-medium")
    return generator 
    
@st.cache
def get_sentiment_model():
    sentiment_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-imdb-sentiment")
    return sentiment_model 

def get_summarizer_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

      
def get_sentiment(text):
    input_ids = sentiment_tokenizer .encode(text + '</s>', return_tensors='pt')
    output = sentiment_extractor.generate(input_ids=input_ids,max_length=2)
    dec = [sentiment_tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label
    

def get_qa_model():
    model_name = "deepset/roberta-base-squad2"

    qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return qa_pipeline
    
review = st.text_area("")

if review:
    sentiment = get_sentiment(review)
    st.write(sentiment)
