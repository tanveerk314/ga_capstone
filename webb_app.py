# -*- coding: utf-8 -*-

#https://www.mihaileric.com/posts/state-of-the-art-question-answering-streamlit-huggingface/


#Prereq: make model into pipeline


#https://towardsdatascience.com/nlp-data-apps-batteries-included-with-streamlit-and-huggingface-828083a89bb2


model_checkpoint = 'run3-checkpoint-40790'

import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import pipeline
st.cache(show_spinner=False)
def load_pipe():
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    qa_pipe = pipeline('question-answering',model=model,tokenizer=tokenizer)
    return qa_pipe
qa_pipe = load_pipe()
st.header("QA Model Demo")
st.text("This demo uses DistelBert Model Trained on Squad V2")
#add_text_sidebar = st.sidebar.title("Menu")
#add_text_sidebar = st.sidebar.text("Just some random text.")
question = st.text_input(label='Enter a question.')
text = st.text_area(label="Enter the context for the question")
if (not len(text)==0) and not (len(question)==0):
    x_dict = qa_pipe(context=text,question=question)
    st.text(f"Answer: ,{x_dict['answer']}")
