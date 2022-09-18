import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd


st.set_page_config(layout="wide")

@st.cache(allow_output_mutation = True)
def get_classifier_model():
    # return pipeline("zero-shot-classification",
    #          model="facebook/bart-large-mnli")
    return pipeline("zero-shot-classification",model="sentence-transformers/paraphrase-MiniLM-L6-v2")


#st.sidebar.image("Suncorp-Bank-logo.png",width=255)

st.image("Suncorp-Bank-logo.png",width=255)

st.title("Detecting Barriers from Conversations")
st.markdown("***")

text = st.text_area(label="Enter text to classify")
st.markdown("***")

col1, col2, col3 = st.columns((1,1,1))
col1.header("Select Sentiments")
sentiments = col1.multiselect("",["Happy","Sad","Anxious","Depressed","Empathetic"],["Happy","Sad","Anxious","Depressed","Empathetic"])
col2.header("Select Entities")
entities = col2.multiselect("",["Employee","Doctor","Family","Friends"],
                            ["Employee","Doctor","Family","Friends"])


col3.header("Select Reasons")

reasons = col3.multiselect("",["Bullying","Alchohol","Abuse","Domestic_Violence",'Chronic_Pain','Driving','Hobbies','Treatment'],
                            ["Bullying","Alchohol","Abuse","Domestic_Violence",'Chronic_Pain','Driving','Hobbies','Treatment'])

is_multi_class =  st.checkbox("Can have more than one classes",value=True)

st.markdown("***")

classify_button_clicked = st.button("Classify")

def get_classification(candidate_labels):
    classification_output = classifier(sequence_to_classify, candidate_labels, multi_class=is_multi_class)
    data = {'Class': classification_output['labels'], 'Scores': classification_output['scores']}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Scores', ascending=False)
    fig = px.bar(df, x='Scores', y='Class', orientation='h', width=800, height=800)
    fig.update_layout(
        yaxis=dict(
            autorange='reversed'
        )
    )
    return fig

if classify_button_clicked:
    if text:
        st.markdown("***")
        with st.spinner("  Please wait while the text is being classified.."):
            classifier = get_classifier_model()
            sequence_to_classify = text
            # candidate_labels = sentiments + entities + reasons

            if sentiments:
                #print(classification_output)
                fig = get_classification(sentiments)
                # col5, col6= st.columns((1, 1))
                col1.write(fig)

            if entities:
                #print(classification_output)
                fig = get_classification(entities)
                # col7, col8= st.columns((1, 1))
                col2.write(fig)

            if reasons:
                #print(classification_output)
                fig = get_classification(reasons)
                # col7, col8= st.columns((1, 1))
                col3.write(fig)

