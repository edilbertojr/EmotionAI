import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title = "EmotionAI",
    page_icon = "😀",
    layout = "centered",
    initial_sidebar_state = "collapsed"
)

st.markdown(
    body = "# <center>EmotionAI</center>",
    unsafe_allow_html = True
)
st.caption("Fine-Tuned with DistilBERT Transformers for optimal performance. Capable of distinguishing between six emotions: `Joy`, `Sadness`, `Anger`, `Fear`, `Love`, `Surprise`")

text_input = st.text_area(
    label = "Provide your text.",
    value = "I saw a movie today and it was really good."
)

model_id = "iamsubrata/distilbert-base-uncased-finetuned-emotion"

classifier = pipeline(
    task = "text-classification",
    model = model_id
)

if text_input:
    button  = st.button(
        label = "Emotion",
        key = "emotion_button"
    )

    if button:
        labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        predictions = classifier(
            text_input, 
            top_k = 6
        )
        predictions_df = pd.DataFrame(predictions)
        predictions_df.columns = ["Emotion", "Score"]
        predictions_df["Emotion"] = predictions_df["Emotion"].apply(lambda x: labels[int(x[-1])])
        predictions_df.sort_values(
            by = "Score",
            ascending = False
        )

        st.bar_chart(
            data = predictions_df,
            x = "Emotion",
            y = "Score"
        )




