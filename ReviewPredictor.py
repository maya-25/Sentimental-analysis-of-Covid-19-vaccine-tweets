
import streamlit as st
import nltk
from ast import literal_eval
import base64
import numpy as np
import pickle as pkl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import load_model, Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
stop = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}

snostem = nltk.stem.SnowballStemmer('english')

model= pkl.load(open("counttokeniseFinal-model.pkl","rb"))
model1= load_model("model_corona_final_tweets.h5")


#re html tags and punc funcs
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',sentence)
    return cleantext
#sentence = 'I am <> Om'
#cleanhtml(sentence)

#func for removing punctuations
import re
# def cleanpunc(sentence):
#     cleaned = re.sub('[?|!|\'|"|#]',' ',sentence)
#     cleaned = re.sub('[.|,|(|)|\|/]',' ',cleaned)
#     return cleaned
def clean_data(tex):
    tex = re.sub(r'@\w+', ' ', tex)
    tex = re.sub(r'#', ' ', tex)
    tex = re.sub(r'[^\w]', ' ', tex)
    tex = re.sub(r'RT[\s]+', ' ', tex)
    tex = re.sub(r'https?:\/\/\S+', ' ', tex)
    tex = tex.lower()
    
    return tex


st.set_page_config(page_title="Review Classifier",page_icon="⚕️",layout="centered",initial_sidebar_state="expanded")


def preprocess(review):   
    
    #Total processing
    def text_preprocessing(final): 
        i=0
        str=''
        s=''
        filtered_sentence=[]
        final = cleanhtml(final)
        for w in final.split():
            for cleaned_words in clean_data(w).split(): #after cleanpunc again one word may become two or multiple so they'll be split and stored as different words
                if((cleaned_words.isalpha()&(len(cleaned_words)>2))):
                    if(cleaned_words.lower() not in stop):
                        s = snostem.stem(cleaned_words.lower())
                        filtered_sentence.append(s)
            str = " ".join(filtered_sentence)
        return str


    # Pre-processing user input   
    rev=model.texts_to_sequences([text_preprocessing(review)])
    rev1=pad_sequences(rev,padding='post',maxlen=31)
    prediction = np.argmax(model1.predict(rev1),axis=-1)

    return prediction

    

       
    # front end elements of the web page 
    
html_temp = """ 
    <div id=1 style ="background-color:	#00008B;padding:10px"> 
    <h1 style ="color:#D8DBFA;text-align:center;">Covid-19 Vaccine tweet Polarity Predictor</h1> 
    </div>
    
    <style>
    body {
    background-image: url("Covid-19_Vaccine.jpg");
    background-size: cover;
    }
    </style>
    """
html1 = """<h2 style ="color:blue;text-align:right;"> -Mayank</h2> """
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
main_bg = "Corona_vaccine.jpg"
main_bg_ext = "jpg"

side_bg = "Covid-19_Vaccine.jpg"
side_bg_ext = "jpg"

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: cover;
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(html1, unsafe_allow_html = True)
st.markdown("""<h2 style ="color:blue;text-align:left;">Enter your review below</h2> """, unsafe_allow_html = True)
st.markdown("""<h2 style ="color:blue;text-align:center;">0: Negative 1: Neutral 2: Positive</h2>""",unsafe_allow_html = True)   
# following lines create boxes in which user can enter data required to make prediction
review=st.text_area("")



#user_input=preprocess(sex,cp,exang, fbs, slope, thal )
pred=preprocess(review)




if st.button("Predict"):
    st.success(pred)
    
st.sidebar.subheader("About App")

st.sidebar.info("This web app is to predict polarity of an user review about the Covid-19 vaccine")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check the same")
st.sidebar.info("Don't forget to rate this app")



feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
    st.header("Thank you for rating the app!")
    st.info("Caution: This is just a prediction") 
