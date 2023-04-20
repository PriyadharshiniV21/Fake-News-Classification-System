import streamlit as st

import re
import nltk
import sklearn
import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from PIL import Image

st.set_page_config(layout = 'wide')

cover_photo = Image.open('cover.jpg')
st.image(cover_photo, use_column_width = True)

lr = load('logistic')
dt = load('decision')
mnb = load('multinomial')
bnb = load('bernoulli')
#rf = load('random')
xgb = load('xgb')

test_df = pd.read_csv('test.zip')
x = test_df.news
y = test_df.label

scores = [lr.score(x, y), dt.score(x, y), mnb.score(x, y), bnb.score(x, y), xgb.score(x, y)]
accuracy = []
for i in scores:
     accuracy.append(i * 100)

Technique = ['LogisticRegression', 'DecisionTreeClassifier', 'MultinomialNB', 'BernoulliNB', 'XGBClassifier']
results = pd.DataFrame({'Model': Technique, 'Accuracy': accuracy})
results = results.sort_values('Accuracy', ascending = False)

color = sns.color_palette('PuBuGn', as_cmap = True)
score = results.style.background_gradient(cmap = color)

tab1, tab2 = st.tabs(['Home', 'Classification'])

#page = st.radio('Navigate to', ['Home', 'Classification', '⚙️'], horizontal = True)

#if page == 'Home':
with tab1:

    st.title('Fake News Classification')
    st.write("Fake news is a growing problem in today's society. It is often used to spread misinformation, create confusion, and even influence political outcomes. With the rise of social media, it has become increasingly easy for fake news to spread quickly and reach a wide audience. In order to combat this problem, many researchers and organizations have turned to **machine learning** algorithms to classify news articles as either **_real_** or **_fake_**.")
    st.write("Fake news classification is a task that involves using **natural language processing** techniques to analyze text and determine if the information presented in the text is accurate or not. There are a number of different approaches to this task, including supervised learning, unsupervised learning, and deep learning.")

    col1, col2, col3 = st.columns([0.5, 6, 0.5])
    with col2:
        img = Image.open('fake.jpg')
        st.image(img)

    st.header('WELFake Dataset')
    st.write("WELFake is a dataset of **72,134 news articles** with **35,028 real** and **37,106 fake** news. For this, authors merged four popular news datasets _**(i.e. Kaggle, McIntire, Reuters, BuzzFeed Political)**_ to prevent over-fitting of classifiers and to provide more text data for better ML training.")

    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        sty_img = Image.open('stylecloud.png')
        st.image(sty_img)

    st.header('Models used for Text Classification')
    col1, col2, col3 = st.columns([0.5, 1, 1])
    with col2:
        st.table(score)

#elif page == 'Classification':
with tab2:

    def preprocess_text(text):

        port = PorterStemmer()
        sw = stopwords.words('english')
        corpus = []

        line = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
        line = [port.stem(word) for word in line if word not in sw]
        corpus.append(" ".join(line))
        return corpus

    models = ['LogisticRegression', 'DecisionTreeClassifier', 'MultinomialNB', 'BernoulliNB',
              'XGBClassifier']

    cls = []
    cls_labels = {0: 'FAKE', 1: 'REAL'}

    def predict(text):

        text = preprocess_text(text)

        if model_choice == 'LogisticRegression':
            cls.append(lr.predict(text)[0])
        elif model_choice == 'DecisionTreeClassifier':
            cls.append(dt.predict(text)[0])
        elif model_choice == 'MultinomialNB':
            cls.append(mnb.predict(text)[0])
        elif model_choice == 'BernoulliNB':
            cls.append(bnb.predict(text)[0])
        #elif model_choice == 'RandomForestClassifier':
        #    cls.append(rf.predict(text)[0])
        elif model_choice == 'XGBClassifier':
            cls.append(xgb.predict(text)[0])
        else:
            return None

        for k, v in cls_labels.items():
            if cls[0] == k:
                return v

    text = st.text_area('**Enter a text**')

    st.subheader('**Word Cloud**')
    st.info('Generating WordCloud...')

    if text:
        def gen_wordcloud():
            wc = WordCloud(background_color = 'white', width = 1500, height = 1000).generate(text)
            plt.imshow(wc, interpolation = 'bilinear')
            plt.axis("off")
            plt.savefig('WC.jpg')
            img = Image.open("WC.jpg")
            return img

        img = gen_wordcloud()
        st.image(img)

    st.subheader('**Classify Text**')
    model_choice = st.selectbox('**Choose ML model to classify text**', models)

    if st.button('Classify Text'):
        st.markdown('**Original Text:**')
        st.markdown(text)
        classify = predict(text)
        st.success(f'**The given news text is {classify}**')

#elif page == "⚙️":
#with tab3:

    #profile_photo = st.file_uploader('Profile Photo', type = ['jpg', 'jpeg', 'png'])
    #name = st.text_input('Name')
    #bio = st.text_area('Bio')

    #if profile_photo is not None:
    #    st.image(profile_photo, width = 150)
    #if name:
    #    st.write(f'Name: {name}')
    #if bio:
    #    st.write(f'Bio: {bio}')
