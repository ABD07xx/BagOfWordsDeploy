import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
nltk.download("punkt")

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

st.title("Wikipedia Scraper")

def scrape_wikipedia(inp):
    link  = 'https://www.google.com/search?q=' + str(inp) + "+Wikipedia"
    link  = link.replace(' ','+')
    st.caption(link)

    res = requests.get(link)
    soup = BeautifulSoup(res.text,'html.parser')

    for sp in soup.find_all('div'):
        try:
            link = sp.find('a').get('href')
            if 'en.wikipedia.org' in link:
                break
        except:
            pass
    link = (link[7:]).split('&')[0]
    st.write(link)

    res = requests.get(link)
    soup = BeautifulSoup(res.text,'html.parser')

    heading = soup.find('h1').text
    st.write(heading)

    paragraphs = ''
    for p in soup.find_all('p'):
        paragraphs += p.text + '\n'

    return paragraphs.strip()

inp = st.text_input("Enter Person's Name")
if inp:
    paragraphs = scrape_wikipedia(inp)
    st.download_button('Download Text File', data=paragraphs, file_name=f'{inp}_wikipedia.txt')

st.title("Bag of Words")
if st.checkbox("Click"):
    if inp:
        sentences = nltk.sent_tokenize(paragraphs)
        corpus=[]
        for i in range(len(sentences)):  
            review = re.sub('[^a-zA-Z]',' ',sentences[i])
            review = review.lower()
            corpus.append(review)

        updated_corpus,updated_corpus_2=[],[]

        for i in corpus:
            words = nltk.word_tokenize(i)
            for word in words:
                if word not in set(stopwords.words('english')):
                    filtered_words = stemmer.stem(word)
                    updated_corpus.append(filtered_words)
                    filtered_words2= lemma.lemmatize(word)
                    updated_corpus_2.append(filtered_words2)

        st.write("Stemmed Corpus:")
        st.text(updated_corpus)

        st.write("Lemmatized Corpus:")
        st.text(updated_corpus_2)

st.title("Vectors")
if st.checkbox("Visualize Vectors"):
    show_vectors = st.checkbox("Show Vectors")
    if show_vectors:
        cv = CountVectorizer()
        X = cv.fit_transform(updated_corpus).toarray()
        st.write("Stemmed Vectors:")
        st.write(X)
        cv = CountVectorizer()
        X = cv.fit_transform(updated_corpus_2).toarray()
        st.write("Lemmatized Vectors:")
        st.write(X)


st.title("Visualise")
if st.checkbox("Visualize Word Frequencies"):
    if inp:
        stemmed_word_freq = Counter(updated_corpus)
        stemmed_top_words = [word for word, freq in stemmed_word_freq.most_common(20)]
        stemmed_top_freq = [freq for word, freq in stemmed_word_freq.most_common(20)]
        fig_stemmed = px.bar(x=stemmed_top_words, y=stemmed_top_freq,
                            labels={'x': 'Words', 'y': 'Frequency'},
                            title="Top 20 Most Common Words in Stemmed Corpus")
        st.header("Top 20 Most Common Words in Stemmed Corpus")
        st.plotly_chart(fig_stemmed)

        lemmatized_word_freq = Counter(updated_corpus_2)
        lemmatized_top_words = [word for word, freq in lemmatized_word_freq.most_common(20)]
        lemmatized_top_freq = [freq for word, freq in lemmatized_word_freq.most_common(20)]
        fig_lemmatized = px.bar(x=lemmatized_top_words, y=lemmatized_top_freq,
                                labels={'x': 'Words', 'y': 'Frequency'},
                                title="Top 20 Most Common Words in Lemmatized Corpus")
        st.header("Top 20 Most Common Words in Lemmatized Corpus")
        st.plotly_chart(fig_lemmatized)
