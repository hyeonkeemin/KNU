import pandas as pd
import numpy as np
import re

from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import tqdm


df = pd.read_csv('DASC705001_HW1_News-1.csv', encoding='latin1', low_memory=False)
df = df.iloc[:, :5]


def custom_nlp(pandas_text_columns):

    clean = re.compile("[^A-Za-z ]")
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    # 특수문자, 숫자 제거
    preprocessing_text = pandas_text_columns.apply(lambda x: clean.sub('', str(x)))
    # 영소문자 변경
    preprocessing_text = preprocessing_text.apply(lambda x: x.lower()).to_list()
    
    result = []
    
    for words in tqdm.tqdm(preprocessing_text):
        # 단어 토큰화
        for word in words.split():
            # 의미있는 단어를 추출하기 위하여 명사, 형용사 순으로 표제어 추출 진행
            lemma = lemmatizer.lemmatize(word, pos='n')
            lemma = lemmatizer.lemmatize(word, pos='a')            
            # 불용어 및 단어길이 2이하 제거
            if lemma not in stop_words and len(lemma) > 2:
                result.append(lemma)
    
    return result


# 가장 많이 등장한 상위 25개 단어 선정
def custom_most_common(words):
    return [i[0] for i in FreqDist(words).most_common(25)]


# title
title = custom_nlp(pandas_text_columns=df['title'])
title = custom_most_common(title)

# text
text = custom_nlp(pandas_text_columns=df['text'])
text = custom_most_common(text)

# subject_worldnews_title
worldnews_title = custom_nlp(pandas_text_columns=df[df['subject'] == 'worldnews']['title'])
worldnews_title = custom_most_common(worldnews_title)

# subject_worldnews_text
worldnews_text = custom_nlp(pandas_text_columns=df[df['subject'] == 'worldnews']['text'])
worldnews_text = custom_most_common(worldnews_text)

# subject_US_news_title
USnews_title = custom_nlp(pandas_text_columns=df[df['subject'] != 'worldnews']['title'])
USnews_title = custom_most_common(USnews_title)

# subject_US_news_text
USnews_text = custom_nlp(pandas_text_columns=df[df['subject'] != 'worldnews']['text'])
USnews_text = custom_most_common(USnews_text)


print(f'title 상위 20개 단어: {title}')
print(f'text 상위 20개 단어: {text}\n')

# 'video', 'says', 'watch', 'reuters' 등 뉴스 그 자체와 연관된 단어 및 'state' 등 의미파악이 힘든 단어 위주로 제외 후 상위 10개 선정
select_title = ['trump', 'obama', 'house', 'hillary', 'white', 'president', 'clinton', 'bill', 'russia', 'republican']
select_text = ['trump', 'president', 'people', 'house', 'government', 'clinton', 'republican', 'obama', 'united', 'white']

print(f'----- title 선정 단어: {select_title}')
print(f'----- text 선정 단어: {select_text}')


print(f'worldnews_title 상위 20개 단어: {worldnews_title}')
print(f'worldnews_text 상위 20개 단어: {worldnews_text}\n')

select_worldnews_title = ['north', 'korea', 'trump', 'china', 'minister', 'brexit', 'south', 'police', 'iran', 'russia']
select_worldnews_text = ['government', 'president', 'people', 'party', 'minister', 'united', 'north', 'military', 'country']

print(f'----- worldnews_title 선정 단어: {select_worldnews_title}')
print(f'----- worldnews_text 선정 단어: {select_worldnews_text}')


print(f'USnews_title 상위 20개 단어: {USnews_title}')
print(f'USnews_text 상위 20개 단어: {USnews_text}\n')

select_USnews_title = ['trump', 'obama', 'house', 'hillary', 'white', 'clinton', 'president', 'bill', 'republican', 'black']
select_USnews_text = ['trump', 'president', 'people', 'house', 'clinton', 'republican', 'obama', 'white', 'campaign', 'united']

print(f'----- USnews_title 선정 단어: {select_USnews_title}')
print(f'----- USnews_text 선정 단어: {select_USnews_text}')


corpus = [(' ').join(i) for i in [select_title, select_text,select_worldnews_title, select_worldnews_text, select_USnews_title, select_USnews_text]]
tfidfv = TfidfVectorizer().fit(corpus)


print(f'tf-idf 결과: {tfidfv.transform(corpus).toarray()}\n')
print(f'tf-idf 벡터 사이즈: {tfidfv.transform(corpus).toarray().shape}\n')
print(tfidfv.vocabulary_)


