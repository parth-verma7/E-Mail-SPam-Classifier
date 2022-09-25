import pandas as pd
df=pd.read_csv('spam.txt')
print(df.head())
print(df.groupby('Category').describe())

# hme numbers mein convert krna h dono columns ko
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
print(x_train_count.toarray()[:3])

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

print(model.fit(x_train_count,y_train))

emails=[
    'Hey mohan can we get together to wacth football match tomorrow??',
    'Upto 20% off on parking, exclusive offer just for u. Dont miss thus reward!',
    '3 multiply by 2 is 6'
]
emails_count=v.transform(emails)
print(model.predict(emails_count))

x_test_count=v.transform(x_test)
print(model.score(x_test_count,y_test))

# ALITER

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
print(clf)

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
clf.predict(emails)