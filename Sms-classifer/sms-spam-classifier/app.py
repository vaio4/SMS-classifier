# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
# import nltk
# # nltk.download('punkt_tab')
# # stopwords




# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# # Download necessary NLTK resources
# # nltk.download('punkt')
# # nltk.download('stopwords')

# # Check versions
# import streamlit
# import sklearn
# import nltk

# print("Streamlit version:", streamlit.__version__)
# print("Scikit-learn version:", sklearn.__version__)
# print("NLTK version:", nltk.__version__)

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Load the vectorizer and model
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # 1. Preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. Vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. Predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")









# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Check versions

# import streamlit
# import sklearn
# import nltk

# print("Streamlit version:", streamlit.__version__)
# print("Scikit-learn version:", sklearn.__version__)
# print("NLTK version:", nltk.__version__)

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Fit the TfidfVectorizer (or TfidfTransformer if you have saved it separately)
# tfidf = TfidfVectorizer()
# # Assuming you have some training data
# training_data = ["sample text for training", "another text for training"]
# tfidf.fit(training_data)

# # Serialize the fitted transformer and model for later use
# with open('vectorizer.pkl', 'wb') as f:
#     pickle.dump(tfidf, f)

# # Load the vectorizer and model
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     # 1. Preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. Vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. Predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")



import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Sample training data
training_data = ["sample text for training", "another text for training"]
training_labels = [0, 1]

# Fit the TfidfVectorizer and model
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(training_data)
model = MultinomialNB()
model.fit(X_train, training_labels)

# Serialize the fitted transformer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
