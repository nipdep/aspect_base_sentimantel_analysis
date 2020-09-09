# import libraries
from pickle import load
from tensorflow.keras.models import load_model
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd

#import trained models parameters
#sentiment analysis model
model = load_model('./Sup/sentimental_model.h5')
model.load_weights('./Sup/sentimental_model_weights.h5')

# loading tokenizer
with open('./Sup/sentimental_tokenizer.pickle', 'rb') as handle:
    tokenizer = load(handle)

# loading labeleEncoder
with open('./Sup/sentimental_labeleEncorder.pkl', 'rb') as handle:
    label_encoder = load(handle)

new_review = ['A wonderful little production.']

#chunks = [token.lemma_ for token in new_review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]
#print(chunks)
test_sentiment_terms = []
for review in nlp.pipe(new_review):
        if review.is_parsed:
            test_sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            test_sentiment_terms.append('')
print(test_sentiment_terms)
test_sentiment_terms = tokenizer.texts_to_matrix([test_sentiment_terms[0]])
pred = model.predict(test_sentiment_terms).astype('int32')
test_sentiment = label_encoder.inverse_transform([[2]])
print(test_sentiment)


