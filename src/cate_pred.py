
from pickle import load
from tensorflow.keras.models import load_model
import spacy
nlp = spacy.load('en_core_web_sm')

model = load_model('./Sup/categorical_model.h5')
model.load_weights('./Sup/categorical_model_weights.h5')

# loading tokenizer
with open('./Sup/categorical_tokenizer.pickle', 'rb') as handle:
    tokenizer = load(handle)

# loading labeleEncoder
with open('./Sup/categorical_labeleEncorder.pkl', 'rb') as handle:
    label_encoder = load(handle)

new_review = "This italian place is nice and cosy"

chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']
new_review_aspect_terms = ' '.join(chunks)
new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])
print(new_review_aspect_tokenized)
new_review_category = model.predict_classes(new_review_aspect_tokenized)
print(new_review_aspect_tokenized)
new_review_category = label_encoder.inverse_transform(model.predict_classes(new_review_aspect_tokenized))
print(new_review_category)

