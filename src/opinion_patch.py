# final prediction model
# Input : csv file with filled one column of reviews of a movie
# Output : percentages of positive reviews on each 6 categories.

from pickle import load
from tensorflow.keras.models import load_model
import spacy
from pandas import read_csv,DataFrame
nlp = spacy.load('en_core_web_sm')

cat_model = load_model('../Sup/categorical_model.h5')
cat_model.load_weights('../Sup/categorical_model_weights.h5')

# loading tokenizer
with open('../Sup/categorical_tokenizer.pickle', 'rb') as handle:
    cat_tokenizer = load(handle)

# loading labeleEncoder
with open('../Sup/categorical_labeleEncorder.pkl', 'rb') as handle:
    cat_label_encoder = load(handle)

sen_model = load_model('../Sup/sentimental_model.h5')
sen_model.load_weights('../Sup/sentimental_model_weights.h5')

# loading tokenizer
with open('../Sup/sentimental_tokenizer.pickle', 'rb') as handle:
    sen_tokenizer = load(handle)

# loading labeleEncoder
with open('../Sup/sentimental_labeleEncorder.pkl', 'rb') as handle:
    sen_label_encoder = load(handle)

categories = ['DIRECTING#PERFORMANCE', 'WRITING#PERFORMANCE', 'CAST#PERFORMANCE',
              'PERFOMANCE#GENERAL', 'CREW#PERFORMANCE', 'PRODUCTION#PERFOMANCE']

def predictions(csv_path):
    data = read_csv(csv_path)
    reviews = data['review']

    test_reviews = [review.lower() for review in reviews]
    test_aspect_terms = []
    for review in nlp.pipe(test_reviews):
        chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
        test_aspect_terms.append(' '.join(chunks))
    test_aspect_terms = DataFrame(cat_tokenizer.texts_to_matrix(test_aspect_terms))

    # Sentiment preprocessing
    test_sentiment_terms = []
    for review in nlp.pipe(test_reviews):
        if review.is_parsed:
            test_sentiment_terms.append(' '.join([token.lemma_ for token in review if (
                        not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            test_sentiment_terms.append('')
    test_sentiment_terms = DataFrame(sen_tokenizer.texts_to_matrix(test_sentiment_terms))

    test_aspect_categories = cat_label_encoder.inverse_transform(cat_model.predict_classes(test_aspect_terms))
    test_sentiment = sen_label_encoder.inverse_transform(sen_model.predict_classes(test_sentiment_terms))

    categorized = [[],[],[],[],[],[]]
    for ind,i in enumerate(test_aspect_categories):
        if i == 'DIRECTING#PERFORMANCE':
            categorized[0].append(ind)
            continue
        elif i == 'WRITING#PERFORMANCE':
            categorized[1].append(ind)
            continue
        elif i == 'CAST#PERFORMANCE' :
            categorized[2].append(ind)
            continue
        elif i == 'PERFOMANCE#GENERAL' :
            categorized[3].append(ind)
            continue
        elif i == 'CREW#PERFORMANCE' :
            categorized[4].append(ind)
        elif i == 'PRODUCTION#PERFOMANCE':
            categorized[5].append(ind)

    result = {}
    for ind in range(len(categorized)):
        lis = categorized[ind]
        tol = len(lis)
        pos = 0
        for val in lis:
            if test_sentiment[val] == 'Neutral':
                pos+=1
        result[categories[ind]] = pos/tol if tol != 0 else 0.0

    print(result)

predictions('./Sup/test.csv')


# sentiments = ['Positive','Negative','Neutral']
