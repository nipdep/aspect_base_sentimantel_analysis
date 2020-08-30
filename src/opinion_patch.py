# final prediction model
# Input : csv file with filled one column of reviews of a movie
# Output : percentages of positive reviews on each 6 categories.

from pickle import load
from tensorflow.keras.models import load_model
import spacy
from pandas import read_csv,DataFrame
nlp = spacy.load('en_core_web_sm')

cat_model = load_model('./Sup/categorical_model.h5')
cat_model.load_weights('./Sup/categorical_model_weights.h5')

# loading tokenizer
with open('./Sup/categorical_tokenizer.pickle', 'rb') as handle:
    cat_tokenizer = load(handle)


sen_model = load_model('./Sup/sentimental_model.h5')
sen_model.load_weights('./Sup/sentimental_model_weights.h5')

# loading tokenizer
with open('./Sup/sentimental_tokenizer.pickle', 'rb') as handle:
    sen_tokenizer = load(handle)


categories = ['DIRECTING PERFORMANCE', 'CAST PERFORMANCE']

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

    test_aspect_categories = cat_model.predict_classes(test_aspect_terms)
    test_sentiment = sen_model.predict_classes(test_sentiment_terms)

    sen_column = DataFrame(list(map(lambda x:'DIRECTING PERFORMANCE' if x==0 else 'CAST PERFORMANCE',test_sentiment)))
    cat_column = DataFrame(list(map(lambda y:'Positive' if y==1 else 'Negative',test_aspect_categories)))
    data['sentiment_results'] = sen_column
    data['categorical_result'] = cat_column

    categorized = [[],[]]
    for ind,i in enumerate(test_aspect_categories):
        if i == 0:
            categorized[0].append(ind)
        elif i == 1:
            categorized[1].append(ind)


    result = {}
    for ind in range(len(categorized)):
        lis = categorized[ind]
        tol = len(lis)
        pos = 0
        for val in lis:
            if test_sentiment[val] == 1:
                pos+=1
        result[categories[ind]] = pos/tol if tol != 0 else 0.0

    print(result)

    return data,result


predictions('./Sup/test.csv')


# sentiments = ['Positive','Negative','Neutral']
