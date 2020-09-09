
# import libraries
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from numpy import array,vectorize
from pandas import read_csv,DataFrame
nlp = spacy.load('en_core_web_sm')

#import trained models' parameters
#categorical model
cat_model = load_model('../Sup/categorical_genre_model.h5')
cat_model.load_weights('../Sup/categorical_genre_model_weights.h5')

# loading tokenizer
with open('../Sup/categorical_genre_tokenizer.pickle', 'rb') as handle:
    cat_tokenizer = load(handle)

# loading labeleEncoder
with open('../Sup/categorical_genre_labeleEncorder.pkl', 'rb') as handle:
    cat_label_encoder = load(handle)
#senriment analysis model
sen_model = load_model('../Sup/sentimental_model.1.h5')
sen_model.load_weights('../Sup/sentimental_model_weights1.1.h5')

# loading tokenizer
with open('../Sup/sentimental_1_tokenizer.pickle', 'rb') as handle:
    sen_tokenizer = load(handle)

# loading labeleEncoder
def binarizing(a):
    if a == 'positive':
        return 1
    else:
        return 0

bina = vectorize(binarizing)
# labels list
categories = ['Action', 'Horror', 'Romance',
              'Comedy', 'Animation']
# init nlp preprocessing libraries
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('br')

from string import punctuation
table = str.maketrans('', '', punctuation)
# define a function to cooperates with two models
def predictions(csv_path):
    data = read_csv(csv_path)
    reviews = data['review']

    test_reviews = [review.lower() for review in reviews]
    test_aspect_terms = []
    filtered_str = []
    for review in test_reviews:
        tokens = word_tokenize(review)
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        words = ' '.join([w for w in words if not w in stop_words])
        filtered_str.append(words)

    for review in nlp.pipe(filtered_str):
        chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
        test_aspect_terms.append(' '.join(chunks))

    filtered = array(filtered_str)
    test_aspect_terms = array(test_aspect_terms)
    sequence = cat_tokenizer.texts_to_sequences(test_aspect_terms)
    gen_tokenized = pad_sequences(sequence,maxlen=120,padding='post',truncating='post',value=0)

    sequence = sen_tokenizer.texts_to_sequences(filtered)
    sen_tokenized = pad_sequences(sequence,maxlen=120,padding='post',truncating='post',value=0)

    test_aspect_categories = cat_label_encoder.inverse_transform(cat_model.predict_classes(gen_tokenized))
    test_sentiment = sen_model.predict_classes(sen_tokenized)

    categorized = [[],[],[],[],[]]
    for ind,i in enumerate(test_aspect_categories):
        if i == 'Action':
            categorized[0].append(ind)
            continue
        elif i == 'Horror':
            categorized[1].append(ind)
            continue
        elif i == 'Comedy' :
            categorized[2].append(ind)
            continue
        elif i == 'Romance' :
            categorized[3].append(ind)
            continue
        elif i == 'Animation' :
            categorized[4].append(ind)

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

# call to the function
predictions('../Sup/test.csv')
