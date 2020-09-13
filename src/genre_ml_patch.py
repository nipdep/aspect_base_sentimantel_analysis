
#import libraries
import pickle
from pandas import read_csv,DataFrame

# import weight metrics of trained models
with open('../Sup/sen_svm_model1.0.sav','rb') as pf:
    sen_model = pickle.load(pf)

with open('../Sup/gen_kmn_model1.0.sav','rb') as pf:
    gen_model = pickle.load(pf)

# prediction gen. function
def predictions(csv_path):
    data = read_csv(csv_path)
    reviews = data['review'].values

    sen_results = sen_model.predict(reviews)
    gen_results = gen_model.predict(reviews)

    genres = [[],[],[],[],[]]
    for i in range(len(gen_results)):
        if gen_results[i]==0:
            genres[0].append(i)
        elif gen_results[i]==1:
            genres[1].append(i)
        elif gen_results[i]==2:
            genres[2].append(i)
        elif gen_results[i]==3:
            genres[3].append(i)
        else:
            genres[4].append(i)
    gen_qua = []
    for gens in genres:
        c = 0
        for i in gens:
            if sen_results[i]==1:
                c+=1
        if len(gens)!=0:
            prec = c/len(gens)
        else:
            prec = 0.0
        gen_qua.append(prec)
    fin_result =  {'Action':gen_qua[0],'Horror':gen_qua[1],'Comedy':gen_qua[2],'Romance':gen_qua[3],'Animation':gen_qua[4]}
    sen_result = list(map(lambda x:'Positive' if x == 1 else 'Negative',sen_results))
    sen_res = DataFrame(sen_result)
    gen_res = DataFrame(gen_results)
    data['Genre result'] = gen_res
    data['Sentiment result'] = sen_res
    print(fin_result)
    return data,fin_result

#call to the function
predictions('../Sup/test.csv')
