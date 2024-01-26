import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
st.title("Medical Test Recommender")

load = st.checkbox("CHECK TO LOAD")
if load:
    @st.cache(allow_output_mutation=True)
    def load_data():
        df1 = pd.read_csv("https://raw.githubusercontent.com/John-Alex07/MINI_PR0JECT/main/df_diseases.csv", index_col='Unnamed: 0')
        dat = df1.drop(columns='link')
        dat.fillna("Unknown", inplace=True)
        return dat


    data = load_data()
    t_dat = data


    @st.cache
    def stop_word_add():
        st_wrd = stopwords.words('english')
        li = ["Unknown", ",", ".", "'", "`", "[", "]", "(", ")", 'The', ":", "include", 'sometimes', 'signs', 'sign',
              'symptoms', 'doctor', 'may', 'See', 'worry', ' ', '"', '\n', '\t' ]
        for l_i in li:
            st_wrd.append(l_i)
            return st_wrd


    stop_words = stop_word_add()

    @st.cache
    def stemming_func():
        data['symptoms'] = data['symptoms'].apply(lambda x: nltk.word_tokenize(''.join(x)))
        data['symptoms'] = data['symptoms'].apply(lambda x: [stemmer.stem(y) for y in x if y not in set(stop_words)])


    stemmer = PorterStemmer()
    stemming_func()
    symptoms_patient = st.text_input("State Patient Symptoms : ")
    symptoms_patient = nltk.word_tokenize(symptoms_patient)
    symptoms_patient = [stemmer.stem(x) for x in symptoms_patient if x not in set(stop_words)]

    st.write("DISEASE DIAGNOSTICS : ")
    final_report = []

    for i in range(len(data['symptoms'])):
        count = 0
        for check_word in data['symptoms'][i]:
            if check_word in symptoms_patient:
                count = count + 1
        if count == (len(symptoms_patient)):
            final_report.append(i)

    disease_report = []
    for i in final_report:
        disease_report.append(t_dat.iloc[i])

    disease_report = pd.DataFrame(disease_report)

    for i in disease_report.index:
        disease_report.append(disease_report.loc[[i]])
    if 'symptoms' in disease_report.columns:
        disease_report = disease_report.drop(columns=['symptoms', 'overview'])
    st.dataframe(disease_report)

    nxt_report = st.checkbox("For further Survey")
    if nxt_report:
        df = pd.read_csv("https://raw.githubusercontent.com/John-Alex07/MINI_PR0JECT/main/Training.csv")

        question_set = []
        for i in df.columns:
            question_set.append(i)

        symptom_q = []
        question_set.pop(133)
        question_set.pop(132)

        symptom_q = st.multiselect("SELECT SYMPTOMS : ", question_set)
        if len(symptom_q) > 0:
            train = np.zeros((1, 132))
            train = pd.DataFrame(train, columns=question_set, index=[0])

            for i in symptom_q:
                train[i] = 1

            final_q_report = pd.merge(train, df, how='inner', on=symptom_q)
            report = []

            for i in final_q_report['prognosis']:
                report.append(i)
            report = set(report)
            count_report = 1
            c1, c2 = st.columns(2)
            for i in report:
                c1.metric(label="", value=count_report)
                c2.metric(label="POSSIBLE :", value=i)
                count_report = count_report + 1

else:
    st.warning('CHECK THE BOX TO INITIATE THE ENGINE')
