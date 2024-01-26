import pandas as pd
import streamlit as st
import numpy as np

df = pd.read_csv("J:\\Study Material\\Mini Project-Sem 4\\Training.csv")

question_set = []
for i in df.columns:
    question_set.append(i)

symptom_q = []
question_set.pop(133)
question_set.pop(132)

symptom_q = st.multiselect("SELECT SYMPTOMS : ", question_set)

train = np.zeros((1, 132))
train = pd.DataFrame(train, columns=question_set, index=[0])

for i in symptom_q:
    train[i] = 1

final_q_report = pd.merge(train, df, how='inner', on=symptom_q)
report = []

for i in final_q_report['prognosis']:
    report.append(i)
report = set(report)


