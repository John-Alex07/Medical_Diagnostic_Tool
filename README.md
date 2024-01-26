# Medical Test Recommender

![Medical Test Recommender](https://github.com/John-Alex07/Portfolio/blob/master/static/img/portfolio/portfolio-2.jpg)

This Streamlit app provides medical test recommendations based on patient symptoms. The application utilizes a dataset of diseases, symptoms, and associated information to offer diagnostic insights. Users can interact with the app by inputting symptoms and exploring further survey options for a detailed diagnosis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Code Explanation](#code-explanation)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction

This Streamlit app is designed to assist users in identifying potential diseases based on symptoms. The user-friendly interface allows for easy input of symptoms and provides diagnostic reports. The app also offers the option to explore additional survey questions for a more detailed diagnosis.

## Features

- **Data Loading:** Check the box to load a dataset of diseases and symptoms.
- **Symptom Input:** Input patient symptoms to receive diagnostic recommendations.
- **Further Survey:** Optionally proceed with additional survey questions for a more detailed diagnosis.
- **Clear User Interface:** Intuitive design for seamless interaction.

## Dependencies

Make sure you have the following Python libraries installed:

- `numpy`
- `streamlit`
- `pandas`
- `nltk`

```bash
pip install numpy
pip install streamlit
pip install pandas
pip install nltk
python -m nltk.downloader stopwords punkt
```
## Code Explanation

### 1. Data Loading and Preprocessing
This section handles the loading and preprocessing of the medical data. It uses Streamlit widgets for user interaction and employs caching to optimize data loading. NLTK resources are downloaded, and custom stopwords are added. The stemming_func function applies tokenization and stemming to the symptom data, preparing it for further analysis.
```python
# Load necessary libraries
import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
```

#### a. Handling Unknown Values

 In the dataset, unknown values may arise due to missing or undefined entries. Removing these unknown values ensures a cleaner and more consistent dataset, preventing potential issues during analysis and improving the accuracy of diagnostic results.
```python
if load:
    # Function to load and preprocess data
    @st.cache(allow_output_mutation=True)
    def load_data():
        df1 = pd.read_csv("./df_diseases.csv", index_col='Unnamed: 0')
        dat = df1.drop(columns='link')
        dat.fillna("Unknown", inplace=True)
        return dat

    # Load data and create a copy
    data = load_data()
    t_dat = data
```
#### b. Removal of Stop Words
Stop words are common words that do not contribute significantly to the meaning of a sentence. These words can introduce noise and hinder accurate analysis. By removing stop words, we focus on the essential content, enhancing the effectiveness of natural language processing techniques and resulting in more meaningful symptom representations.
```python
    # Function to add custom stopwords
    @st.cache
    def stop_word_add():
        st_wrd = stopwords.words('english')
        li = ["Unknown", ",", ".", "'", "`", "[", "]", "(", ")", 'The', ":", "include", 'sometimes', 'signs', 'sign',
              'symptoms', 'doctor', 'may', 'See', 'worry', ' ', '"', '\n', '\t' ]
        for l_i in li:
            st_wrd.append(l_i)
        return st_wrd

    # Add custom stopwords
    stop_words = stop_word_add()
```

#### c. Stemming
Stemming involves reducing words to their root or base form. This process is crucial for standardizing variations of words, capturing the core meaning while discarding inflections. Stemming ensures that similar symptoms expressed in different forms are treated uniformly during analysis, leading to more accurate symptom matching and disease diagnostics.
```python
    # Function for stemming symptoms
    @st.cache
    def stemming_func():
        data['symptoms'] = data['symptoms'].apply(lambda x: nltk.word_tokenize(''.join(x)))
        data['symptoms'] = data['symptoms'].apply(lambda x: [stemmer.stem(y) for y in x if y not in set(stop_words)])

    # Create a stemmer object and apply stemming
    stemmer = PorterStemmer()
    stemming_func()
```
### 2. Patient Symptoms Input and Disease Diagnostics
In this part, the user inputs their symptoms, which are then tokenized and stemmed for compatibility with the dataset. The app performs disease diagnostics by matching the patient's symptoms with the preprocessed dataset. The results are displayed in a DataFrame, providing users with information about potential diseases based on their symptoms.
```python
# Input for patient symptoms
symptoms_patient = st.text_input("State Patient Symptoms : ")
symptoms_patient = nltk.word_tokenize(symptoms_patient)
symptoms_patient = [stemmer.stem(x) for x in symptoms_patient if x not in set(stop_words)]

# Display section header for disease diagnostics
st.write("DISEASE DIAGNOSTICS : ")

# Initialize an empty list to store diagnostic results
final_report = []

# Loop through the dataset for symptom matching
for i in range(len(data['symptoms'])):
    count = 0
    for check_word in data['symptoms'][i]:
        if check_word in symptoms_patient:
            count += 1
    if count == len(symptoms_patient):
        final_report.append(i)

# Create a DataFrame with the diagnostic results
disease_report = [t_dat.iloc[i] for i in final_report]
disease_report = pd.DataFrame(disease_report)

# Drop unnecessary columns if present
if 'symptoms' in disease_report.columns:
    disease_report = disease_report.drop(columns=['symptoms', 'overview'])
```
### 3. Further Survey
This section allows users to initiate a further survey for a more detailed diagnosis. It involves loading additional survey data and enabling users to select symptoms of interest. The selected symptoms are used to filter the dataset, and the results are displayed using Streamlit columns and metrics, providing a clearer view of possible diseases.
```python
# Checkbox for initiating further survey
nxt_report = st.checkbox("For further Survey")

if nxt_report:
    # Load additional data for further survey
    df = pd.read_csv("./Training.csv")

    # Get a list of survey questions
    question_set = list(df.columns)
    question_set.pop(133)
    question_set.pop(132)

    # Allow user to select symptoms for the survey
    symptom_q = st.multiselect("SELECT SYMPTOMS : ", question_set)

    if len(symptom_q) > 0:
        # Create a DataFrame for the selected symptoms
        train = pd.DataFrame(np.zeros((1, 132)), columns=question_set, index=[0])

        for i in symptom_q:
            train[i] = 1

        # Merge with the main dataset based on selected symptoms
        final_q_report = pd.merge(train, df, how='inner', on=symptom_q)
        report = set(final_q_report['prognosis'])

        # Display survey results using Streamlit columns and metrics
        count_report = 1
        c1, c2 = st.columns(2)
        for i in report:
            c1.metric(label="", value=count_report)
            c2.metric(label="POSSIBLE :", value=i)
            count_report += 1
```
## How to use

- Clone the repository:

  ```bash
  git clone https://github.com/John-Alex07/medical-test-recommender.git
  cd medical-test-recommender
  ```
- Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```
- Run the App
  ``` bash
  streamlit run app.py
  ```
- Interact with the app through the provided widgets for  a medical diagnosis!
## License

This project is licensed under the [MIT License](LICENSE).

[MIT License](https://opensource.org/licenses/MIT)

 

