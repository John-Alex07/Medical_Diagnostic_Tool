# Medical Test Recommender

![Medical Test Recommender](link_to_your_image)

This Streamlit app provides medical test recommendations based on patient symptoms. The application utilizes a dataset of diseases, symptoms, and associated information to offer diagnostic insights. Users can interact with the app by inputting symptoms and exploring further survey options for a detailed diagnosis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [Code Explanation](#code-explanation)
- [Credits](#credits)
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

### 2. Handling Unknown Values
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
### 3. Removal of Stop Words
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

### 4. Stemming
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
