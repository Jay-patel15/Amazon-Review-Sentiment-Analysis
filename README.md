

# 🛒 Amazon Review Sentiment Analysis  

This project analyzes **Amazon product reviews** using **Natural Language Processing (NLP)** techniques in **Google Colab**. The analysis is performed using **VADER** sentiment analysis and the **RoBERTa pre-trained model** to classify sentiments accurately.  

## 🚀 Features  
- **VADER Sentiment Analysis** for quick, rule-based sentiment scoring.  
- **RoBERTa Transformer Model** for deep learning-based sentiment classification.  
- **Exploratory Data Analysis (EDA)** including sentiment distributions and comparisons.  
- **Data Visualization** using Matplotlib and Seaborn.  
- **Comparison of VADER and RoBERTa sentiment scores** to understand model behavior.  

---

## 📌 Tech Stack & Libraries  
- **Platform:** Google Colab (Jupyter Notebook)  
- **Data Handling:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **NLP Processing:** `nltk`  
- **Sentiment Analysis:** `VADER (nltk.sentiment)`, `Transformers (Hugging Face)`  
- **Machine Learning:** `scipy.special.softmax`, `transformers`  

---

## 🚀 Getting Started (Google Colab)  

### 🔹 Step 1: Open in Google Colab  
Click the link below to open the project in **Google Colab**:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BOU20wdaELCpZXPnQb6HZ1xUbVds684j)  

### 🔹 Step 2: Install Required Libraries  
Run the following command in Colab to install missing dependencies:  
```python
!pip install nltk pandas numpy matplotlib seaborn transformers tqdm
```

### 🔹 Step 3: Import Necessary Libraries  
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  

import nltk  
from nltk.sentiment import SentimentIntensityAnalyzer  
from tqdm.notebook import tqdm  
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
from scipy.special import softmax  
```

### 🔹 Step 4: Download NLTK Resources  
```python
nltk.download('punkt')  
nltk.download('averaged_perceptron_tagger')  
nltk.download('maxent_ne_chunker')  
nltk.download('words')  
nltk.download('vader_lexicon')  
```

---

## 📊 Project Workflow  

### **Step 1: Load and Explore Data**  
- Load **Amazon Reviews Dataset (`Reviews.csv`)**.  
- Extract the **first 500 reviews** for quick analysis.  
- Plot **distribution of review scores** (1-star to 5-star).  

```python
df = pd.read_csv('Reviews.csv')
df = df.head(500)

ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()
```

---

### **Step 2: VADER Sentiment Analysis**  
- Tokenize text and apply **VADER sentiment scoring**.  
- Extract **positive, neutral, negative, and compound sentiment scores**.  

```python
sia = SentimentIntensityAnalyzer()
df['vader_sentiment'] = df['Text'].apply(lambda text: sia.polarity_scores(text)['compound'])
```

---

### **Step 3: RoBERTa Transformer Model Sentiment Analysis**  
- Load **CardiffNLP's `twitter-roberta-base-sentiment` model**.  
- Tokenize and process text using **Hugging Face's Transformers library**.  
- Compute **softmax probability scores** to classify sentiment.  

```python
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    return {'roberta_neg': scores[0], 'roberta_neu': scores[1], 'roberta_pos': scores[2]}

df['roberta_sentiment'] = df['Text'].apply(polarity_scores_roberta)
```

---

### **Step 4: Compare Sentiment Analysis Models**  
- Combine **VADER and RoBERTa results** into a single DataFrame.  
- Plot **sentiment score distribution per review star rating**.  

```python
sns.pairplot(data=df, vars=['vader_sentiment', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
plt.show()
```

---

## 📸 Visualizations  

1. **Distribution of Review Scores**  
   ```python
   ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
   ax.set_xlabel('Review Stars')
   plt.show()
   ```
   

2. **VADER vs. RoBERTa Sentiment Comparison**  
   ```python
   sns.pairplot(data=df, vars=['vader_sentiment', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
   plt.show()
   ```
   

---

## 📜 Observations & Insights  
- **VADER** is good at identifying **strongly worded** sentiments but struggles with **context and sarcasm**.  
- **RoBERTa** performs better in **nuanced sentiment detection** but is computationally heavier.  
- **Hybrid approach** combining VADER and RoBERTa can improve real-world sentiment analysis.  

---

## 🛠 Future Improvements  
- Support **multi-language sentiment analysis**.  
- Deploy as a **web application** using Flask/Django.  
- Implement **real-time review analysis for e-commerce platforms**.  

---
---

## 🔗 Colab Notebook  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BOU20wdaELCpZXPnQb6HZ1xUbVds684j)  

---
---
## 🔗 Dataset link  
[![Open in Google Drive](https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Google_Drive_logo.svg/1200px-Google_Drive_logo.svg.png)](https://drive.google.com/file/d/1i_tsF-2MOPwKRvgXBSjHkKKqsFgvdhzQ/view?usp=sharing)


---
Let me know if you need any modifications! 🚀🔥
