# Twitter Sentiment Analysis

This project focuses on **sentiment analysis** using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The goal is to build a **logistic regression model** to classify tweets as **positive or negative** based on the sentiment expressed in the text. The project involves data preprocessing, feature extraction, model training, and evaluation.

---

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Model Accuracy](#model-accuracy)
- [Usage](#usage)
- [References](#references)

---

## Dataset

**Sentiment140 dataset with 1.6 million tweets**  
**[Dataset Link](https://www.kaggle.com/datasets/kazanova/sentiment140)**  

### About the Dataset:
- **Context:**  
  This dataset contains **1,600,000 tweets** extracted using the **Twitter API**. It has been annotated for **sentiment detection** with the following labels:
  - **0** = Negative   
  - **4** = Positive  

### Content and Fields:
1. **target**: Polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)  
2. **ids**: Tweet ID (e.g., 2087)  
3. **date**: Date and time of the tweet (e.g., Sat May 16 23:58:44 UTC 2009)  
4. **flag**: The query used to extract the tweet (e.g., "lyx"). If there’s no query, it is **NO_QUERY**.  
5. **user**: Username of the person who tweeted (e.g., robotickilldozr)  
6. **text**: The actual content of the tweet (e.g., "Lyx is cool")  

---

## Installation

### Requirements
Make sure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk pickle-mixin
```

You also need to **download the Sentiment140 dataset** from Kaggle:
```bash
!kaggle datasets download -d kazanova/sentiment140
```

---

## Project Workflow

1. **Data Loading and Preprocessing:**  
   - Extract the data from the ZIP file.  
   - Assign proper column names to the dataset.  
   - Replace positive labels from **4 to 1** and drop neutral tweets (2) to simplify the problem as binary classification.
   - Clean the text: Remove **stopwords, special characters, URLs, and numbers**.  
   - Apply **stemming** to reduce words to their root form.

2. **Feature Extraction:**  
   - Use **TF-IDF Vectorizer** to convert the text into numerical features.

3. **Train-Test Split:**  
   - Split the dataset into **80% training** and **20% test** sets.

4. **Model Building:**  
   - Train a **Logistic Regression** model using the training data.

5. **Evaluation:**  
   - Evaluate the model’s performance on both **training and test datasets** using accuracy scores.

---

## Results

- **Training Accuracy:** 79.8%  
- **Test Accuracy:** 77.8%

### Sentiment Distribution  
Below is a visualization of sentiment distribution within the dataset:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=twitter_data, x='target')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.show()
```

---

## Model Accuracy

| Dataset | Accuracy |
|---------|----------|
| Training | 79.8% |
| Test     | 77.8% |

---

## Usage

1. Clone the repository or run the notebook on **Google Colab**.  
2. Ensure the dataset is extracted into your working directory.  
3. Train the model by running:
   ```python
   model.fit(X_train, Y_train)
   ```
4. Test the model:
   ```python
   X_test_prediction = model.predict(X_test)
   accuracy = accuracy_score(Y_test, X_test_prediction)
   print(f'Accuracy Score on Test Data: {accuracy}')
   ```
5. Save the trained model:
   ```python
   import pickle

   with open('sentiment_model.pkl', 'wb') as file:
       pickle.dump(model, file)
   ```

---

## References

- [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- [Scikit-learn Documentation](https://scikit-learn.org/)  
- [NLTK Documentation](https://www.nltk.org/)

---

## License

This project is for educational purposes only. Please refer to the dataset’s license on Kaggle for more information.

---
