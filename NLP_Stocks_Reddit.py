import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation as LDA
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure you have the NLTK data needed
nltk.download('wordnet')
nltk.download('stopwords')

#%% Step 1: Data Loading
combined_news_djia = pd.read_csv(r'C:\Users\alfredo.serrano.fig1\Desktop\Personal\Github\Stock_Price\Combined_News_DJIA.csv')
reddit_news = pd.read_csv(r'C:\Users\alfredo.serrano.fig1\Desktop\Personal\Github\Stock_Price\RedditNews.csv')
djia_table = pd.read_csv(r'C:\Users\alfredo.serrano.fig1\Desktop\Personal\Github\Stock_Price\upload_DJIA_table.csv')
# Step 2: Data Preprocessing and Feature Engineering
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep only alphabets
    text = text.lower().split()  # Lowercase and split into words
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(text)

# Apply preprocessing to each headline
for i in range(1, 26):
    combined_news_djia[f'Top{i}'] = combined_news_djia[f'Top{i}'].apply(preprocess_text)

# Combine all headlines into a single text feature
combined_news_djia['Combined_News'] = combined_news_djia.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Step 3: Feature Engineering with TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(combined_news_djia['Combined_News']).toarray()
y = combined_news_djia['Label'].values

# Step 4: Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Step 5: Topic Modeling with LDA
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(combined_news_djia['Combined_News'])

lda = LDA(n_components=5, n_jobs=-1, random_state=42)
lda.fit(count_data)

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx+1}:")
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("Topics found via LDA:")
print_topics(lda, count_vectorizer, n_top_words=10)