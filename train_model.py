import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # 🔄 Updated Import for Modern Compliance

# Download text preprocessing assets safely
try:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
except:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    # Strip non-alphabetic characters and lowercase the tokens
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return " ".join([w for w in text.split() if w not in stop_words])

print("📦 Reading raw datasets from 'data' folder...")
fake_path = os.path.join('data', 'fake.csv')
true_path = os.path.join('data', 'real.csv')

if not os.path.exists(fake_path) or not os.path.exists(true_path):
    print("❌ Error: Could not find 'fake.csv' or 'real.csv' inside the 'data' folder!")
    exit()

# Load custom data dataframes
df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

# Assign structural text class indicators
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

# Clean structural boundaries
df_fake = df_fake[['title', 'label']].dropna()
df_true = df_true[['title', 'label']].dropna()

print(f"📊 Dataset Volumes Found: {len(df_fake)} Fake, {len(df_true)} Real.")

# Merge and shuffle the dataset matrix completely
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("🧹 Preprocessing and cleaning text tokens...")
df['clean_title'] = df['title'].apply(clean_text)

# Split into training and testing evaluation structures
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_title'], df['label'], test_size=0.2, random_state=42
)

print("🧪 Vectorizing text data (TF-IDF)...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 🤖 OPTIMIZED: Using SGDClassifier configured with Hinge loss and 'pa1' learning rate
print("🤖 Training modern Stochastic Gradient Descent (SGD) Classifier...")
model = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=50, random_state=42)
model.fit(X_train_tfidf, y_train)

# Calculate testing analytics
accuracy = model.score(X_test_tfidf, y_test) * 100
print(f"📈 Evaluation Test Accuracy: {accuracy:.2f}%")

# Save compiled models for backend storage
with open('fake_news_model.pkl', 'wb') as f: 
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f: 
    pickle.dump(tfidf_vectorizer, f)
    
print("✅ Done! Warning-free modern model assets successfully generated.")
