import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded locally
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_assets():
    try:
        with open('fake_news_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except Exception as e:
        st.error(f" Error loading model assets: {e}")
        return None, None

model, vectorizer = load_assets()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

# Streamlit Page Layout Config
st.set_page_config(page_title="Fake News Detector Engine", layout="centered")

st.title(" Fake News Detection System")
st.markdown("---")
st.write("Enter a news headline below to evaluate if it is **Real** or **Fabricated (Fake)**.")

# Input text box
user_input = st.text_area("Paste News Headline Here:", height=100, placeholder="Type text...")

if st.button("Analyze Headline", type="primary"):
    if user_input.strip() == "":
        st.warning(" Please enter some text before analyzing.")
    elif model is None or vectorizer is None:
        st.error(" Model assets are missing. Please re-run train_model.py first.")
    else:
        try:
            # 1. Preprocess the incoming user text string
            cleaned_input = clean_text(user_input)
            
            # 2. Vectorize the string into numerical arrays
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # 3. Classify using the pre-trained model
            prediction_array = model.predict(vectorized_input)
            
            # Extract prediction flag
            final_result = str(prediction_array[0]).upper().strip()
            
            st.markdown("### **Prediction Result:**")
            
            # 4. Display clean color banners based on exact output values
            if final_result == "REAL":
                st.success(" **This news appears to be REAL.**")
            elif final_result == "FAKE":
                st.error(" **Warning: This news appears to be FAKE / FABRICATED!**")
            else:
                st.warning(f" Unrecognized classification type: {final_result}")
                
        except Exception as e:
            st.error(f" An error occurred during calculations: {e}")

st.markdown("---")
st.caption("Academic Project Submission Evaluation Environment")
