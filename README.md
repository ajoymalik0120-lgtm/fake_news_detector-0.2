# Fake News Detector v0.2

A machine learning project designed to classify news articles as fake or real.

## Project Structure
* `prepare_data.py`: Preprocesses the raw text data.
* `train_model.py`: Trains the TF-IDF Vectorizer and model classifier.
* `app.py`: Main application script to run predictions.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ajoymalik0120-lgtm/fake_news_detector-0.2.git

   cd fake_news_detector-0.2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data & Training:**
   * Place your `fake.csv` and `real.csv` data inside a `data/` directory.
   * Run `python train_model.py` to generate the `.pkl` model files.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
