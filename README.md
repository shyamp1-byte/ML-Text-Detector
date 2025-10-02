# SMS Spam Detector (Streamlit)

A small, beginner-friendly app that predicts whether a text message is **Spam** or **Not Spam** using scikit-learn (CountVectorizer + MultinomialNB). Type a message and see the result right away.

## Quick start
1. Install requirements  
   `python3 -m pip install -r requirements.txt`
2. Run the app  
   `streamlit run SpamDetection.py`

## How it works
- Loads an SMS spam dataset (`spam.csv`)
- Splits data into train/test
- Turns text into features with `CountVectorizer(stop_words="english")`
- Trains a `MultinomialNB` model
- Simple Streamlit UI for live predictions

## Features
- Real-time message input and prediction
- Lightweight model that runs fast on a laptop
- Clean, minimal interface

## Notes
- Dataset is included as `spam.csv`
- This project is meant for learning and a quick demo, not production

## Future ideas
- Show accuracy/F1 and a confusion matrix on a “Metrics” tab
- Add confidence scores (`predict_proba`)
- Try TF-IDF + Logistic Regression and compare
- Deploy to Streamlit Community Cloud and link it here

## Technologies Used
Python · pandas · scikit-learn · Streamlit
