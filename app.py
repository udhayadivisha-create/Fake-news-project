import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Predict whether a news article is REAL or FAKE")

# --------------------------------
# Load Dataset
# --------------------------------
df = pd.read_csv("evaluation.csv", sep=';', on_bad_lines='skip')

# --------------------------------
# Combine Title + Text
# --------------------------------
df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

# --------------------------------
# Features and Target
# --------------------------------
X = df['content']
y = df['label']

# --------------------------------
# Convert Text into Numerical Data
# --------------------------------
vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# --------------------------------
# Train Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Train Model
# --------------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# --------------------------------
# Prediction on Test Data
# --------------------------------
y_pred = model.predict(X_test)

# --------------------------------
# Accuracy
# --------------------------------
acc = accuracy_score(y_test, y_pred)

# --------------------------------
# Display Accuracy
# --------------------------------
st.subheader("📊 Model Accuracy")
st.success(f"Accuracy: {acc * 100:.2f}%")

# --------------------------------
# User Input
# --------------------------------
st.subheader("📝 Predict News")

title_input = st.text_input("Enter News Title")

text_input = st.text_area("Enter News Text")

# --------------------------------
# Prediction Button
# --------------------------------
if st.button("Predict"):

    news = title_input + " " + text_input

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.error("🚨 Fake News Detected")
    else:
        st.success("✅ Real News")

# --------------------------------
# Show Metrics
# --------------------------------
st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# --------------------------------
# Dataset Preview
# --------------------------------
st.subheader("📂 Dataset Preview")

st.dataframe(df[['title', 'text', 'label']].head())