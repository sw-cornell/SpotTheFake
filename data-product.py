import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def initialize_session_state():
    """Initialize session state variables"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

def preprocess_text(text):
    """Function to preprocess text data"""
    # Text standardization
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters

    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def load_model_parameters():
    """Load SVM parameters from JSON file"""
    try:
        with open('svm_params.json', 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        st.error(f"Error loading model parameters: {e}")
        st.stop()

def load_model_and_vectorizer():
    """Load the trained model and fitted vectorizer"""
    try:
        model = joblib.load('svm_model.joblib')
        params = load_model_parameters()

        try:
            vectorizer = joblib.load('tfidf_vectorizer.joblib')
            st.sidebar.success("✅ Vectorizer loaded successfully")
        except FileNotFoundError:
            st.error("TF-IDF vectorizer file not found. Please make sure 'tfidf_vectorizer.joblib' exists.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading vectorizer: {e}")
            st.stop()

        with st.sidebar:
            st.markdown("### Model Configuration")
            st.json(params)

        st.sidebar.success("✅ Model loaded successfully")
        return model, vectorizer

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def predict_fake_reviews(texts, model, vectorizer):
    """Predict fake reviews using the model with preprocessing"""
    try:
        # Preprocess each text in the input list
        processed_texts = [preprocess_text(text) for text in texts]

        # Transform the texts using the fitted vectorizer
        X = vectorizer.transform(processed_texts)

        # Get predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        if predictions is None or probabilities is None:
            st.error("Prediction failed")
            st.stop()

        return predictions, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()


def create_analysis_plots(df):
    """Create various analysis plots for the reviews data"""

    # Check for all-real or all-fake cases and adjust accordingly
    unique_predictions = df['prediction'].nunique()
    all_real_or_fake = unique_predictions == 1
    single_type_label = "All reviews classified as Real" if df['prediction'].iloc[0] == 1 else "All reviews classified as Fake"

    if all_real_or_fake:
        st.warning(f"{single_type_label}. Some plots are adjusted accordingly.")

    # 1. Rating Distribution Plot
    if all_real_or_fake:
        fig_rating = px.histogram(
            df,
            x='rating',
            title=f'Rating Distribution: {single_type_label}',
            labels={'rating': 'Rating'},
            color_discrete_sequence=['#00CC96'] if df['prediction'].iloc[0] == 1 else ['#EF553B']
        )
    else:
        fig_rating = px.histogram(
            df,
            x='rating',
            color='prediction',
            barmode='group',
            title='Rating Distribution: Real vs Fake Reviews',
            labels={'prediction': 'Review Type', 'rating': 'Rating'},
            color_discrete_map={1: '#00CC96', 0: '#EF553B'}
        )
    st.plotly_chart(fig_rating, use_container_width=True)

    # 2. Review Timeline Plot
    df['month_year'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m')
    timeline_data = df.groupby(['month_year', 'prediction']).size().reset_index(name='count')
    timeline_data = timeline_data.pivot(index='month_year', columns='prediction', values='count').fillna(0)

    # Sort the index to ensure chronological order
    timeline_data = timeline_data.sort_index()

    # Filter out future dates
    current_date = datetime.now()
    timeline_data = timeline_data[timeline_data.index <= current_date.strftime('%Y-%m')]

    fig_timeline = go.Figure()

    # Add traces based on available prediction types
    if all_real_or_fake:
        prediction_value = df['prediction'].iloc[0]
        color = '#00CC96' if prediction_value == 1 else '#EF553B'
        name = 'Real Reviews' if prediction_value == 1 else 'Fake Reviews'

        fig_timeline.add_trace(go.Scatter(
            x=timeline_data.index,
            y=timeline_data[prediction_value],
            name=name,
            line=dict(color=color)
        ))
    else:
        if 1 in timeline_data.columns:
            fig_timeline.add_trace(go.Scatter(
                x=timeline_data.index,
                y=timeline_data[1],
                name='Real Reviews',
                line=dict(color='#00CC96')
            ))
        if 0 in timeline_data.columns:
            fig_timeline.add_trace(go.Scatter(
                x=timeline_data.index,
                y=timeline_data[0],
                name='Fake Reviews',
                line=dict(color='#EF553B')
            ))

    fig_timeline.update_layout(
        title='Review Timeline: Real vs Fake Reviews' if not all_real_or_fake else f'Review Timeline: {single_type_label}',
        xaxis_title='Date',
        yaxis_title='Number of Reviews'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # 3. Confidence Distribution Plot
    fig_conf = px.histogram(
        df,
        x='real_probability',
        nbins=20,
        title='Model Confidence Distribution',
        labels={'real_probability': 'Probability of Being Real'},
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # 4. Review Length Analysis
    df['review_length'] = df['text'].str.len()
    if all_real_or_fake:
        fig_length = px.box(
            df,
            y='review_length',
            title=f'Review Length Distribution for {single_type_label}',
            labels={'review_length': 'Review Length (characters)'},
            color_discrete_sequence=['#00CC96'] if df['prediction'].iloc[0] == 1 else ['#EF553B']
        )
    else:
        fig_length = px.box(
            df,
            x='prediction',
            y='review_length',
            color='prediction',
            title='Review Length Distribution by Type',
            labels={'prediction': 'Review Type', 'review_length': 'Review Length (characters)'},
            color_discrete_map={1: '#00CC96', 0: '#EF553B'}
        )
    st.plotly_chart(fig_length, use_container_width=True)

    # 5. Summary Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    if not all_real_or_fake or (all_real_or_fake and df['prediction'].iloc[0] == 1):
        avg_real_rating = df[df['prediction'] == 1]['rating'].mean()
        col1.metric("Avg Real Review Rating", f"{avg_real_rating:.1f}⭐")
        avg_real_length = df[df['prediction'] == 1]['review_length'].mean()
        col3.metric("Avg Real Review Length", f"{avg_real_length:.0f}")
    else:
        col1.metric("Avg Real Review Rating", "N/A")
        col3.metric("Avg Real Review Length", "N/A")

    if not all_real_or_fake or (all_real_or_fake and df['prediction'].iloc[0] == 0):
        avg_fake_rating = df[df['prediction'] == 0]['rating'].mean()
        col2.metric("Avg Fake Review Rating", f"{avg_fake_rating:.1f}⭐")
        avg_fake_length = df[df['prediction'] == 0]['review_length'].mean()
        col4.metric("Avg Fake Review Length", f"{avg_fake_length:.0f}")
    else:
        col2.metric("Avg Fake Review Rating", "N/A")
        col4.metric("Avg Fake Review Length", "N/A")


class GooglePlacesReviewer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    def find_place(self, query):
        endpoint = f"{self.base_url}/findplacefromtext/json"
        params = {
            'input': query,
            'inputtype': 'textquery',
            'fields': 'place_id,name,formatted_address',
            'key': self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'OK' and data['candidates']:
                st.sidebar.success(f"Found {len(data['candidates'])} places")
                return data['candidates']
            else:
                st.error(f"No places found: {data['status']}")
                return []

        except requests.exceptions.RequestException as e:
            st.error(f"Error searching for place: {e}")
            return []

    def get_place_details(self, place_id):
        endpoint = f"{self.base_url}/details/json"
        params = {
            'place_id': place_id,
            'fields': 'name,rating,reviews,formatted_address,user_ratings_total',
            'language': 'en',
            'key': self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'OK':
                st.sidebar.success(f"Successfully fetched place details")
                return data['result']
            else:
                st.error(f"Failed to get place details: {data['status']}")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Error getting place details: {e}")
            return None

    def process_reviews(self, reviews, place_name):
        if not reviews:
            st.warning("No reviews available for processing")
            return pd.DataFrame(columns=[
                'place_name', 'author', 'rating', 'text',
                'time', 'language', 'likes'
            ])

        processed_reviews = []
        for review in reviews:
            try:
                processed_review = {
                    'place_name': place_name,
                    'author': review.get('author_name', ''),
                    'rating': review.get('rating', 0),
                    'text': review.get('text', ''),
                    'time': datetime.fromtimestamp(review.get('time', 0)),
                    'language': review.get('language', ''),
                    'likes': review.get('rating', 0)
                }
                processed_reviews.append(processed_review)
            except Exception as e:
                st.warning(f"Skipped processing review due to error: {str(e)}")
                continue

        return pd.DataFrame(processed_reviews)

    def save_to_csv(self, df, filename):
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            st.success(f"Reviews saved to: {filename}")
            return filename
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None

def main():
    st.title("Fake Review Detector")
    st.write("Analyze Google Places reviews and detect potential fake reviews using machine learning.")

    # Initialize session state
    initialize_session_state()

    # API Key input section
    with st.sidebar:
        st.header("API Configuration")

        # Create a text input for the API key
        api_key = st.text_input(
            "Enter Google Places API Key:",
            type="password",  # Masks the API key
            help="Enter your Google Places API key. This will not be stored permanently.",
            value=st.session_state.api_key if st.session_state.api_key else ""
        )

        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API Key provided")
        else:
            st.error("Please enter your Google Places API Key to continue")
            st.stop()

        st.markdown("---")

        st.header("About")
        st.markdown("""
        This app uses:
        - Google Places API to fetch reviews
        - Machine learning model to detect fake reviews
        """)

    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Input box for direct review analysis
    review_text = st.text_input("Enter a review text to analyze directly:")

    if review_text:
        # Preprocess and analyze the review
        predictions, probabilities = predict_fake_reviews([review_text], model, vectorizer)

        # Display prediction and probability
        if predictions[0] == 1:
            st.write("**Prediction:** ✅ Likely Real Review")
        else:
            st.write("**Prediction:** ⚠️ Potential Fake Review")

        st.write(f"**Probability of being real:** {probabilities[0][1]:.2%}")

    # Divider line for Google API functionality
    st.markdown("---")

    reviewer = GooglePlacesReviewer(st.session_state.api_key)

    search_query = st.text_input("Enter place name to search:")

    if search_query:
        with st.spinner("Searching for places..."):
            places = reviewer.find_place(search_query)

        if not places:
            st.error("No places found")
            st.stop()

        place_options = {
            f"{place.get('name', '')} - {place.get('formatted_address', '')}": place
            for place in places
        }

        with st.sidebar:
            st.markdown("### Found Places")
            for place in places:
                st.text(f"ID: {place.get('place_id')}")
                st.text(f"Name: {place.get('name')}")
                st.text("---")

        selected_place_name = st.selectbox(
            "Select a place:",
            options=list(place_options.keys())
        )

        if st.button("Analyze Reviews"):
            selected_place = place_options[selected_place_name]

            with st.spinner("Fetching and analyzing reviews..."):
                place_details = reviewer.get_place_details(selected_place.get('place_id'))

                if not place_details:
                    st.error("Failed to get place details")
                    st.stop()

                reviews = place_details.get('reviews', [])
                if not reviews:
                    st.warning("No reviews found for this place")
                    st.stop()

                df = reviewer.process_reviews(reviews, place_details['name'])
                predictions, probabilities = predict_fake_reviews(df['text'].tolist(), model, vectorizer)

                df['prediction'] = predictions
                df['real_probability'] = probabilities[:, 1]

                # Display initial metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(reviews))
                with col2:
                    st.metric("Average Rating", round(place_details.get('rating', 0), 2))
                with col3:
                    fake_review_count = len(reviews) - sum(predictions)
                    st.metric("Potential Fake Reviews", fake_review_count)

                # Create Analysis Plots
                st.subheader("📊 Review Analysis Visualizations")
                create_analysis_plots(df)

                # Display individual reviews
                st.subheader("📝 Individual Review Analysis")
                df_sorted = df.sort_values('real_probability', ascending=False)

                filename = f"{place_details['name']}_reviews_{datetime.now().strftime('%Y%m%d')}.csv"
                reviewer.save_to_csv(df_sorted, filename)

                for _, review in df_sorted.iterrows():
                    with st.expander(f"Review by {review['author']} - {'⚠️ Potential Fake' if review['prediction'] == 0 else '✅ Likely Real'}"):
                        st.write(f"**Rating:** {'⭐' * int(review['rating'])}")
                        st.write(f"**Review:** {review['text']}")
                        st.write(f"**Date:** {review['time'].strftime('%Y-%m-%d')}")
                        st.progress(review['real_probability'])
                        st.write(f"Probability of being real: {review['real_probability']:.2%}")

                with open(filename, 'rb') as f:
                    st.download_button(
                        "Download Analysis CSV",
                        f,
                        filename,
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()
