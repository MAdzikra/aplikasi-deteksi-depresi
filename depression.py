import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Enhanced Page Configuration
st.set_page_config(
    page_title="Depression Detection App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
<style>
    /* Custom color palette */
    :root {
        --primary-color: #3498db;
        --secondary-color: #2ecc71;
        --background-color: #f4f6f7;
        --text-color: #2c3e50;
    }

    /* Main container styling */
    .main-container {
        background-color: var(--background-color);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Header styling */
    .title {
        color: var(--primary-color);
        text-align: center;
        font-weight: bold;
    }

    /* Form styling */
    .stRadio > div {
        background-color: #172D43;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Result styling */
    .result-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Data Loading and Model Training
@st.cache_data
def load_data():
    df = pd.read_csv('Deepression.csv')
    return df

df = load_data()

# Model Preparation
label_encoder = LabelEncoder()
df['Depression State'] = label_encoder.fit_transform(df['Depression State'])

X = df.drop(columns=["Number", "Depression State"])
y = df["Depression State"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2, weights='distance')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">🧠 Depression Detection App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Depression Detection Application
    
    This app helps you assess your mental health status using a machine learning model.
    
    #### Team Members:
    - **140810220044** - Candra Wibawa
    - **140810220046** - Muhammad Adzikra Dhiya Alfauzan
    - **140810220052** - Ivan Arsy Himawan
    """)


    depression_levels = {
        "No depression": "No signs of depression. Your mental health appears to be stable.",
        "Mild": "Mild depression symptoms. Pay attention to your mental well-being.",
        "Moderate": "Moderate depression symptoms. Consider consulting a mental health professional.",
        "Severe": "Severe depression. Immediate professional help is recommended."
    }

    with st.form("depression_form"):
        st.subheader("Mental Health Questionnaire")
        st.markdown("""
        :warning: Please answer all questions honestly. 
        Initially, no options are selected. Choose the option that best describes your experience in the past two weeks.
        """)
        
        # Question placeholders with columns for better layout
        cols = st.columns(2)
        questions = [
            "Sleep Quality", "Appetite", "Interest in Activities", 
            "Fatigue Levels", "Self-Worth", "Concentration",
            "Irritability", "Self-Harm Thoughts", "Sleep Disturbances", 
            "Aggression", "Panic Attacks", "Hopelessness",
            "Restlessness", "Energy Levels"
        ]
        
        inputs = {}
        for i, question in enumerate(questions):
            with cols[i % 2]:
                inputs[question.lower().replace(" ", "_")] = st.radio(
                    f"How often do you experience {question.lower()}?", 
                    ["", "Never", "Rarely", "Sometimes", "Often", "Always"],
                    index=0  # This sets the initial selection to the first (empty) option
                )

        submitted = st.form_submit_button("Assess My Mental Health")

    # Prediction and Result Display
    if submitted:
        # Check if all questions have been answered
        if all(inputs.values()):
            mapping = {"": 0, "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
            input_data = [[mapping[inputs[key]] for key in inputs.keys()]]
            
            prediction = knn.predict(input_data)
            depression_state = label_encoder.inverse_transform(prediction)[0]
            level_description = depression_levels.get(depression_state, "Description not available.")

            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.subheader("Assessment Results")
            st.markdown(f"### 📊 Depression Level: **{depression_state}**")
            st.write(f"**Description:** {level_description}")
            
            st.warning("🚨 If you need help, please consult a mental health professional.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Please answer ALL questions before submitting.")

    # Depression Levels Information
    st.subheader("Depression Level Details")
    depression_info = {
        "Depression Level": ["No depression", "Mild", "Moderate", "Severe"],
        "Description": [
            "No depression symptoms. Mental health is stable.",
            "Mild depression signs. Slight impact on daily life.",
            "Moderate depression affecting daily functioning.",
            "Severe depression significantly disrupting life."
        ],
        "Recommendations": [
            "Maintain healthy lifestyle and positive activities",
            "Increase positive activities and social support",
            "Consult a mental health professional",
            "Immediate professional intervention required"
        ]
    }

    df_info = pd.DataFrame(depression_info)
    st.table(df_info)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()