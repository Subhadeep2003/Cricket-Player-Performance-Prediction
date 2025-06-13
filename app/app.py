import streamlit as st
import numpy as np
import pandas as pd
import joblib
# Load model and encoders
model = joblib.load("D:/Performance_Predictor/models_hist/kohli_model.pkl")
le_opponent = joblib.load("D:/Performance_Predictor/models_hist/le_opponent.pkl")
le_stadium = joblib.load("D:/Performance_Predictor/models_hist/le_stadium.pkl")

# Load raw dataset to validate input combinations
raw_df = pd.read_csv("D:/Performance_Predictor/data/Virat_kohli_DataSet_Final.csv")
raw_df.rename(columns={'Oponentes': 'Opponent', 'Stadiam': 'Stadium'}, inplace=True)
# Get class labels
opponents = list(le_opponent.classes_)
stadiums = list(le_stadium.classes_)

# Page configuration
st.set_page_config(page_title="Kohli Match Predictor", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #cc3300;'>Player Performance Predictor</h1>
    <h4 style='text-align: center; color: #555;'>Predict Kohli’s performance based on match conditions</h4>
    <hr style='border: 1px solid #cc3300;'>
""", unsafe_allow_html=True)

# Inputs
opp_input = st.selectbox("🔻 Select Opponent Team", opponents)
venue_input = st.selectbox("🏟️ Select Stadium", stadiums)
match_type = st.radio("📍 Match Type", ['Home', 'Away', 'Neutral'], horizontal=True)
innings_input = st.radio("🕒 Innings", [1, 2], horizontal=True)

# Year is fixed
year_input = 2024

# Encoding inputs
opp_encoded = le_opponent.transform([opp_input])[0]
venue_encoded = le_stadium.transform([venue_input])[0]
is_home_match = 1 if match_type == 'Home' else 0 if match_type == 'Away' else 2

# Check data existence for warning
exists = ((raw_df['Opponent'] == opp_input) & (raw_df['Stadium'] == venue_input)).any()
if not exists:
    st.warning(f"⚠️ Kohli may not have played against **{opp_input}** at **{venue_input}**. Prediction may be less accurate.")

# Prepare input and predict
X_input = np.array([[opp_encoded, venue_encoded,year_input, is_home_match, innings_input]])
pred = model.predict(X_input)[0]
# Button to trigger prediction
if st.button("Predict Performance"):
    # Prepare input
    X_input = np.array([[opp_encoded, venue_encoded, year_input, is_home_match, innings_input]])
    pred = model.predict(X_input)[0]

    # Styling for metric cards
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f9f9f9;
        border: 1px solid #cc3300;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] > label {
        font-size: 16px;
        color: #cc3300;
    }
    </style>
    """, unsafe_allow_html=True)

    # Show metrics
    st.markdown("<h3 style='color: #007acc;'>📊 Predicted Kohli Match Stats</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    r = (pred[0] / pred[1]) * 100  # Runs, Balls, Strike Rate
    col1.metric("Runs", f"{pred[0]:.0f}")
    col2.metric("Balls", f"{pred[1]:.0f}")
    col3.metric("Strike Rate", f"{r:.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("4s", f"{pred[3]:.0f}")
    col5.metric("6s", f"{pred[4]:.0f}")
    col6.metric("Fantasy Score", f"{pred[5]:.1f}")

    # Milestone Chances
    fifty_chance = pred[6]
    hundred_chance = pred[7]

    st.markdown("<h3 style='color: #ff6600;'>🎯 Milestone Chances</h3>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)

    if fifty_chance >= 0.5:
        col7.success(f"50+ Chance: {fifty_chance * 100:.1f}%")
    else:
        col7.warning(f"50+ Chance: {fifty_chance * 100:.1f}%")

    if hundred_chance >= 0.5:
        col8.success(f"100+ Chance: {hundred_chance * 100:.1f}%")
    else:
        col8.warning(f"100+ Chance: {hundred_chance * 100:.1f}%")
# Footer
st.markdown("""
<hr>
<p style='text-align: center; color: gray;'>Made with ❤️ by Subhadeep Mukherjee | Subham Paul | Santu Kapri</p>
""", unsafe_allow_html=True)

