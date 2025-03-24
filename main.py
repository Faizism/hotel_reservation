import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
import category_encoders as ce

# Page Configuration
st.set_page_config(
    page_title="Hotel Reservation Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    body { background-color: #F0F8FF; }
    .main-header { font-size: 2.8rem; color: #4CAF50; text-align: center; margin: 2rem 0; }
    .sub-header { font-size: 1.5rem; color: #333333; text-align: center; margin-bottom: 1.5rem; }
    .feature-box {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem auto;
        width: 80%;
    }
    .cta-button {
        display: block;
        width: fit-content;
        margin: 2rem auto;
        padding: 1rem 2rem;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        border-radius: 25px;
        text-decoration: none;
        transition: background-color 0.3s ease;
    }
    .cta-button:hover { background-color: #45A049; }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">🏨 Hotel Reservation Cancellation Predictor 🏨</h1>', unsafe_allow_html=True)

# Model Loading
try:
    model = pickle.load(open('GBC-final-clfCalibrated-0800.pkl', 'rb'))
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'GBC-final-clfCalibrated-0800.pkl' is in the correct directory.")
    model = None

# Debug mode
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Input Section
st.markdown('<div class="feature-box">', unsafe_allow_html=True)
st.subheader("📝 Enter Hotel Reservation Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Stay & Guest Information")
    lead_time = st.number_input('Lead Time (Days)', min_value=0, max_value=500, step=1)
    stays_in_weekend_nights = st.number_input('Weekend Nights', min_value=0, max_value=20, step=1)
    stays_in_week_nights = st.number_input('Weekday Nights', min_value=0, max_value=50, step=1)
    adults = st.number_input('Adults', min_value=0, max_value=10, step=1)
    children = st.number_input('Children', min_value=0, max_value=10, step=1, value=0)
    babies = st.number_input('Babies', min_value=0, max_value=10, step=1, value=0)
    total_guests = adults + children + babies
    total_stay_nights = stays_in_weekend_nights + stays_in_week_nights
    
    # Adding the missing features
    st.markdown("#### Guest History")
    is_repeated_guest = st.selectbox('Repeated Guest', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    previous_cancellations = st.number_input('Previous Cancellations', min_value=0, max_value=50, step=1)
    previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, max_value=50, step=1)

with col2:
    st.markdown("#### Booking Details")
    adr = st.number_input('Average Daily Rate (ADR)', min_value=0.0, max_value=1000.0, step=10.0)
    required_car_parking_spaces = st.number_input('Required Parking Spaces', min_value=0, max_value=5, step=1)
    total_of_special_requests = st.number_input('Special Requests', min_value=0, max_value=10, step=1)
    booking_changes = st.number_input('Booking Changes', min_value=0, max_value=50, step=1)
    days_in_waiting_list = st.number_input('Days in Waiting List', min_value=0, max_value=200, step=1)

# Categorical features
col5, col6 = st.columns(2)
with col5:
    hotel = st.selectbox('Hotel Type', ['Resort Hotel', 'City Hotel'])
    meal = st.selectbox('Meal Plan', ['BB', 'FB', 'HB', 'SC', 'Undefined'])
    market_segment = st.selectbox('Market Segment', ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Aviation', 'Groups'])
    distribution_channel = st.selectbox('Distribution Channel', ['Direct', 'Corporate', 'TA/TO', 'GDS', 'Undefined'])

with col6:
    reserved_room_type = st.selectbox('Reserved Room Type', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    deposit_type = st.selectbox('Deposit Type', ['No Deposit', 'Refundable', 'Non Refund'])
    customer_type = st.selectbox('Customer Type', ['Transient', 'Contract', 'Transient-Party', 'Group'])
    country = st.text_input('Country Code', 'USA')

st.markdown('</div>', unsafe_allow_html=True)

# Call to Action Button
if st.button("🔍 Predict Cancellation"):
    if model is not None:
        input_data = pd.DataFrame([{  
            'lead_time': lead_time, 
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights, 
            'adults': adults,
            'children': children, 
            'babies': babies, 
            'total_guests': total_guests,
            'total_stay_nights': total_stay_nights, 
            'adr': adr,
            'required_car_parking_spaces': required_car_parking_spaces,
            'total_of_special_requests': total_of_special_requests,
            'booking_changes': booking_changes, 
            'days_in_waiting_list': days_in_waiting_list,
            'hotel': hotel, 
            'meal': meal, 
            'market_segment': market_segment,
            'distribution_channel': distribution_channel, 
            'reserved_room_type': reserved_room_type,
            'deposit_type': deposit_type, 
            'customer_type': customer_type, 
            'country': country,
            # Adding the missing columns that caused the error
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'is_repeated_guest': is_repeated_guest,
            'previous_cancellations': previous_cancellations
        }])
        
        if debug_mode:
            st.write("Input data for prediction:")
            st.write(input_data)

        try:
            prediction = model.predict(input_data)
            result_text = "✅ Booking Confirmed!" if prediction[0] == 0 else "❌ High Chance of Cancellation!"
            color = "green" if prediction[0] == 0 else "red"
            st.markdown(f'<h3 style="color:{color}; text-align:center;">{result_text}</h3>', unsafe_allow_html=True)
            
            # If model can provide prediction probabilities
            try:
                proba = model.predict_proba(input_data)[0]
                cancellation_probability = proba[1] * 100 if prediction[0] == 1 else proba[0] * 100
                st.markdown(f'<p style="text-align:center;">Confidence: {cancellation_probability:.1f}%</p>', unsafe_allow_html=True)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if debug_mode:
                st.write("Error details:", e)