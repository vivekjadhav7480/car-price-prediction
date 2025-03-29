import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Load the model
model = pk.load(open(r"C:\Users\VIVEK JADHAV\model.pkl", 'rb'))

# Page title and description
st.title('🚗 Car Price Prediction ML Model')
st.markdown("""
Welcome to the **Car Price Prediction App**!  
Fill in the details below to predict the price of your car.
""")

# Load and preprocess data
cars_data = pd.read_csv('C:\\Users\\VIVEK JADHAV\\Videos\\CAR PRICE PREDICTION PROJECT\\Cardetails.csv')

def get_brand_name(car_name):  
    car_name = car_name.split(' ')[0]
    return car_name.strip()
cars_data['name'] = cars_data['name'].apply(lambda x: get_brand_name(str(x)))

# Create input fields in a structured layout
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    name = st.selectbox('Select Car Brand', cars_data['name'].unique())
    year = st.slider('Car Manufactured Year', 1994, 2024)
    km_driven = st.number_input('No. of kms Driven', min_value=11, max_value=200000, step=100)
    fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())

with col2:
    transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
    owner = st.selectbox('Ownership Type', cars_data['owner'].unique())
    mileage = st.slider('Car Mileage (km/l)', 10, 40)
    engine = st.slider('Engine Capacity (cc)', 700, 5000)
    max_power = st.slider('Max Power (bhp)', 0, 200)
    seats = st.slider('Number of Seats', 5, 10)

# Predict button
if st.button("🔍 Predict Price"):
    # Prepare input data
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Apply transformations to input_data_model
    input_data_model['owner'] = input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5]
    )
    input_data_model['fuel'] = input_data_model['fuel'].replace(
        ['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4]
    )
    input_data_model['seller_type'] = input_data_model['seller_type'].replace(
        ['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3]
    )
    input_data_model['transmission'] = input_data_model['transmission'].replace(
        {'Manual': 1, 'Automatic': 2}
    )
    input_data_model['name'] = input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    )

    # Debugging: Display input data
    st.write("Input Data for Prediction (Debugging):")
    

    try:
        # Predict car price
        car_price = model.predict(input_data_model)

        # Ensure the predicted price is not negative
        car_price = max(0, car_price[0])

        # Handle zero predictions
        if car_price == 0:
            st.warning("⚠️ The predicted price is very low. Please check the input values or retrain the model.")
        else:
            st.success(f"💰 The predicted price of the car is **₹{car_price:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")








