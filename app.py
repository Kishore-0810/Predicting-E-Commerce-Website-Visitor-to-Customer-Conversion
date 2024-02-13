# Importing the Necessary Libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle


# reading the cleaned data
df = pd.read_csv("cleaned_classification_data.csv")


# function to predict customer convert or not convert
def predict(transaction_revenue, device_operating_system, time_on_site, session_quality, products_array):

    dos = {"(not set)":0, "Android": 1, "Linux": 2, "Chrome OS": 3, "Macintosh": 4, "Windows": 5, "iOS": 6}
    device_operating_system = dos[device_operating_system]

    product_list = df.columns[40:].to_list()
    for i in product_list:
        if i in products_array:
            product_list[product_list.index(i)] = 1
        else:
            product_list[product_list.index(i)] = 0

    features = [transaction_revenue, device_operating_system, time_on_site, session_quality] 
    
    features.extend(product_list)

    with open("standard_scaler.pkl", "rb") as file:
        ss = pickle.load(file)

    with open("logistic_reg_model.pkl", "rb") as file:
        model = pickle.load(file)

    features = ss.transform([features])

    prediction = model.predict(features)

    if prediction[0] == 1:
        return "Convert"
    
    if prediction[0] == 0:
        return "Not Convert"



# Streamlit Setup
st.set_page_config("Customer Conversion", layout = "wide")


selected = option_menu(None,
                       options = ["Menu", "Prediction"],
                       icons = ["house"],
                       orientation = "horizontal",
                       styles = {"nav-link": {"font-size": "18px", "text-align": "center", "margin": "1px"},
                                 "icon": {"color": "yellow", "font-size": "20px"},
                                 "nav-link-selected": {"background-color": "#9457eb"}} )


if selected == "Menu":
    
    st.title(''':red[**Predicting E-Commerce Website Visitor to Customer Conversion**]''')

    st.markdown("")

    st.markdown('''* In the rapidly evolving E-commerce industry, understanding customer behavior is crucial for business growth. 
                    The goal of this project is to analyze an E-commerce dataset and develop a predictive model that can accurately 
                    classify visitors based on their likelihood to convert into customers. The dataset contains a variety of features, 
                    including a target variable has_converted which indicates whether a visitor has converted into a customer.''')
    
    st.markdown('''* The challenge is to use this data to train a classification model that can predict the has_converted status for future visitors. 
                    The outcome of this project will provide valuable insights that can help in strategizing effective customer conversion techniques. ''')


if selected == "Prediction":

    with st.form("classification"):

        st.number_input(":blue[**Transaction Revenue**]", min_value = 0, key = "tr")

        st.number_input(":blue[**Time On Site**]", min_value = 0, key = "tos")

        st.number_input(":blue[**Session Quality**]", min_value = 0, max_value = 100, key = "sq")

        st.selectbox(":blue[**Device Operating System**]", 
                    options = ['Android', 'iOS', 'Macintosh', 'Windows', 'Chrome OS', '(not set)', 'Linux'],
                    key = "dos")

        st.multiselect(":blue[**Products Array**]", options = df.columns[40:], key = "pa")


        if st.form_submit_button("**Predict**"):

            pred = predict(st.session_state["tr"], st.session_state["dos"], st.session_state["tos"], 
                           st.session_state["sq"], st.session_state["pa"])
            
            st.success(f"**The Visitor is likely to :green[{pred}] as a Customer**")

    
# ------------------------------x--------------------------x----------------------------x--------------------------x------------------------------x--------------------------------------------------