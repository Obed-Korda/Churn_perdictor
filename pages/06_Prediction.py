# Standard library imports
import base64
import os
import sqlite3
import datetime

# Third-party imports
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Local application imports
from utils.login import invoke_login_widget
from utils.lottie import display_lottie_on_page


# Invoke the login form
invoke_login_widget('Future Projections')

# Fetch the authenticator from session state
authenticator = st.session_state.get('authenticator')

# Ensure the authenticator is available
if not authenticator:
    st.error("Authenticator not found. Please check the configuration.")
    st.stop()

# Check authentication status
if st.session_state.get("authentication_status"):
    username = st.session_state['username']
    st.title("Churn Prediction Dashboard")
    st.write("---")

    # Page Introduction
    with st.container():        
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("""
            Welcome to the Churn Prediction Dashboard.
            Utilize this dashboard to project customer churn and develop targeted retention strategies. 
            By uploading your customer data, you enable our models to assess the probability of churn. 
            You can perform single, bulk, or template predictions by selecting the appropriate option from the sidebar and adhering to the provided instructions. 
            This functionality will assist you in directing your efforts where they are most needed.
            """)
        with right_column:
            display_lottie_on_page("Future Projections")

    # Load the initial data from a local file
    @st.cache_data(persist=True, show_spinner=False)
    def load_initial_data():
        df = pd.read_csv('./data/LP2_train_pred.csv')
        return df
    
    initial_df = load_initial_data()

    # Ensure 'data_source' is initialized in session state
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = 'initial' 

    # Function to load the most recent table from the user's SQLite database
    def load_most_recent_table(username):
        # Define the path for the user's SQLite database
        db_path = os.path.join("data", username, f"{username}.db")

        if not os.path.exists(db_path):
            st.error("No database found for the user. Please ensure a file has been uploaded on the data overview page.")
            return None, None

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Get the most recent table name
        tables_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE ?
        ORDER BY name DESC LIMIT 1;
        """
        try:
            most_recent_table = conn.execute(tables_query, (f"{username}_table%",)).fetchone()

            if most_recent_table:
                table_name = most_recent_table[0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                st.session_state['data_source'] = 'uploaded'
            else:
                st.error("No tables found in the database.")
                return None, None
        except Exception as e:
            st.error(f"An error occurred while loading the table: {e}")
            return None, None
        finally:
            conn.close()

        return df, table_name

    # Load models
    @st.cache_resource(show_spinner=False)
    def models():
        rf_model = joblib.load('./models/RF.joblib')
        gb_model = joblib.load('./models/GB.joblib')
        xgb_model = joblib.load('./models/XB.joblib')
        return rf_model, gb_model, xgb_model

    RF, GB, XB = models()

    # Sidebar radio buttons for selecting prediction type
    prediction_type = st.sidebar.radio(
        "Choose Prediction Type:",
        ('Single Prediction', 'Bulk Prediction', 'Template Prediction')
    )

    # Select model 
    selected_model = st.selectbox('Select a Model', ['', 'Random Forest', 'GBoost', 'XGBoost'], 
                                    key='selected_model',
                                    index=0)

    # Function to get the selected model
    @st.cache_resource(show_spinner='Loading models...')
    def get_model(selected_model):
        if selected_model == '':
            st.warning('Please select a model before making a prediction.')
            return None, None
        elif selected_model == 'Random Forest':
            pipeline = RF
        elif selected_model == 'GBoost':
            pipeline = GB
        else:
            pipeline = XB
        encoder = joblib.load('./models/encoder.joblib')
        return pipeline, encoder
    
    # Initialize session states for relevant variables
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None
    if 'probability' not in st.session_state:
        st.session_state['probability'] = None
    if 'customer_id' not in st.session_state:
        st.session_state['customer_id'] = None

    # Function to make a single prediction
    def make_single_prediction(pipeline, encoder):
        if pipeline is None:
            return  

        else:
        # Collect user input from session state
            user_input = {
                'customerID': st.session_state['customer_id'],
                'gender': st.session_state['gender'],
                'SeniorCitizen': st.session_state['senior_citizen'],
                'Partner': st.session_state['partner'],
                'Dependents': st.session_state['dependents'],
                'tenure': st.session_state['tenure'],
                'PhoneService': st.session_state['phone_service'],
                'MultipleLines': st.session_state['multiple_lines'],
                'InternetService': st.session_state['internet_service'],
                'OnlineSecurity': st.session_state['online_security'],
                'OnlineBackup': st.session_state['online_backup'],
                'DeviceProtection': st.session_state['device_protection'],
                'TechSupport': st.session_state['tech_support'],
                'StreamingTV': st.session_state['streaming_tv'],
                'StreamingMovies': st.session_state['streaming_movies'],
                'Contract': st.session_state['contract'],
                'PaperlessBilling': st.session_state['paperless_billing'],
                'PaymentMethod': st.session_state['payment_method'],
                'MonthlyCharges': st.session_state['monthly_charges'],
                'TotalCharges': st.session_state['total_charges'],
            }

            # Convert the input data to a DataFrame
            df = pd.DataFrame(user_input, index=[0])

            # Set 'customerID' as the index
            if 'customerID' in df.columns:
                df.set_index('customerID', inplace=True)
            else:
                st.error("Column 'customerID' not found in the dataset.")

            # Ensure numerical columns are correctly typed
            df = df.apply(pd.to_numeric, errors='ignore')

            # Ensure there are no zero or missing values to avoid division by zero
            df['MonthlyCharges'].replace(0, 1, inplace=True)
            df['TotalCharges'].replace(0, 1, inplace=True)
            df['tenure'].replace(0, 1, inplace=True)

            # Create the 'AvgMonthlyCharges' feature (TotalCharges/tenure)
            df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure']

            # Create the 'MonthlyChargesToTotalChargesRatio' feature (MonthlyCharges/TotalCharges)
            df['MonthlyChargesToTotalChargesRatio'] = df['MonthlyCharges'] / df['TotalCharges']   

            # Make predictions
            pred = pipeline.predict(df) 
            pred_int = int(pred[0])   
            prediction = encoder.inverse_transform([[pred_int]])[0]

            # Calculate the probability of churn
            probability = pipeline.predict_proba(df)
            prediction_labels = "Churn" if pred == 1 else "No Churn"

            # Display prediction results
            customer_id = st.session_state['customer_id']

            # Update the session state with the prediction and probabilities
            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability
            st.session_state['prediction_labels'] = prediction_labels

            # Copy the original dataframe to the new dataframe
            hist_df = df.copy()
            hist_df['PredictionTime'] = datetime.date.today()
            hist_df['ModelUsed'] = st.session_state['selected_model']
            hist_df['Prediction'] = prediction
            hist_df['Predicted Churn'] = prediction_labels
            hist_df['Probability'] = np.where(pred == 1, np.round(probability[:, 1] * 100, 2), np.round(probability[:, 0] * 100, 2))

            # Save the history dataframe to SQLite database
            db_path = f"./data/{st.session_state['username']}/{st.session_state['username']}.db"
            conn = sqlite3.connect(db_path)
            hist_df.to_sql('single_predict', conn, if_exists='append', index=True)
            conn.close()

            return prediction, probability, prediction_labels

    # Function to get user input
    def get_user_input():
        pipeline, encoder = get_model(selected_model)

        if pipeline == None:
            return
        else:
            st.info('Please ensure all fields are properly filled.')
            with st.form('input-feature', clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text_input('Customer ID', key='customer_id')
                    st.write('### Subscriber Demographic')
                    st.selectbox('Gender', options=['Male', 'Female'], key='gender')
                    st.selectbox('Senior Citizen', options=['No', 'Yes'], key='senior_citizen')
                    st.selectbox('Dependents', options=['No', 'Yes'], key='dependents')
                    st.selectbox('Partner?', options=['Yes', 'No'], key='partner')
                
                with col2:
                    st.write('### Subscriber Account Details')
                    st.number_input('Key in tenure', min_value=0.00, max_value=200.00, step=0.10, key='tenure')
                    st.number_input('Key in monthly charges', min_value=0.00, max_value=500.00, step=0.10, key='monthly_charges')
                    st.number_input('Key in total charges per year', min_value=0.00, max_value=100000.00, step=0.10, key='total_charges')
                    st.selectbox('Select payment method', options=['Mailed check', 'Electronic Check', 'Bank Transfer (automatic)', 'Credit Card(automatic)'], key='payment_method')
                    st.selectbox('Select contract type', options=['Month-to-month', 'One year', 'Two years'], key='contract')
                    st.selectbox('Paperless Billing', options=['Yes', 'No'], key='paperless_billing')
                
                with col3:
                    st.write('### Subscriptions')
                    st.selectbox('Phone Service', options=['Yes', 'No'], key='phone_service')
                    st.selectbox('Multiple Lines', options=['Yes', 'No', 'No phone service'], key='multiple_lines')
                    st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'], key='internet_service')
                    st.selectbox('Online Security', options=['Yes', 'No', 'No internet service'], key='online_security')
                    st.selectbox('Online Backup', options=['Yes', 'No', 'No internet service'], key='online_backup')
                    st.selectbox('Device Protection', options=['Yes', 'No', 'No internet service'], key='device_protection')
                    st.selectbox('Tech Support', options=['Yes', 'No', 'No internet service'], key='tech_support')
                    st.selectbox('Streaming TV', options=['Yes', 'No', 'No internet service'], key='streaming_tv')
                    st.selectbox('Streaming movies', options=['Yes', 'No', 'No internet service'], key='streaming_movies')
                    st.form_submit_button('Make Prediction', on_click=make_single_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))
        
            st.info('Prediction results will be shown here.')

    # Function to make bulk predictions using uploaded dataset
    def make_bulk_prediction(pipeline, encoder, bulk_input_df):
        # Convert the input data to a DataFrame
        df = bulk_input_df.copy()

        if 'customerID' in df.columns:
            df.set_index('customerID', inplace=True)

        # Ensure numerical columns are correctly typed
        df = df.apply(pd.to_numeric, errors='ignore')

        # Define the list of specific columns to check and coerce
        columns_to_coerce = ['tenure', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'MonthlyChargesToTotalChargesRatio']

        try:
            # Ensure numerical columns are correctly typed for specific columns
            for column in columns_to_coerce:
                if column in df.columns and df[column].dtype == 'object':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
        except Exception as e:
            st.error(f"An error occurred while processing the column '{column}': {e}")
            st.warning(
                """
                Please refer to the Data Overview page to apply the correct data structure, 
                ensuring numerical columns have strictly numeric values and categorical columns 
                have strictly categorical values.
                """
            )

        # Check if 'AvgMonthlyCharges' column exists
        # if 'AvgMonthlyCharges' not in df.columns:
        #  
        #     # Ensure there are no zero or missing values to avoid division by zero
        #     df['TotalCharges'].replace(0, 1, inplace=True)
        #     df['Tenure'].replace(0, 1, inplace=True)
            
        #     # Create the 'AvgMonthlyCharges' feature (TotalCharges/tenure)
        #     df['AvgMonthlyCharges'] = df['TotalCharges'] / df['Tenure']

        # # Check if 'MonthlyChargesToTotalChargesRatio' column exists
        # if 'MonthlyChargesToTotalChargesRatio' not in df.columns:
            
        #     # Ensure there are no zero or missing values to avoid division by zero
        #     df['TotalCharges'].replace(0, 1, inplace=True)
        #     df['MonthlyCharges'].replace(0, 1, inplace=True)
            
        #     # Create the 'MonthlyChargesToTotalChargesRatio' feature (MonthlyCharges/TotalCharges)
        #     df['MonthlyChargesToTotalChargesRatio'] = df['MonthlyCharges'] / df['TotalCharges']

        # Make predictions
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
        prediction_labels = encoder.inverse_transform(predictions)

        # Update the session state with the prediction and probabilities
        st.session_state['predictions'] = predictions
        st.session_state['probability'] = probabilities
        st.session_state['prediction_labels'] = prediction_labels
        
        # Add prediction columns to the DataFrame
        df['PredictionTime'] = datetime.date.today()
        df['ModelUsed'] = st.session_state['selected_model']
        df['Prediction'] = prediction_labels
        df['Predicted Churn'] = predictions
        df['Predicted Churn'] = df['Predicted Churn'].map({1: 'Churn', 0: 'No Churn'})
        df['Probability'] = np.where(predictions == 1, np.round(probabilities[:, 1] * 100, 2), np.round(probabilities[:, 0] * 100, 2))

        # Make a copy of the dataframe
        dfp = df.copy()

        # Determine the correct file name for bulk predictions
        db_path = f"./data/{st.session_state['username']}/{st.session_state['username']}.db"
        conn = sqlite3.connect(db_path)
        
        # Create a unique table name based on the selected model
        model_name = st.session_state['selected_model'].replace(" ", "_").lower()
        db_path = f"./data/{st.session_state['username']}/{st.session_state['username']}.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Determine the next table name for the selected model with padding
        tables_query = f"""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '{model_name}_bulk_predict%'
        ORDER BY name DESC LIMIT 1;
        """
        result = cursor.execute(tables_query).fetchone()
        if result:
            last_table_name = result[0]
            last_table_number = int(last_table_name.split("_")[-1])
            new_table_number = last_table_number + 1
        else:
            new_table_number = 0

        # Add padding of 5 zeros to the table number
        new_table_name = f"{model_name}_bulk_predict_{str(new_table_number).zfill(6)}"

        # Save the DataFrame to the new bulk_predict table
        df.to_sql(new_table_name, conn, if_exists='replace', index=True)

        # Close the database connection
        conn.close()

        return predictions, probabilities, dfp
    
    # Fuction to make bulk predictions using template dataset
    def make_template_prediction(pipeline, encoder, template_df):
        # Convert the input data to a DataFrame
        df = template_df.copy()

        if 'customerID' in df.columns:
            df.set_index('customerID', inplace=True)

        # Ensure numerical columns are correctly typed
        df = df.apply(pd.to_numeric, errors='ignore')

        # Define the list of specific columns to check and coerce
        columns_to_coerce = ['tenure', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharges', 'MonthlyChargesToTotalChargesRatio']

        try:
            # Ensure numerical columns are correctly typed for specific columns
            for column in columns_to_coerce:
                if column in df.columns and df[column].dtype == 'object':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
        except Exception as e:
            st.error(f"An error occurred while processing the column '{column}': {e}")
            st.warning(
                """
                Please refer to the Data Overview page to apply the correct data structure, 
                ensuring numerical columns have strictly numeric values and categorical columns 
                have strictly categorical values.
                """
            )

        # Check if 'AvgMonthlyCharges' column exists
        # if 'AvgMonthlyCharges' not in df.columns:
        #  
        #     # Ensure there are no zero or missing values to avoid division by zero
        #     df['TotalCharges'].replace(0, 1, inplace=True)
        #     df['Tenure'].replace(0, 1, inplace=True)
            
        #     # Create the 'AvgMonthlyCharges' feature (TotalCharges/tenure)
        #     df['AvgMonthlyCharges'] = df['TotalCharges'] / df['Tenure']

        # # Check if 'MonthlyChargesToTotalChargesRatio' column exists
        # if 'MonthlyChargesToTotalChargesRatio' not in df.columns:
            
        #     # Ensure there are no zero or missing values to avoid division by zero
        #     df['TotalCharges'].replace(0, 1, inplace=True)
        #     df['MonthlyCharges'].replace(0, 1, inplace=True)
            
        #     # Create the 'MonthlyChargesToTotalChargesRatio' feature (MonthlyCharges/TotalCharges)
        #     df['MonthlyChargesToTotalChargesRatio'] = df['MonthlyCharges'] / df['TotalCharges']

        # Make predictions
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
        prediction_labels = encoder.inverse_transform(predictions)

        # Update the session state with the prediction and probabilities
        st.session_state['predictions'] = predictions
        st.session_state['probability'] = probabilities
        st.session_state['prediction_labels'] = prediction_labels
        
        # Add prediction columns to the DataFrame
        df['PredictionTime'] = datetime.date.today()
        df['ModelUsed'] = st.session_state['selected_model']
        df['Prediction'] = prediction_labels
        df['Predicted Churn'] = predictions
        df['Predicted Churn'] = df['Predicted Churn'].map({1: 'Churn', 0: 'No Churn'})
        df['Probability'] = np.where(predictions == 1, np.round(probabilities[:, 1] * 100, 2), np.round(probabilities[:, 0] * 100, 2))

        # Make a copy of the dataframe
        dfp = df.copy()

        # Create a unique table name based on the selected model
        model_name = st.session_state['selected_model'].replace(" ", "_").lower()
        
        # Define the path to the user's database
        db_path = f"./data/template/churn_predict.db"
        
        # Check if the database directory exists, create if not
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Connect to the SQLite database (creates the database if it doesn't exist)
        conn = sqlite3.connect(db_path)

        # Save the template DataFrame to the SQLite database as 'template_predict'
        df.to_sql(f'{model_name}_template_predict000000', conn, if_exists='replace', index=True)

        # Close the database connection
        conn.close()

        return predictions, probabilities, dfp
    
    # Functionality for Single Prediction
    if prediction_type == 'Single Prediction':
        st.subheader("Single Prediction")     
        get_user_input()
        
        # Display prediction results
        prediction = st.session_state['prediction']
        probability = st.session_state['probability']

        if prediction is None:
            st.stop()
        elif prediction == "Yes":
            st.markdown("##### Prediction Results:")
            probability_of_churn = probability[0][1] * 100
            st.markdown(f'Customer {st.session_state["customer_id"]} is likely to churn with a probability of {round(probability_of_churn, 2)}%.')
            st.info("Visit the History page to view and download the most recent single prediction dataset.")
        else:
            st.markdown("##### Prediction Results:")
            probability_of_no_churn = probability[0][0] * 100
            st.markdown(f'Customer {st.session_state["customer_id"]} is unlikely to churn with a probability of {round(probability_of_no_churn, 2)}%.')
            st.info("Visit the History page to view and download the most recent single prediction dataset.")

    # Functionality for Bulk Prediction
    if prediction_type == 'Bulk Prediction':
        st.subheader("Bulk Prediction")
        pipeline, encoder = get_model(selected_model)

        if pipeline is None:
            st.stop()

        if st.button("Run Bulk Prediction"):
            st.session_state['data_source'] = 'uploaded'
            uploaded_df, table_name = load_most_recent_table(username)
            st.write(f"Using dataset: {table_name}")
            df = uploaded_df

            if df is not None:
                st.write("First 5 rows of the uploaded dataset:")
                st.dataframe(df.head())                
                predictions, probabilities, dfp = make_bulk_prediction(pipeline, encoder, df)
                st.write("First 5 rows of the prediction results:")
                st.dataframe(dfp.head())
                st.info("Visit the History page to view and download the full bulk prediction dataset.")

    # Functionality for Template Prediction
    if prediction_type == 'Template Prediction':
        st.subheader("Template Prediction")
        pipeline, encoder = get_model(selected_model)

        if pipeline is None:
            st.stop()

        if st.button("Run Template Prediction"):
            st.session_state['data_source'] = 'initial'
            table_name = "Template Dataset"
            st.write(f"Using dataset: {table_name}")
            df = initial_df

            if df is not None:
                st.write("First 5 rows of the template dataset:")
                st.dataframe(df.head())                
                predictions, probabilities, dfp = make_template_prediction(pipeline, encoder, df)
                st.write("First 5 rows of the prediction results:")
                st.dataframe(dfp.head())
                st.info("Visit the History page to view and download the full template prediction dataset.")
else:
    st.warning("Please log in to make predictions.")
    

# Function to convert an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Image paths
image_paths = ["./assets/favicon.png"]

# Convert images to base64
image_b64 = [image_to_base64(img) for img in image_paths]

# Need Help Section
st.markdown("Need help? Contact support at [sdi@azubiafrica.org](mailto:sdi@azubiafrica.org)")

st.write("---")

# Contact Information Section
st.markdown(
f"""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="flex: 1;">
        <h2>Contact Us</h2>
        <p>For inquiries, please reach out to us:</p>
        <p>📍 Address: Accra, Ghana</p>
        <p>📞 Phone: +233 123 456 789</p>
        <p>📧 Email: sdi@azubiafrica.org</p>
    </div>
    <div style="flex: 0 0 auto;">
        <img src="data:image/png;base64,{image_b64[0]}" style="width:100%";" />
    </div>
</div>
""",
unsafe_allow_html=True
) 