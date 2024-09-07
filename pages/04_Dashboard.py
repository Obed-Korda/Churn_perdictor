# Standard library imports
import base64
import os
import sqlite3

# Third-party imports
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Local application imports
from utils.login import invoke_login_widget
from utils.lottie import display_lottie_on_page


# Invoke the login form
invoke_login_widget('Analytics Dashboard')

# Fetch the authenticator from session state
authenticator = st.session_state.get('authenticator')

# Ensure the authenticator is available
if not authenticator:
    st.error("Authenticator not found. Please check the configuration.")
    st.stop()

# Check authentication status
if st.session_state.get("authentication_status"):
    username = st.session_state['username']
    st.title('Telco Churn Analysis')
    st.write("---")

    # Page Introdution
    with st.container():        
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("""
            Welcome to the **Telco Churn Analysis** dashboard. This tool provides comprehensive insights into customer churn through two primary analysis types:

            1. **Exploratory Data Analysis (EDA):** Explore customer data through visualizations to identify key demographic and account trends that can influence churn.
            2. **Key Performance Indicators (KPIs):** Analyze critical metrics such as total customers, retention rates, and revenue, with the ability to filter and compare data.

            Select the type of analysis you wish to conduct from the dropdown menu and dive into the data to uncover actionable insights that can drive strategic decisions for customer retention and growth.
            """)
        with right_column:
            display_lottie_on_page("Analytics Dashboard")

    # Add selectbox to choose between EDA and KPIs
    selected_analysis = st.selectbox('Select Analysis Type', ['', '🔍 Exploratory Data Analysis (EDA)', '📊 Key Performance Indicators (KPIs)'], index=0)
    
    # Load the initial data from a local file
    @st.cache_data(persist=True, show_spinner=False)
    def load_initial_data():
        df = pd.read_csv('./data/LP2_train_final.csv')
        return df
    
    initial_df = load_initial_data()

    # Ensure 'data_source' is initialized in session state
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = 'initial'  

    # Function to load the most recent table from the user's SQLite database
    def load_most_recent_table(username):
        # Define the path for the user's SQLite database
        db_path = f"./data/{username}/{username}.db"

        if not os.path.exists(db_path):
            st.error("No database found for the user. Please ensure a file has been uploaded on the data overview page.")
            return None, None

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Get the most recent table name
        tables_query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY tbl_name DESC LIMIT 1;
        """
        try:
            most_recent_table = conn.execute(tables_query).fetchone()

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

    # User interface for selecting the dataset
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Template Dataset"):
            st.session_state['data_source'] = 'initial'

    with col2:
        if st.button("Uploaded Dataset"):
            st.session_state['data_source'] = 'uploaded'

    # Load the appropriate dataset based on the user's choice
    if st.session_state['data_source'] == 'initial':
        df = initial_df
        table_name = "Template Dataset"
    else:
        uploaded_df, table_name = load_most_recent_table(username)
        df = uploaded_df if uploaded_df is not None else initial_df

    # Display the dataset currently in use
    st.write(f"Using dataset: {table_name}")

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
        st.stop() 
       
    # Handle missing values
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    int64_columns = df.select_dtypes(include=['int64']).columns.tolist()

    # Impute numerical columns with median
    numerical_imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    # Convert columns that were originally int64 back to int64
    for column in int64_columns:
        df[column] = df[column].astype('int64')


    # Create a function to apply filters
    def apply_filters(df):
        slider_values = {}
        for column in numerical_columns:
            if df[column].dtype == 'int64':
                min_value = int(df[column].min())
                max_value = int(df[column].max())
            else:
                min_value = float(df[column].min())
                max_value = float(df[column].max())
            slider_values[column] = st.sidebar.slider(
                column,
                min_value,
                max_value,
                (min_value, max_value)
            )

        filtered_data = df.copy()
        for column, (min_val, max_val) in slider_values.items():
            filtered_data = filtered_data[
                (filtered_data[column] >= min_val) & (filtered_data[column] <= max_val)
            ]
        return filtered_data

    # Apply filters to the data
    filtered_data = apply_filters(df)

    # Exploratory Data Analysis
    if selected_analysis == '':
        st.write("Please select an analysis type to begin.")

    elif selected_analysis == '🔍 Exploratory Data Analysis (EDA)':
        st.subheader("🕵🏾‍♂️ Churn EDA Dashboard")
        st.write(
            """This dashboard provides an exploratory analysis of customer churn data, 
            including detailed insights into customer contractual obligations and subscription patterns. 
            The visualizations help identify key demographic and account characteristics, correlations, 
            and trends that can guide strategic decisions. Use the filters and plots to analyze customer 
            behavior, understand contractual impacts, and identify potential areas for improvement in 
            subscription models."""
        )
        
        # Customer Demographic Analysis
        st.markdown("#### Customer Demographic Analysis")
        st.write(
            "This section analyzes customer demographics to understand the distribution of key attributes such as gender, age, and relationships. By examining these factors, you can uncover patterns that might influence customer retention and acquisition."
        )
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                gender_plot = px.histogram(filtered_data, x="Gender", color="Churn", barmode="group", title="Gender Distribution")
                st.plotly_chart(gender_plot, use_container_width=True)

            with col2:
                senior_citizen_plot = px.histogram(filtered_data, x="SeniorCitizen", color="Churn", barmode="group", title="Senior Citizen Distribution")
                st.plotly_chart(senior_citizen_plot, use_container_width=True)
 
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                partner_plot = px.histogram(filtered_data, x="Partner", color="Churn", barmode="group", title="Partner Distribution")
                st.plotly_chart(partner_plot, use_container_width=True)
            
            with col2:
                dependents_plot = px.histogram(filtered_data, x="Dependents", color="Churn", barmode="group", title="Dependents Distribution")
                st.plotly_chart(dependents_plot, use_container_width=True)
        
        # Customer Account Analysis
        st.markdown("#### Customer Account Analysis")        
        st.write(
            "This section provides insights into customer account characteristics, including monthly and total charges, as well as tenure. The distributions and histograms reveal patterns in spending and account duration, which are crucial for understanding customer value and predicting churn."
        )
        with st.container():
            col1, col2, col3 = st.columns(3)

            with col1:
                monthly_charges_plot = px.histogram(filtered_data, x='MonthlyCharges', nbins=20, color='Churn', title='Monthly Charges Distribution')
                st.plotly_chart(monthly_charges_plot, use_container_width=True)

            with col2:
                total_charges_plot = px.histogram(filtered_data, x='TotalCharges', nbins=20, color='Churn', title='Total Charges Distribution')
                st.plotly_chart(total_charges_plot, use_container_width=True)

            with col3:                         
                tenure_plot = px.histogram(filtered_data, x='Tenure', nbins=20, color='Churn', title='Tenure Distribution')
                st.plotly_chart(tenure_plot, use_container_width=True)   

        with st.container():
            filtered_data['Churn'] = filtered_data['Churn'].map({'Yes': 1, 'No': 0})
            col1, col2 = st.columns(2)

            with col1:
                # Correlation Heatmap
                corr_matrix = filtered_data[["Churn", "MonthlyCharges", "TotalCharges", "Tenure"]].dropna().corr()

                # Annotate the heatmap with correlation values
                heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    text=corr_matrix.values,  
                    texttemplate="%{text:.2f}",  
                    showscale=True  
                ))

                heatmap.update_layout(
                    title="Correlation Matrix",
                    xaxis_nticks=36
                )

                st.plotly_chart(heatmap)
            
            with col2:
                filtered_data['Churn'] = filtered_data['Churn'].map({1: 'Yes', 0: 'No'}).fillna('Unknown')
                # Pair Plot
                pairplot_fig = px.scatter_matrix(
                    filtered_data[["Churn", "Tenure", "MonthlyCharges", "TotalCharges"]],
                    dimensions=["TotalCharges", "Tenure", "MonthlyCharges"],
                    color="Churn",
                    title="Pairplot"
                )
                st.plotly_chart(pairplot_fig)

        # Customer Contractual Analysis
        st.markdown("#### Customer Contractual Analysis")
        st.write(
            "This section examines customer contracts and payment methods. It provides an overview of contract types, payment methods, and billing preferences, which can offer insights into customer loyalty and potential areas for optimizing pricing strategies."
        )
        with st.container():
            
            col1, col2, col3 = st.columns(3)

            with col1:
                contract_plot = px.histogram(filtered_data, x="Contract", color="Churn", barmode="group", title="Contract Distribution")
                st.plotly_chart(contract_plot, use_container_width=True)

            with col2:
                payment_method_plot = px.histogram(filtered_data, x="PaymentMethod", color="Churn", barmode="group", title="Payment Method Distribution")
                st.plotly_chart(payment_method_plot, use_container_width=True)

            with col3:
                paperless_billing_plot = px.histogram(filtered_data, x="PaperlessBilling", color="Churn", barmode="group", title="Paperless Billing Distribution")
                st.plotly_chart(paperless_billing_plot, use_container_width=True)

        # Customer Subscription Analysis
        st.markdown("#### Customer Subscription Analysis")
        st.write(
            "This section explores customer service subscriptions, including phone, internet, and tech support services. It highlights the distribution of these services among customers and their relationship with churn behavior."
        )
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                phone_service_plot = px.histogram(filtered_data, x="PhoneService", color="Churn", barmode="group", title="Phone Service Distribution")
                st.plotly_chart(phone_service_plot, use_container_width=True)

            with col2:
                multiple_lines_plot = px.histogram(filtered_data, x="MultipleLines", color="Churn", barmode="group", title="Multiple Lines Distribution")
                st.plotly_chart(multiple_lines_plot, use_container_width=True)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                internet_service_plot = px.histogram(filtered_data, x="InternetService", color="Churn", barmode="group", title="Internet Service Distribution")
                st.plotly_chart(internet_service_plot, use_container_width=True)

            with col2:
                techsupport_plot = px.histogram(filtered_data, x="TechSupport", color="Churn", barmode="group", title="Tech Support Distribution")
                st.plotly_chart(techsupport_plot, use_container_width=True)

        # Key Business Insights for template dataset
        if st.session_state['data_source'] == 'initial':
            st.markdown("""
            #### Key Business Insights

            ##### 1. Customer Demographic Analysis
            - Churn rates are consistent across genders, indicating that gender is not a major factor in predicting churn.
            - Senior citizens show higher churn rates, suggesting they might have unique needs or challenges with the service.
            - Customers with partners are less likely to churn, hinting at a correlation between relationship stability and customer loyalty.
            - Customers with dependents also exhibit lower churn rates, possibly due to family responsibilities influencing their decision to stay.

            ##### 2. Customer Account Analysis
            - Higher monthly charges do not appear to have a clear relationship with churn, as the distributions for both churn and non-churn customers overlap significantly.
            - There is no strong indication that total charges correlate with churn, with most customers having lower total charges, regardless of churn status.
            - Churn is more frequent among customers with shorter tenures. As tenure increases, churn decreases, implying that newer customers are at a higher risk of leaving, while longer-term customers are more likely to stay.

            ##### 3. Customer Contractual Analysis
            - Month-to-month contracts have a significantly higher churn rate compared to one-year or two-year contracts, suggesting that encouraging longer-term commitments could reduce churn.
            - Customers using electronic checks are more likely to churn, potentially indicating dissatisfaction with this payment method.
            - Customers who opt for paperless billing tend to have higher churn rates, which could imply that these customers, being more digitally savvy, are more open to exploring other service options.

            ##### 4. Customer Subscription Analysis
            - Most customers use phone services, and while churn is significant, a larger portion of these customers do not churn. Customers without phone service show lower churn rates overall.
            - Most customers do not have multiple lines. The churn rate is almost the same for customers with and without multiple lines, but in both cases, the number of customers who do not churn is significantly higher than those who do.
            - Fiber optic internet service users have higher churn rates compared to DSL users, which may point to dissatisfaction with the value or quality of fiber optic service.
            - The lack of tech support is strongly linked to higher churn, emphasizing the importance of providing reliable tech support to retain customers.
            """)

    # Key Performance Indicators
    elif selected_analysis == '📊 Key Performance Indicators (KPIs)':
        st.subheader("📈 Churn KPI Dashboard")

        st.markdown("""
        This dashboard provides key performance indicators (KPIs) related to customer churn. It offers insights into:

        - **Total Customers:** Number of customers after applying filters.
        - **Total Customers Retained:** Number of customers retained, showing changes after filtering.
        - **Average Tenure:** Average duration customers stay, and how it has changed.
        - **Average Monthly Charges:** Changes in the average charges customers incur.
        - **Total Revenue:** How total revenue has shifted with applied filters.
        - **Churn Rate Gauge:** Visual representation of churn rate changes relative to the unfiltered data.

        Use this dashboard to analyze the impact of various filters on customer retention and overall business metrics.
        """)

        # Apply a map to the data frame for the chun column
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        filtered_data['Churn'] = filtered_data['Churn'].map({'Yes': 1, 'No': 0})
  
        # Calculate unfiltered values
        unfiltered_total_customers = df.shape[0]
        unfiltered_total_customers_retained = len(df[df["Churn"] == 0])
        unfiltered_avg_tenure = df['Tenure'].mean()
        unfiltered_avg_monthly_charges = df['MonthlyCharges'].mean()
        unfiltered_total_revenue = df['TotalCharges'].sum()

        # Churn KPI metrics
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                # KPI 1: Total Customers
                total_customers = filtered_data.shape[0]
                total_customers_delta = (total_customers - unfiltered_total_customers) / unfiltered_total_customers * 100
                st.metric(
                    label="Total Customers", 
                    value=f"{total_customers:,}", 
                    delta=f"{total_customers_delta:.2f}%", 
                    help="This percentage shows how the total number of customers has changed after applying the selected filters."
                )

            with col2:
                # KPI 2: Total Customers Retained
                total_customers_retained = len(filtered_data[filtered_data["Churn"] == 0])
                total_customers_retained_delta = (total_customers_retained - unfiltered_total_customers_retained) / unfiltered_total_customers_retained * 100
                st.metric(
                    label="Total Customers Retained", 
                    value=f"{total_customers_retained:,}", 
                    delta=f"{total_customers_retained_delta:.2f}%",
                    help="This percentage shows the change in the number of customers retained after applying the selected filters."
                )

            with col3:
                # KPI 3: Average Tenure
                avg_tenure = filtered_data['Tenure'].mean()
                avg_tenure_delta = (avg_tenure - unfiltered_avg_tenure) / unfiltered_avg_tenure * 100
                st.metric(
                    label="Avg. Tenure (Months)", 
                    value=f"{avg_tenure:.1f}", 
                    delta=f"{avg_tenure_delta:.2f}%",
                    help="This percentage shows how the average customer tenure has changed after applying the selected filters."
                )

            with col4:
                # KPI 4: Average Monthly Charges
                avg_monthly_charges = filtered_data['MonthlyCharges'].mean()
                avg_monthly_charges_delta = (avg_monthly_charges - unfiltered_avg_monthly_charges) / unfiltered_avg_monthly_charges * 100
                st.metric(
                    label="Avg. Monthly Charges", 
                    value=f"${avg_monthly_charges:.2f}", 
                    delta=f"{avg_monthly_charges_delta:.2f}%",
                    help="This percentage indicates how average monthly charges have changed after applying the selected filters."
                )

            with col5:
                # KPI 5: Total Revenue
                total_revenue = filtered_data['TotalCharges'].sum()
                total_revenue_delta = (total_revenue - unfiltered_total_revenue) / unfiltered_total_revenue * 100
                st.metric(
                    label="Total Revenue", 
                    value=f"${total_revenue/1e6:,.2f}M", 
                    delta=f"{total_revenue_delta:.2f}%",
                    help="This percentage shows how total revenue has shifted after applying the selected filters."
                )

        # KPI 6: Churn Rate Gauge
        churn_rate = filtered_data['Churn'].mean() * 100
        unfiltered_churn_rate = df['Churn'].mean() * 100

        fig_churn_rate = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_rate,
            number={'suffix': "%", 'valueformat': ".2f"},
            delta={
                'reference': unfiltered_churn_rate, 
                'relative': True, 
                'position': "top", 
                'valueformat': ".2f",
                'suffix': "%",  
                'increasing': {'color': "red"}, 
                'decreasing': {'color': "green"}  
            },
            title={'text': "Churn Rate"},
            gauge={
                "axis": {"range": [0, 100], "tickformat": ".2f%"},
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": churn_rate
                }
            }
        ))

        st.plotly_chart(fig_churn_rate)

        # Display a description or Chunn Rate Gauge
        st.markdown("""
        **Churn Rate Gauge:** This gauge displays the current predicted churn rate along with the percentage change (delta) compared to the previous churn rate you entered. It helps visualize how the churn rate has changed after applying the selected filters.

        - **Positive Delta (in red):** Indicates that the churn rate has increased after filtering, meaning more customers are leaving.
        - **Negative Delta (in green):** Indicates that the churn rate has decreased after filtering, meaning fewer customers are leaving.

        In simpler terms, the gauge shows not just the current churn rate but also how the current rate compares to the rate before you applied the filters. 
        For example, if the churn rate was 10% before filtering and now it’s 15%, a positive delta of 50% would show that the churn rate increased by half relative to the initial rate. 
        The gauge provides insights into whether customer retention has improved or worsened after applying your filters.
        """)

        # Distribution of Features
        st.markdown("#### Distribution of Features")
        st.markdown("""
        This section visualizes the distribution of key features in the dataset. It includes:
        - **Contract Distribution:** Shows the breakdown of customer contracts.
        - **Payment Method Distribution:** Displays how customers are distributed across different payment methods.
        - **Internet Service Distribution:** Illustrates how customers are distributed across various internet service types.
        - **Phone Service Distribution:** Demonstrates the distribution of customers based on their phone service options.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Plot: Contract Distribution
            contract_distribution = filtered_data['Contract'].value_counts()
            fig_contract = px.pie(contract_distribution, values=contract_distribution.values, names=contract_distribution.index, hole=0.3, title="Contract Distribution")
            st.plotly_chart(fig_contract, use_container_width=True)

            # Plot: Internet Service Distribution
            internet_service_distribution = filtered_data['InternetService'].value_counts()
            fig_internet_service_distribution = px.pie(internet_service_distribution, values=internet_service_distribution.values, names=internet_service_distribution.index, hole=0.3, title="Internet Service Distribution")
            st.plotly_chart(fig_internet_service_distribution, use_container_width=True)

        with col2:
            # Plot: Payment Method Distribution
            payment_method_distribution = filtered_data['PaymentMethod'].value_counts()
            fig_payment_method_distribution = px.pie(payment_method_distribution, values=payment_method_distribution.values, names=payment_method_distribution.index, hole=0.3, title="Payment Method Distribution")
            st.plotly_chart(fig_payment_method_distribution, use_container_width=True)

            # Plot: Phone Service Distribution
            phone_service_distribution = filtered_data['PhoneService'].value_counts()
            fig_phone_service_distribution = px.pie(phone_service_distribution, values=phone_service_distribution.values, names=phone_service_distribution.index, hole=0.3, title="Phone Service Distribution")
            st.plotly_chart(fig_phone_service_distribution, use_container_width=True)


        # Comparing Feature Parameters Based on Churn Rate
        st.markdown("#### Comparing Feature Parameters Based on Churn Rate")
        st.markdown("""
        This section provides insights into how different feature parameters relate to churn rates. It includes:
        - **Churn Rate by Gender:** Compares churn rates across different genders.
        - **Churn Rate Over Tenure:** Shows how churn rate changes with customer tenure.
        - **Churn Rate by Contract Type:** Examines how churn rates vary with different contract types.
        - **Churn Rate by Payment Method:** Investigates churn rates based on different payment methods.
        - **Average Monthly Charges by Contract Type:** Analyzes average charges based on contract type.
        - **Total Monthly Charges by Contract Type:** Displays total monthly charges for each contract type.
        - **Churn Rate by Internet Service:** Looks at how internet service type affects churn rates.
        - **Churn Rate by Phone Service:** Analyzes churn rates based on whether customers have phone service.
        """)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                # Plot: Churn Rate by Gender
                churn_by_gender = filtered_data.groupby('Gender')['Churn'].mean().reset_index()
                churn_by_gender['Churn'] = churn_by_gender['Churn'] * 100
                fig_gender_churn = px.bar(churn_by_gender, x='Gender', y='Churn', title='Churn Rate by Gender')
                st.plotly_chart(fig_gender_churn, use_container_width=True)

            with col2:
                # Plot: Line Chart for Churn Rate over Tenure
                churn_rate_by_tenure = filtered_data.groupby('Tenure')['Churn'].mean().reset_index()
                fig_churn_tenure = px.line(churn_rate_by_tenure, x='Tenure', y='Churn', title='Churn Rate over Tenure')
                st.plotly_chart(fig_churn_tenure, use_container_width=True)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                # Plot: Churn Rate by Contract Type
                churn_by_contract = filtered_data.groupby('Contract')['Churn'].mean().reset_index()
                churn_by_contract['Churn'] = churn_by_contract['Churn'] * 100
                fig_contract_churn = px.bar(churn_by_contract, x='Contract', y='Churn', title='Churn Rate by Contract Type')
                st.plotly_chart(fig_contract_churn, use_container_width=True)

            with col2:
                # Plot: Churn Rate by Payment Method
                churn_by_payment_method = filtered_data.groupby('PaymentMethod')['Churn'].mean().reset_index()
                churn_by_payment_method['Churn'] = churn_by_payment_method['Churn'] * 100
                fig_churn_by_payment_method = px.bar(churn_by_payment_method, x='PaymentMethod', y='Churn', title='Churn Rate by Payment Method')
                st.plotly_chart(fig_churn_by_payment_method, use_container_width=True)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                # Plot: Average Monthly Charges by Contract Type
                avg_charges_by_contract = filtered_data.groupby('Contract')['MonthlyCharges'].mean().reset_index()
                fig_avg_contract_charges = px.bar(avg_charges_by_contract, x='Contract', y='MonthlyCharges', title='Avg. Monthly Charges by Contract Type')
                st.plotly_chart(fig_avg_contract_charges)

            with col2:
                # Plot: Total Monthly Charges by Contract Type
                total_charges_by_contract = filtered_data.groupby('Contract')['TotalCharges'].mean().reset_index()
                fig_total_contract_charges = px.bar(total_charges_by_contract, x='Contract', y='TotalCharges', title='Total Monthly Charges by Contract Type')
                st.plotly_chart(fig_total_contract_charges)

        with st.container():
            col1, col2 = st.columns(2)

            with col1:           
                # Plot: Churn Rate by Internet Service
                churn_by_internet_service = filtered_data.groupby('InternetService')['Churn'].mean().reset_index()
                churn_by_internet_service['Churn'] = churn_by_internet_service['Churn'] * 100
                fig_churn_by_internet_service = px.bar(churn_by_internet_service, x='InternetService', y='Churn', title='Churn Rate by Internet Service')
                st.plotly_chart(fig_churn_by_internet_service)

            with col2:
                # Plot: Churn Rate by Phone Service
                churn_by_phone_service = filtered_data.groupby('PhoneService')['Churn'].mean().reset_index()
                churn_by_phone_service['Churn'] = churn_by_phone_service['Churn'] * 100
                fig_churn_by_phone_service = px.bar(churn_by_phone_service, x='PhoneService', y='Churn', title='Churn Rate by Phone Service')
                st.plotly_chart(fig_churn_by_phone_service)

        # KPI data
        kpi_data = {
            'KPI': ['Total Customers', 'Total Customers Retained', 'Churn Rate', 'Avg. Tenure', 'Avg. Monthly Charges', 'Total Revenue'],
            'Value': [f"{total_customers:,}", f"{total_customers_retained:,}", f"{churn_rate:.2f}%", f"{avg_tenure:.1f} months", f"${avg_monthly_charges:.2f}", f"${total_revenue:,.2f}"]
        }

        # Create DataFrame
        kpi_df = pd.DataFrame(kpi_data)
        kpi_df.set_index('KPI', inplace=True)

        # Function to apply conditional formatting based on the value
        def color_kpi_value(value):
            if '%' in value:
                percent_value = float(value.strip('%'))
                if percent_value < 30:
                    color = 'green'
                elif 30 <= percent_value < 70:
                    color = 'yellow'
                else:  
                    color = 'red'
            else:
                color = 'lightblue'
            return f'color: {color}'

        # Function to apply conditional formatting
        def highlight_churn(index):
            color = 'background-color: #4B61F5' if index.name == 'Total Revenue' else ''
            return [color] * len(index)

        # Apply the color_negative_red function to the 'Value' column
        styled_df = kpi_df.style.applymap(color_kpi_value, subset=['Value'])

        # Apply the highlight_churn function to the entire row
        styled_df = styled_df.apply(highlight_churn, axis=1)

        # Display the styled DataFrame
        st.markdown("#### Key Performance Indicators (KPIs)")
        st.table(styled_df)

        # Key Business Insights for template dataset
        if st.session_state['data_source']== "initial":
            st.markdown("""
            #### Key Business Insights
            
            ##### 1. Distribution of Features
            **Contract Distribution:**
            - **High Month-to-Month Contracts:** With **54.4%** of customers on month-to-month contracts, the business faces a higher risk of customer churn since this contract type typically lacks long-term commitment. This indicates a need for strategies to convert these customers to longer-term contracts, such as one-year or two-year plans, to improve retention.

            **Payment Method Distribution:**
            - **Electronic Check Dominance:** The preference for electronic checks (**33.7%**) suggests that a significant portion of customers prefers this method, which correlates with higher churn rates. This may indicate that customers using this payment method are less satisfied or more likely to leave, potentially due to perceived financial instability or dissatisfaction with the payment process. Efforts could be made to encourage the use of more stable, automated payment methods like credit cards.

            **Internet Service Distribution:**
            - **Fiber Optic Popularity:** Fiber optic being the most common internet service (**44.6%**) indicates that customers value high-speed internet. However, the higher churn rate among fiber optic users suggests possible issues with service quality or pricing that need to be addressed to retain these customers.

            **Phone Service Distribution:**
            - **High Phone Service Adoption:** The fact that **90.3%** of customers have phone service implies that phone services are still a critical part of the product offering. However, the slight increase in churn among these customers might indicate dissatisfaction with the service or pricing, suggesting a need for reviewing the phone service offerings.

            ##### 2. Comparing Feature Parameters Based on Churn Rate
            **Churn Rate by Gender:**
            - **Equal Churn Across Genders:** The nearly equal churn rates between females and males suggest that gender does not significantly influence churn, meaning retention strategies can be uniformly applied across genders without needing gender-specific adjustments.

            **Churn Rate by Tenure:**
            - **Tenure's Protective Effect:** The decrease in churn rate as customer tenure increases highlights the importance of building long-term relationships with customers. Retention strategies should focus on the early stages of customer relationships, possibly offering incentives for loyalty early on to reduce churn.

            **Churn Rate by Contract Type:**
            - **Month-to-Month Vulnerability:** Customers on month-to-month contracts show the highest churn, underscoring the need for transitioning these customers to longer-term contracts. These could include incentives like discounts, additional services, or bundled offers to lock in commitment.

            **Churn Rate by Payment Method:**
            - **Risk with Electronic Checks:** The high churn rate among customers paying by electronic check indicates a potential risk group. Encouraging a switch to automated credit card payments could reduce churn, as these customers are less likely to leave.

            **Churn Rate by Internet Service:**
            - **Fiber Optic Churn Concerns:** The higher churn rate among fiber optic customers suggests that despite its popularity, there may be underlying issues such as pricing, competition, or service quality that need addressing to retain these customers.

            **Churn Rate by Phone Service:**
            - **Phone Service Challenges:** Although the majority of customers have phone service, the slight increase in churn for these customers might point to issues with the perceived value of this service, indicating a need to reassess and possibly enhance phone service offerings.

            ##### 3. Financial Indicators
            **Average Monthly Charges by Contract Type:**
            - **Higher Charges for Short-Term Contracts:** The higher monthly charges for month-to-month contracts suggest that while these customers are generating more revenue per month, they are also at a higher risk of churn. This could indicate a need for balancing pricing strategies to make longer-term contracts more attractive without significantly lowering the overall revenue.

            **Total Monthly Charges by Contract Type:**
            - **Revenue Generation by Contract Length:** Two-year contracts generate the most revenue, indicating that customers on longer-term contracts contribute more to the financial health of the company. This reinforces the need to shift customers towards longer-term commitments.

            ##### 4. Key Performance Indicators (KPIs)
            - **Churn Rate:** The current churn rate is **26.50%**, which, while below the 30-40% threshold typically seen as high on an annual basis, is still significant. This rate suggests that retention strategies are partially effective, but there is still room for improvement. Focus should be on enhancing customer satisfaction and reducing churn further to maintain a healthy customer base.
            - **Stable Customer Tenure:** An average tenure of **32.6 months** suggests that while there is a significant long-term customer base, efforts should still focus on extending this average by improving early retention.
            - **Revenue and Charges Stability:** The average monthly charge of 65 us dollars and total revenue of 11.60M us dollars indicate a strong financial position. However, maintaining or growing this will depend on reducing churn and increasing customer satisfaction and retention.
            """)
else:
    st.warning("Please log in to visualize your data.")
    

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