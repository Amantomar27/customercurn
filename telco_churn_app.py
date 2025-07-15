import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        self.df = None
        
    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload a CSV file")
                return None
            return self.df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self):
        """Preprocess the data"""
        if self.df is None:
            st.error("No data loaded")
            return
            
        # Handle TotalCharges column
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Fill missing values
        if self.df['TotalCharges'].isnull().sum() > 0:
            median_total_charges = self.df['TotalCharges'].median()
            self.df['TotalCharges'].fillna(median_total_charges, inplace=True)
        
        # Convert SeniorCitizen to categorical
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Create binary target
        self.df['Churn_Binary'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        
    def feature_engineering(self):
        """Create engineered features"""
        if self.df is None:
            return
            
        # Average monthly charges
        self.df['AvgMonthlyCharges'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)
        
        # Charges per service
        service_count = self.df[['PhoneService', 'InternetService']].apply(
            lambda x: sum([1 for val in x if val not in ['No', 'No internet service']]), axis=1
        ).replace(0, 1)
        self.df['ChargesPerService'] = self.df['MonthlyCharges'] / service_count
        
        # Tenure groups
        self.df['TenureGroup'] = pd.cut(
            self.df['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-1_year', '1-2_years', '2-4_years', '4plus_years'],
            include_lowest=True
        ).astype(str)
        
        # Monthly charges groups
        self.df['MonthlyChargesGroup'] = pd.cut(
            self.df['MonthlyCharges'], 
            bins=[0, 35, 65, 95, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Very_High'],
            include_lowest=True
        ).astype(str)
        
        # Count total services
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        def count_services(row):
            return sum([1 for col in service_cols 
                       if row[col] not in ['No', 'No internet service', 'No phone service']])
        
        self.df['TotalServices'] = self.df.apply(count_services, axis=1)
    
    def prepare_features(self):
        """Prepare features for ML"""
        if self.df is None:
            return None, None
            
        # Drop unnecessary columns
        feature_df = self.df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        
        # Get categorical and numerical features
        categorical_features = feature_df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Churn_Binary' in numerical_features:
            numerical_features.remove('Churn_Binary')
        
        # Encode categorical variables
        encoded_df = feature_df.copy()
        
        for col in categorical_features:
            if col != 'Churn_Binary':
                encoded_df[col] = encoded_df[col].fillna('Unknown')
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Prepare final matrices
        X = encoded_df.drop('Churn_Binary', axis=1)
        y = encoded_df['Churn_Binary']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, X, y, model_type='Random Forest'):
        """Train the selected model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        self.model = models[model_type]
        
        # Train model
        if model_type in ['Logistic Regression', 'SVM']:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            accuracy = self.model.score(X_test_scaled, y_test)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            accuracy = self.model.score(X_test, y_test)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test,
            'model_type': model_type
        }
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }
            
            with open('churn_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            with open('churn_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

def create_visualizations(df):
    """Create interactive visualizations"""
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Churn Distribution 1")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, 
                     color_discrete_sequence=['#2E86AB', '#A23B72'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Monthly Charges by Churn")
        fig = px.box(df, x='Churn', y='MonthlyCharges', 
                     color='Churn', color_discrete_sequence=['#2E86AB', '#A23B72'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Contract analysis
    st.subheader("📋 Churn Analysis by Contract Type")
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])
    fig = px.bar(contract_churn, barmode='group', 
                 color_discrete_sequence=['#2E86AB', '#A23B72'])
    fig.update_layout(xaxis_title="Contract Type", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu', color_continuous_midpoint=0)
    fig.update_layout(title="Feature Correlations")
    st.plotly_chart(fig, use_container_width=True)

def create_model_evaluation_plots(results):
    """Create model evaluation visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        
        fig = px.imshow(cm, text_auto=True, aspect="auto", 
                        color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Churn', 'Churn']),
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f"ROC Curve (AUC = {results['auc_score']:.3f})",
                                line=dict(color='#2E86AB', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name="Random", line=dict(dash='dash', color='gray')))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

def predict_single_customer(predictor):
    """Interface for single customer prediction"""
    
    st.subheader("🔮 Single Customer Churn Prediction")
    
    if not predictor.model:
        st.warning("Please train a model first or load a pre-trained model.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    
    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 18.0, 8500.0, 1500.0)
    
    if st.button("🚀 Predict Churn Probability", type="primary"):
        try:
            # Create customer data
            customer_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }
            
            # Create engineered features
            customer_data['AvgMonthlyCharges'] = total_charges / (tenure + 1)
            
            # Charges per service
            service_count = sum([1 for val in [phone_service, internet_service] 
                               if val not in ['No', 'No internet service']])
            customer_data['ChargesPerService'] = monthly_charges / max(service_count, 1)
            
            # Tenure group
            if tenure <= 12:
                customer_data['TenureGroup'] = '0-1_year'
            elif tenure <= 24:
                customer_data['TenureGroup'] = '1-2_years'
            elif tenure <= 48:
                customer_data['TenureGroup'] = '2-4_years'
            else:
                customer_data['TenureGroup'] = '4plus_years'
            
            # Monthly charges group
            if monthly_charges <= 35:
                customer_data['MonthlyChargesGroup'] = 'Low'
            elif monthly_charges <= 65:
                customer_data['MonthlyChargesGroup'] = 'Medium'
            elif monthly_charges <= 95:
                customer_data['MonthlyChargesGroup'] = 'High'
            else:
                customer_data['MonthlyChargesGroup'] = 'Very_High'
            
            # Total services
            service_cols = [phone_service, multiple_lines, internet_service, online_security,
                           online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
            total_services = sum([1 for val in service_cols 
                                if val not in ['No', 'No internet service', 'No phone service']])
            customer_data['TotalServices'] = total_services
            
            # Convert to DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Encode categorical variables
            for col in customer_df.select_dtypes(include=['object']).columns:
                if col in predictor.label_encoders:
                    le = predictor.label_encoders[col]
                    try:
                        customer_df[col] = le.transform(customer_df[col])
                    except ValueError:
                        # Handle unseen categories
                        customer_df[col] = 0
            
            # Ensure all features are present
            for feature in predictor.feature_names:
                if feature not in customer_df.columns:
                    customer_df[feature] = 0
            
            # Reorder columns to match training data
            customer_df = customer_df[predictor.feature_names]
            
            # Scale if needed
            model_type = getattr(predictor, 'model_type', 'Random Forest')
            if model_type in ['Logistic Regression', 'SVM']:
                customer_scaled = predictor.scaler.transform(customer_df)
                probability = predictor.model.predict_proba(customer_scaled)[0][1]
            else:
                probability = predictor.model.predict_proba(customer_df)[0][1]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{probability:.1%}")
            
            with col2:
                risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            
            with col3:
                recommendation = "Immediate Action" if probability > 0.7 else "Monitor" if probability > 0.3 else "Maintain"
                st.metric("Recommendation", recommendation)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability %"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70}}))
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def main():
    # Title and description
    st.markdown('<h1 class="main-header">📱 Telco Customer Churn Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Predict customer churn with machine learning to improve retention strategies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitChurnPredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Navigation")
        
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "💾 Model Management"]
        )
        
        st.markdown("---")
        
        # File upload
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)",
            type=['csv'],
            help="Upload the telco customer churn dataset"
        )
        
        if uploaded_file:
            if st.button("🔄 Load Data"):
                with st.spinner("Loading data..."):
                    df = predictor.load_data(uploaded_file)
                    if df is not None:
                        predictor.preprocess_data()
                        predictor.feature_engineering()
                        st.success("✅ Data loaded successfully!")
                        st.info(f"Dataset shape: {df.shape}")
    
    # Main content based on selected page
    if page == "🏠 Home":
        st.header("Welcome to the Telco Churn Predictor!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="color: #000000">
                <h3 >📊 Data Analysis</h3>
                <p>Explore your customer data with interactive visualizations and insights.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" color: #000000>
                <h3>🤖 Model Training</h3>
                <p>Train machine learning models to predict customer churn accurately.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" color: #000000>
                <h3>🔮 Predictions</h3>
                <p>Make real-time predictions for individual customers.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload Data**: Use the sidebar to upload your telco customer dataset (CSV format)
        2. **Analyze Data**: Go to the Data Analysis page to explore your dataset
        3. **Train Model**: Train machine learning models on the Model Training page
        4. **Make Predictions**: Use the trained model to predict churn for new customers
        5. **Manage Models**: Save and load trained models for future use
        """)
        
        if not uploaded_file:
            st.info("👆 Please upload a dataset to get started!")
    
    elif page == "📊 Data Analysis":
        st.header("📊 Data Analysis & Visualization")
        
        if predictor.df is None:
            st.warning("Please upload a dataset first.")
            return
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(predictor.df))
        with col2:
            churn_rate = predictor.df['Churn'].value_counts(normalize=True)['Yes']
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with col3:
            avg_tenure = predictor.df['tenure'].mean()
            st.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")
        with col4:
            avg_charges = predictor.df['MonthlyCharges'].mean()
            st.metric("Avg. Monthly Charges", f"${avg_charges:.2f}")
        
        st.markdown("---")
        
        # Visualizations
        create_visualizations(predictor.df)
        
        # Data table
        st.subheader("📋 Dataset Preview")
        st.dataframe(predictor.df.head(100), use_container_width=True)
    
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training")
        
        if predictor.df is None:
            st.warning("Please upload a dataset first.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Training Configuration")
            
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"]
            )
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        # Prepare features
                        X, y = predictor.prepare_features()
                        
                        if X is None or y is None:
                            st.error("Error preparing features.")
                            return
                        
                        # Train model
                        results = predictor.train_model(X, y, model_type)
                        st.session_state.results = results
                        predictor.model_type = model_type
                        
                        st.success("✅ Model trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        
        with col2:
            if 'results' in st.session_state:
                results = st.session_state.results
                
                st.subheader("📈 Model Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    st.metric("AUC Score", f"{results['auc_score']:.3f}")
                
                # Performance plots
                create_model_evaluation_plots(results)
                
                # Classification report
                st.subheader("📊 Detailed Performance Report")
                report = classification_report(results['y_test'], results['y_pred'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
    
    elif page == "🔮 Predictions":
        st.header("🔮 Customer Churn Predictions")
        predict_single_customer(predictor)
    
    elif page == "💾 Model Management":
        st.header("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Save Model")
            
            if predictor.model is None:
                st.warning("No trained model to save. Please train a model first.")
            else:
                if st.button("💾 Save Current Model"):
                    if predictor.save_model():
                        st.success("✅ Model saved successfully!")
                    else:
                        st.error("❌ Failed to save model.")
        
        with col2:
            st.subheader("📁 Load Model")
            
            if os.path.exists('churn_model.pkl'):
                if st.button("📁 Load Saved Model"):
                    if predictor.load_model():
                        st.success("✅ Model loaded successfully!")
                    else:
                        st.error("❌ Failed to load model.")
            else:
                st.info("No saved model found.")
        
        st.markdown("---")
        
        st.subheader("ℹ️ Model Information")
        if predictor.model:
            model_type = getattr(predictor, 'model_type', 'Unknown')
            st.write(f"**Model Type**: {model_type}")
            st.write(f"**Features**: {len(predictor.feature_names) if predictor.feature_names else 0}")
            
            # Feature importance (for tree-based models)
            if hasattr(predictor.model, 'feature_importances_') and predictor.feature_names:
                st.subheader("🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': predictor.feature_names,
                    'importance': predictor.model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                fig = px.bar(importance_df, x='importance', y='feature', 
                           orientation='h', title="Top 10 Most Important Features")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model currently loaded.")

if __name__ == "__main__":
    main()
