import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)
hide_streamlit_footer = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)
# Title and description
st.title("🌸 Iris Flower Species Predictor")
st.markdown("""
This app predicts the species of Iris flower based on its measurements.
Enter the flower dimensions below and click **Predict** to see the result!
""")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('iris_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please train the model first.")
        return None

# Load iris data for reference
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris.feature_names, iris.target_names

# Load model and data
model = load_model()
feature_names, target_names = load_iris_data()

# Create input form
st.sidebar.header("📏 Input Flower Measurements")

def user_input_features():
    # Create sliders for input
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)", 
        min_value=4.0, 
        max_value=8.0, 
        value=5.4, 
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)", 
        min_value=2.0, 
        max_value=4.5, 
        value=3.4, 
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.sidebar.slider(
        "Petal Length (cm)", 
        min_value=1.0, 
        max_value=7.0, 
        value=4.0, 
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.sidebar.slider(
        "Petal Width (cm)", 
        min_value=0.1, 
        max_value=2.5, 
        value=1.3, 
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    # Create dataframe with EXACT column names that match the training data
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get input features
input_df = user_input_features()

# Display input values
st.subheader("📊 Input Measurements")
st.write(input_df)

# Make prediction when button is clicked
if st.button("🔮 Predict Species", type="primary"):
    if model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display prediction
            st.subheader("🎯 Prediction Result")
            
            # Handle different prediction formats
            if isinstance(prediction[0], (int, np.integer)):
                # If prediction is numeric
                predicted_class = int(prediction[0])
                predicted_species = target_names[predicted_class]
            else:
                # If prediction is already species name
                predicted_species = str(prediction[0])
                # Find the class index for probability display
                predicted_class = list(target_names).index(predicted_species)
            
            # Create columns for better layout
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                # Display prediction with emoji
                if predicted_species == 'setosa':
                    st.success(f"### 🌿 {predicted_species.upper()}")
                elif predicted_species == 'versicolor':
                    st.info(f"### 🎨 {predicted_species.upper()}")
                else:
                    st.warning(f"### 🌺 {predicted_species.upper()}")
            
            # Display prediction probabilities
            st.subheader("📈 Prediction Probabilities")
            
            # Create dataframe for probabilities
            proba_df = pd.DataFrame({
                'Species': target_names,
                'Probability': prediction_proba[0] * 100
            })
            
            # Display as bar chart
            st.bar_chart(proba_df.set_index('Species'))
            
            # Display as table
            st.dataframe(
                proba_df.style.format({'Probability': '{:.2f}%'}),
                hide_index=True,
                use_container_width=True
            )
            
            # Additional info
            confidence = max(prediction_proba[0]) * 100
            st.info(f"💡 The model is **{confidence:.2f}%** confident that this is a **{predicted_species}** flower.")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Debug info - Prediction value:", prediction)
    else:
        st.error("❌ Model not loaded. Please train the model first!")

# Add information about the dataset
with st.expander("ℹ️ About the Iris Dataset"):
    st.markdown("""
    The **Iris flower dataset** was introduced by Sir Ronald Fisher in 1936.
    
    **Features:**
    - Sepal length (cm)
    - Sepal width (cm)
    - Petal length (cm)
    - Petal width (cm)
    
    **Target Species:**
    - **Setosa** (0) - 🌿
    - **Versicolor** (1) - 🎨
    - **Virginica** (2) - 🌺
    
    The dataset contains 150 samples, 50 from each species.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Iris Classification Model")