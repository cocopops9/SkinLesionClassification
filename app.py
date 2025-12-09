"""
Skin Lesion Classification Application
Features:
- User authentication system
- User-specific image storage
- Explainability layer with Grad-CAM
- Non-skin image validation
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
from pathlib import Path
import uuid
import json
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Import custom modules
from database import DatabaseManager, User, ImageRecord
from classification import MelanomaClassifier
from explainability import ExplainabilityEngine
from image_validator import ImageValidator

# Configure Streamlit
st.set_page_config(
    page_title='Skin Lesion Classification',
    page_icon='🔬',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# Initialize components
@st.cache_resource
def get_classifier():
    """Initialize and cache classifier."""
    return MelanomaClassifier()

@st.cache_resource
def get_validator():
    """Initialize and cache image validator."""
    return ImageValidator()

# Create necessary directories
UPLOAD_DIR = Path('user_uploads')
UPLOAD_DIR.mkdir(exist_ok=True)
HEATMAP_DIR = Path('heatmaps')
HEATMAP_DIR.mkdir(exist_ok=True)


def create_user_directory(user_id: str):
    """Create user-specific directory for uploads."""
    user_dir = UPLOAD_DIR / user_id
    user_dir.mkdir(exist_ok=True)
    return user_dir


def login_page():
    """Display login/registration page."""
    st.title("🔬 Skin Lesion Classification")
    st.markdown("### Secure Medical Image Analysis Platform")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if username and password:
                    user = DatabaseManager.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.page = 'main'
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            new_username = st.text_input("Username", key="reg_username",
                                        help="Minimum 3 characters")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password",
                                        help="Minimum 8 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            terms = st.checkbox("I agree to use this system responsibly and understand it's not for medical diagnosis")
            
            if st.button("Register", type="primary", use_container_width=True):
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.warning("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                elif not terms:
                    st.warning("Please accept the terms")
                else:
                    try:
                        user = DatabaseManager.create_user(new_username, new_password, new_email)
                        if user:
                            st.success("Registration successful! Please login.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Username or email already exists")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
    
    # Information section
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🔒 Secure Storage**\n\nYour images are stored securely and associated with your account")
    
    with col2:
        st.info("**🧠 AI Explainability**\n\nUnderstand why the model made its prediction with visual explanations")
    
    with col3:
        st.info("**✅ Image Validation**\n\nAutomatic detection of non-skin images for accurate analysis")


def main_application():
    """Main application interface for authenticated users."""
    classifier = get_classifier()
    validator = get_validator()
    
    # Sidebar
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.user.username}!")
        
        # Navigation
        st.subheader("Navigation")
        page_selection = st.radio(
            "Select Page",
            ["New Analysis", "History", "Settings", "About"]
        )
        
        st.divider()
        
        # Model selection
        st.subheader("Model Configuration")
        
        model_mode = st.radio(
            "Model Selection Mode",
            ["Single Model", "Ensemble (Multiple)"]
        )
        
        available_models = list(classifier.MODEL_CONFIGS.keys())
        
        if model_mode == "Single Model":
            selected_models = [st.selectbox(
                "Choose Model",
                available_models,
                format_func=lambda x: f"{x} ({classifier.MODEL_CONFIGS[x]['accuracy']*100:.0f}% accuracy)"
            )]
        else:
            selected_models = st.multiselect(
                "Select Models for Ensemble",
                available_models,
                default=["EfficientNetB3", "InceptionV3"],
                format_func=lambda x: f"{x} ({classifier.MODEL_CONFIGS[x]['accuracy']*100:.0f}% accuracy)"
            )
            
            if len(selected_models) < 2:
                st.warning("Please select at least 2 models for ensemble")
        
        # Analysis options
        st.divider()
        st.subheader("Analysis Options")
        show_explainability = st.checkbox("Generate Explainability Report", value=True)
        show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)
        save_to_history = st.checkbox("Save to History", value=True)
        
        # Logout
        st.divider()
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()
    
    # Main content area
    if page_selection == "New Analysis":
        analysis_page(classifier, validator, selected_models, 
                     show_explainability, show_heatmap, save_to_history)
    elif page_selection == "History":
        history_page()
    elif page_selection == "Settings":
        settings_page()
    elif page_selection == "About":
        about_page()


def analysis_page(classifier, validator, selected_models, 
                  show_explainability, show_heatmap, save_to_history):
    """New analysis page for image upload and classification."""
    st.title("🔬 Skin Lesion Analysis")

    # File uploader
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a dermoscopic or clinical image of skin lesion",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="For best results, use dermoscopic images with good lighting and focus"
        )
    
    with col2:
        # Show example
        if st.checkbox("Show example image"):
            example_path = Path("media/melanoma.jpg")
            if example_path.exists():
                example_img = Image.open(example_path)
                st.image(example_img, caption="Example: Melanoma dermoscopic image", width=200)
    
    if uploaded_file is not None and len(selected_models) > 0:
        # Create unique ID for this upload
        image_id = str(uuid.uuid4())
        user_dir = create_user_directory(st.session_state.user.id)
        
        # Save uploaded file
        file_path = user_dir / f"{image_id}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Image", width=400)
        
        # Step 1: Validate image
        with st.spinner("Validating image..."):
            validation_results = validator.validate_image(img)
            validation_report = validator.generate_validation_report(validation_results)

        # Show validation result and handle confirmation if needed
        proceed_with_analysis = False

        if validation_results['is_valid']:
            st.success("✅ Image validation passed")
            proceed_with_analysis = True
        else:
            st.error(validation_report)
            # Add confirmation checkbox
            confirm_proceed = st.checkbox(
                "I confirm this is a valid skin lesion image - proceed with analysis",
                key=f"confirm_{uploaded_file.name}"
            )
            if confirm_proceed:
                proceed_with_analysis = True
                st.info("Proceeding with analysis as requested. Note: predictions may not be reliable.")

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):

            if not proceed_with_analysis:
                st.error("Please confirm the image is valid or upload a different image.")
                # Save invalid image record
                if save_to_history:
                    image_data = {
                        'filename': uploaded_file.name,
                        'file_path': str(file_path),
                        'is_valid_skin_image': False,
                        'validation_message': validation_results['reasons'][0] if validation_results['reasons'] else "Invalid image"
                    }
                    DatabaseManager.save_image_record(st.session_state.user.id, image_data)
                return
            
            # Step 2: Classification
            with st.spinner("Analyzing skin lesion..."):
                try:
                    prediction_data = classifier.get_prediction_with_metadata(
                        str(file_path), selected_models
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    return
            
            # Step 3: Generate explainability
            explanation_text = ""
            heatmap_path = None
            
            if show_explainability:
                with st.spinner("Generating explanation..."):
                    # Get model for Grad-CAM (use first model)
                    model, img_size = classifier.get_model_for_explainability(selected_models[0])
                    
                    # Preprocess image for model
                    img_preprocessed = classifier.preprocess_image(str(file_path), selected_models[0])
                    
                    # Compute Grad-CAM
                    heatmap = ExplainabilityEngine.compute_gradcam(
                        model, img_preprocessed, prediction_data['predicted_class']
                    )
                    
                    # Compute other metrics
                    color_metrics = ExplainabilityEngine.analyze_color_distribution(img_array)
                    asymmetry_score = ExplainabilityEngine.detect_asymmetry(img_array)
                    
                    # Generate explanation
                    scores_array = np.array([prediction_data['confidence_scores'][k] 
                                            for k in ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']])
                    
                    explanation_text = ExplainabilityEngine.generate_explanation(
                        prediction_data['predicted_class'],
                        scores_array,
                        color_metrics,
                        asymmetry_score
                    )
                    
                    if show_heatmap:
                        # Save heatmap
                        heatmap_filename = f"{image_id}_heatmap.npy"
                        heatmap_path = HEATMAP_DIR / heatmap_filename
                        np.save(heatmap_path, heatmap)
            
            # Display results
            st.divider()
            st.subheader("📊 Analysis Results")
            
            # Main diagnosis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Diagnosis",
                    prediction_data['diagnosis'],
                    delta=None
                )
            
            with col2:
                lesion_color = "🔴" if "Malignant" in prediction_data['lesion_type'] else "🟢"
                st.metric(
                    "Lesion Type",
                    f"{lesion_color} {prediction_data['lesion_type']}",
                    delta=None
                )
            
            with col3:
                confidence_pct = prediction_data['confidence'] * 100
                st.metric(
                    "Confidence",
                    f"{confidence_pct:.1f}%",
                    delta=None
                )
            
            # Confidence scores table
            st.subheader("Confidence Scores by Diagnosis")
            scores_df = pd.DataFrame({
                'Diagnosis': list(prediction_data['confidence_scores'].keys()),
                'Confidence (%)': [v * 100 for v in prediction_data['confidence_scores'].values()]
            }).sort_values('Confidence (%)', ascending=False)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['red' if scores_df.iloc[i]['Diagnosis'] == 'MEL' else 'blue' 
                     for i in range(len(scores_df))]
            ax.barh(scores_df['Diagnosis'], scores_df['Confidence (%)'], color=colors, alpha=0.7)
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Prediction Confidence Distribution')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            
            # Explainability section
            if show_explainability and explanation_text:
                st.divider()
                st.subheader("🧠 Explanation")
                st.markdown(explanation_text)
                
                if show_heatmap and heatmap_path and heatmap_path.exists():
                    st.subheader("📍 Important Regions (Grad-CAM)")
                    
                    # Load and display heatmap
                    heatmap = np.load(heatmap_path)
                    overlaid = ExplainabilityEngine.overlay_heatmap(img_array, heatmap)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_array, caption="Original", width=350)
                    with col2:
                        st.image(overlaid, caption="Areas of Focus", width=350)
            
            # Save to database
            if save_to_history:
                with st.spinner("Saving to history..."):
                    image_data = {
                        'filename': uploaded_file.name,
                        'file_path': str(file_path),
                        'diagnosis': prediction_data['diagnosis'],
                        'lesion_type': prediction_data['lesion_type'],
                        'confidence_scores': json.dumps(prediction_data['confidence_scores']),
                        'model_used': prediction_data['model_used'],
                        'is_valid_skin_image': True,
                        'explanation_text': explanation_text,
                        'heatmap_path': str(heatmap_path) if heatmap_path else None,
                        'processing_time': prediction_data['processing_time'],
                        'image_width': img_array.shape[1],
                        'image_height': img_array.shape[0]
                    }
                    
                    record = DatabaseManager.save_image_record(
                        st.session_state.user.id, image_data
                    )
                    
                    st.success(f"✅ Analysis saved to history (ID: {record.id[:8]}...)")
            
            # Processing time
            st.divider()
            st.caption(f"⏱️ Total processing time: {prediction_data['processing_time']:.2f} seconds")
            st.caption(f"🤖 Models used: {prediction_data['model_used']}")


def history_page():
    """Display user's analysis history."""
    st.title("📚 Analysis History")
    
    # Get user's images
    images = DatabaseManager.get_user_images(st.session_state.user.id, limit=50)
    
    if not images:
        st.info("No analysis history yet. Upload an image to get started!")
        return
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2, col3 = st.columns(3)

    total_analyses = len(images)
    valid_analyses = sum(1 for img in images if img.is_valid_skin_image)
    malignant_count = sum(1 for img in images if img.lesion_type and "Malignant" in img.lesion_type)

    with col1:
        st.metric("Total Analyses", total_analyses)
    with col2:
        st.metric("Valid Images", valid_analyses)
    with col3:
        st.metric("Malignant Findings", malignant_count)

    st.divider()
    
    # History table
    st.subheader("Recent Analyses")
    
    # Create dataframe
    history_data = []
    for img in images:
        history_data.append({
            'ID': img.id[:8],
            'Date': img.upload_time.strftime("%Y-%m-%d %H:%M"),
            'Filename': img.filename,
            'Diagnosis': img.diagnosis or "N/A",
            'Type': img.lesion_type or "N/A",
            'Valid': "✅" if img.is_valid_skin_image else "❌",
            'Model': img.model_used or "N/A"
        })
    
    df = pd.DataFrame(history_data)

    # Display the table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Selection dropdown
    st.divider()
    st.subheader("Select an Analysis to View/Manage")

    # Create selection options
    selection_options = ["-- Select an analysis --"] + [
        f"{img.id[:8]} - {img.filename} ({img.upload_time.strftime('%Y-%m-%d %H:%M')})"
        for img in images
    ]

    selected_option = st.selectbox(
        "Choose an analysis:",
        selection_options,
        key="history_selection"
    )

    # Show details for selected item
    if selected_option != "-- Select an analysis --":
        selected_idx = selection_options.index(selected_option) - 1  # -1 for the placeholder
        selected_image = images[selected_idx]

        st.divider()
        st.subheader(f"Details for Analysis {selected_image.id[:8]}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Load and display image if exists
            if Path(selected_image.file_path).exists():
                img = Image.open(selected_image.file_path)
                st.image(img, caption=selected_image.filename, width=300)
            else:
                st.warning("Image file not found")
        
        with col2:
            if selected_image.is_valid_skin_image and selected_image.diagnosis:
                st.write(f"**Diagnosis:** {selected_image.diagnosis}")
                st.write(f"**Lesion Type:** {selected_image.lesion_type}")
                st.write(f"**Model Used:** {selected_image.model_used}")
                st.write(f"**Processing Time:** {selected_image.processing_time:.2f}s")

                # Show confidence scores
                if selected_image.confidence_scores:
                    scores = selected_image.get_confidence_scores()
                    st.write("**Confidence Scores:**")
                    for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                        st.write(f"  • {k}: {v*100:.1f}%")
            else:
                st.error(f"Invalid image: {selected_image.validation_message}")

        # Show explanation if available
        if selected_image.explanation_text:
            st.divider()
            st.subheader("Explanation")
            st.markdown(selected_image.explanation_text)

        # Management actions
        st.divider()
        st.subheader("Actions")

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            # Rename functionality
            new_name = st.text_input(
                "Rename file",
                value=selected_image.filename,
                key=f"rename_{selected_image.id}"
            )
            if st.button("Rename", key=f"rename_btn_{selected_image.id}"):
                if new_name and new_name != selected_image.filename:
                    success = DatabaseManager.update_image_filename(selected_image.id, new_name)
                    if success:
                        st.success(f"Renamed to: {new_name}")
                        st.rerun()
                    else:
                        st.error("Failed to rename")

        with col_action2:
            # Delete functionality
            if st.button("🗑️ Delete", key=f"delete_btn_{selected_image.id}", type="secondary"):
                st.session_state[f"confirm_delete_{selected_image.id}"] = True

            if st.session_state.get(f"confirm_delete_{selected_image.id}", False):
                st.warning("Are you sure you want to delete this analysis?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, delete", key=f"confirm_yes_{selected_image.id}"):
                        success = DatabaseManager.delete_image_record(selected_image.id)
                        if success:
                            st.success("Deleted successfully")
                            del st.session_state[f"confirm_delete_{selected_image.id}"]
                            # Reset selection to update statistics
                            if "history_selection" in st.session_state:
                                del st.session_state["history_selection"]
                            st.rerun()
                        else:
                            st.error("Failed to delete")
                with col_no:
                    if st.button("Cancel", key=f"confirm_no_{selected_image.id}"):
                        del st.session_state[f"confirm_delete_{selected_image.id}"]
                        st.rerun()


def settings_page():
    """User settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Account Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Username", value=st.session_state.user.username, disabled=True)
        st.text_input("Email", value=st.session_state.user.email, disabled=True)
    
    with col2:
        st.text_input("User ID", value=st.session_state.user.id[:8] + "...", disabled=True)
        st.text_input("Member Since", value=st.session_state.user.created_at.strftime("%Y-%m-%d"), disabled=True)
    
    st.divider()
    
    st.subheader("Privacy & Data")
    
    if st.button("Clear Analysis History", type="secondary"):
        if st.checkbox("I understand this action cannot be undone"):
            # Implementation would delete user's image records
            st.success("History cleared (not implemented in demo)")
    
    st.divider()
    
    st.subheader("Export Data")
    if st.button("Export My Data (CSV)", type="primary"):
        # Export user data to CSV
        images = DatabaseManager.get_user_images(st.session_state.user.id)
        if images:
            data = []
            for img in images:
                data.append({
                    'Date': img.upload_time,
                    'Filename': img.filename,
                    'Diagnosis': img.diagnosis,
                    'Type': img.lesion_type,
                    'Valid': img.is_valid_skin_image
                })
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"melanoma_history_{st.session_state.user.username}.csv",
                mime="text/csv"
            )


def about_page():
    """About page with system information."""
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Skin Lesion Classification
    
    This advanced medical image analysis system provides:
    
    ### 🔬 Key Features
    
    1. **Multi-Model Ensemble Learning**
       - Support for 6 state-of-the-art CNN architectures
       - Weighted voting for improved accuracy
       - Individual and ensemble predictions
    
    2. **Explainable AI**
       - Grad-CAM visualization for decision transparency
       - Detailed textual explanations
       - Feature importance analysis
    
    3. **Robust Image Validation**
       - Automatic detection of non-skin images
       - Statistical and CNN-based validation
       - Comprehensive validation reports
    
    4. **Secure User Management**
       - Individual user accounts with bcrypt password hashing
       - User-specific image storage
       - Complete analysis history
    
    ### 📊 Supported Diagnoses
    
    - **AKIEC**: Actinic Keratosis / Bowen's disease (Pre-malignant/Malignant)
    - **BCC**: Basal Cell Carcinoma (Malignant)
    - **BKL**: Benign Keratosis (Benign)
    - **DF**: Dermatofibroma (Benign)
    - **MEL**: Melanoma (Malignant)
    - **NV**: Melanocytic Nevus (Benign)
    - **VASC**: Vascular Lesion (Benign/Malignant)
    
    ### 🔒 Data Privacy
    
    - All user data is stored locally
    - Images are associated with user accounts
    - No data sharing with third parties
    
    ### ⚠️ Important Notice
    
    **This system is for research and educational purposes only.**
    
    It should NOT be used for medical diagnosis. Always consult with qualified healthcare 
    professionals for medical advice. The predictions provided are based on machine learning 
    models and may not be accurate.
    
    ### 📈 Technical Specifications
    
    - **Framework**: TensorFlow 2.9+
    - **Models**: EfficientNetB3, InceptionV3, ResNet50, InceptionResNetV2, DenseNet201, NASNet
    - **Database**: SQLAlchemy with SQLite
    - **Security**: bcrypt password hashing
    - **Explainability**: Grad-CAM, Occlusion Sensitivity
    - **Validation**: MobileNetV2 + Statistical Analysis
    
    ### 👥 Credits
    
    Enhanced version with authentication, explainability, and validation features.
    Based on the HAM10000 dataset for skin lesion classification.
    """)


# Main application flow
def main():
    """Main application entry point."""
    if not st.session_state.authenticated:
        login_page()
    else:
        main_application()


if __name__ == "__main__":
    main()
