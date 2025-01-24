import streamlit as st
from streamlit_option_menu import option_menu
import Predict as pred

# Streamlit configuration
st.set_page_config(page_title="RiceLens - Rice Varieties Classification", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS styles
def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            font-family: Oswald, sans-serif;
            background-color: #f8f3e9;
            color: #5b4636;
            margin: 0;
            padding: 0;
        }
        .stButton > button {
            background-color: #f8f3e9;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton > button:hover {
            background-color: #5d6216;
            color: white;
        }
        footer {
            text-align: center;
            padding: 1rem;
            background-color: #a0522d;
            color: white;
        }
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 90px;
            z-index: 1000;
            background: #ffeec2; 
            color: white;
            padding: 1rem 0;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header img {
            width: 120px;
            margin: 0 auto;
        }
        .welcome-section {
            margin-top: 20px;
            text-align: center;
            padding: 2rem 1rem;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 238, 194, 0.5);
            color: white;
        }
        .welcome-section h1 {
            font-size: 5rem;
            margin-bottom: 0.5rem;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        }
        .welcome-section p {
            font-size: 1.2rem;
            color: black;
        }
        .text-section {
            margin-top: 20px;
            text-align: center;
            padding: 2rem 1rem;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 238, 194, 0.5);
            color: white;
        }
        .upload-section {
            display: flex;
            justify-content: center;
        }
        .upload-section .stFileUploader {
            width: 400px;
            backgorund-color: #5d6216;

        }
        /* Hide Streamlit header and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply styles
apply_custom_css()

st.markdown(
        """
        <div class="header">
            <img src="https://i.ibb.co/cLkBhJ0/RiceLens.png" class="logo" alt="RiceLens Logo">
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "About"],
        icons=["house-door", "question-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#ffeec2"},
            "icon": {"color": "black", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "font-family": "Arial, sans-serif",
                "text-align": "left",
                "margin": "0px",
                "color": "black",  # Make the unselected nav links black
                "--hover-color": "#5d6216",
            },
            "nav-link-selected": {"background-color": "#ffeec2", "color": "black"},
        },
    )


# Home Page
if selected == "Home":


    st.markdown(
        """
        <style>
        html, body {
            margin: 0;
            padding: 0;
            overflow-x: hidden; /* Prevent horizontal scrolling */
        }
        .full-page {
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("https://i.ibb.co/ZfC9tkG/bg.png") no-repeat center center fixed;
            background-size: cover;
            height: 100vh;
            width: 100vw;
            position: fixed; /* Ensures it doesn't scroll with content */
            top: 0;
            left: 0;
            display: flex; /* Use flexbox for centering */
            justify-content: center; /* Horizontally center content */
            align-items: center; /* Vertically center content */
        }

        </style>
        <div class="full-page">
        </div>
        """,
        unsafe_allow_html= True
    )

    # Welcome and description section below header
    st.markdown(
        """
        <div class="welcome-section">
            <h1>Welcome to RiceLens</h1>
            <p>Take a picture or upload a photo of the rice variety you'd like to classify, and let our AI model do the work!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Upload Section with File Uploader and Camera Input
    st.markdown(
        """
        <div style="text-align: left; margin-top: 1rem;">
            <h3 style="margin-bottom: 0; font-weight: bold; margin-top: 0;">Select which option to classify:</h3>
            <div class="upload-section" style="margin-top: 0;">
                <div style="width: 100px;">  <!-- Set the width to 100px -->
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .take-picture-btn {
            background-color: #5b4636; /* Button color */
            color: white; /* Text color */
            font-size: 18px; /* Text size */
            padding: 16px 32px; /* Increased padding for larger height */
            border-radius: 8px; /* Rounded corners */
            border: none;
            cursor: pointer;
            width: 100%; /* Make it full width */
            text-align: center; /* Center text horizontally */
        }
        .take-picture-btn:hover {
            background-color: #3e2c1c; /* Hover effect */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Session state to track if the camera has been opened
    if 'camera_open' not in st.session_state:
        st.session_state['camera_open'] = False

    # File Uploader
    st.markdown("<p style='font-size: 18px; font-weight: bold; margin-bottom: 0px;'>Choose a file</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Open Camera Button
    if not st.session_state['camera_open']:
        st.markdown("<p style='font-size: 18px; font-weight: bold;margin-bottom: 27px;'>Open Camera</p>", unsafe_allow_html=True)
        if st.button('Take a Picture', key="take_picture", help="Click to open camera"):
            # Set session state immediately when the button is pressed
            st.session_state['camera_open'] = True
            st.rerun()  # Trigger a rerun after state change
    else:
        # Show the camera input when the button is clicked
        st.markdown("<p style='font-size: 18px; font-weight: bold;margin-bottom: 27px;'>Capture your Image</p>", unsafe_allow_html=True)
        camera_image = st.camera_input("Capture your Image", label_visibility="collapsed")
        
        if camera_image is not None:
            st.image(camera_image, caption="Captured Image", use_container_width=True)
        
        # Button to "close" the camera (reset the state)
        if st.button('Close Camera'):
            st.session_state['camera_open'] = False
            st.rerun()  # Trigger a rerun after state change

    # Display the uploaded file image below
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Classify"):
            st.write("Classifying...")
            predict = pred.Predict(image=uploaded_file)
            predicted_class, confidence = predict.predict()
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence Value: {confidence}")



# About Page
if selected == "About":
    st.markdown(
        """
        <style>
        html, body {
            margin: 0;
            padding: 0;
            overflow-x: hidden; /* Prevent horizontal scrolling */
        }
        .full-page {
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("https://i.ibb.co/ZfC9tkG/bg.png") no-repeat center center fixed;
            background-size: cover;
            height: 100vh;
            width: 100vw;
            position: fixed; /* Ensures it doesn't scroll with content */
            top: 0;
            left: 0;
            display: flex; /* Use flexbox for centering */
            justify-content: center; /* Horizontally center content */
            align-items: center; /* Vertically center content */
        }
        .about-section {
            margin-top: 0px;
            margin-bottom: 20px;
            text-align: center;
            padding: 2rem 1rem;
            border-radius: 10px;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 238, 194, 0.5);
            color: black;
        }
        </style>
        <div class="full-page">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-section">
           <h1>About Us</h1>
           <p style="text-align: justify;">RiceLens is a cutting-edge AI-powered application designed to classify different rice varieties with precision and ease. By uploading a rice image or capturing one using the app's built-in camera feature, users can identify the rice type (e.g., Arborio, Basmati, Jasmine) along with an accuracy score. The app combines advanced machine learning algorithms with an intuitive user interface, making it a perfect tool for farmers, researchers, and consumers to gain insights into rice classification.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="about-section" style="text-align: center;">
                <img src="https://i.ibb.co/52dBMgM/jaims-1.jpg" alt="Jaime Emanuel B. Lucero" style="border-radius: 50%; width: 150px; height: 150px;">
                <h3 style="margin: 1rem 0 0.5rem; color: black;">Jaime Emanuel B. Lucero</h3>
                <p style="color: black; margin-bottom: 0.5rem;">
                    Bachelor of Science in Computer Science, Major in Data Science<br>
                    University of Southeastern Philippines
                </p>
                <p>
                    <strong>Email:</strong> jeblucero00111@usep.edu.ph<br>
                    <strong>Socials:</strong> 
                    <a href="https://facebook.com/jaime" target="_blank" style="color: #5b4636;">Facebook</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="about-section" style="text-align: center;">
                <img src="https://i.ibb.co/RcPMj3J/linds-1.jpg" alt="Lindsay B. Cañete" style="border-radius: 50%; width: 150px; height: 150px;">
                <h3 style="margin: 1rem 0 0.5rem; color: black;">Lindsay B. Cañete</h3>
                <p style="color: black; margin-bottom: 0.5rem;">
                    Bachelor of Science in Computer Science, Major in Data Science<br>
                    University of Southeastern Philippines
                </p>
                <p>
                    <strong>Email:</strong> lbcanete00090@usep.edu.ph<br>
                    <strong>Socials:</strong> 
                    <a href="https://facebook.com/developer" target="_blank" style="color: #5b4636;">Facebook</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

