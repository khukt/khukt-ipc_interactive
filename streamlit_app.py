import streamlit as st

# App title
st.title("🚀 Hello Streamlit Cloud")

# Text input
name = st.text_input("Enter your name:")

# Button
if st.button("Say Hello"):
    if name:
        st.success(f"Hello, {name}! 👋 Welcome to Streamlit Cloud.")
    else:
        st.warning("Please enter your name first.")

# Slider example
age = st.slider("Select your age:", 1, 100, 25)
st.write(f"Your selected age is: {age}")
