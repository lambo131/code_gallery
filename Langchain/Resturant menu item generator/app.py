import streamlit as st
import functions as func

left, nn,nm, right = st.columns(4)
right.text("by Lambo Qin")
'''
# Resturant name generator :green_salad:
'''

text = st.text_input("cruisine type", key="cruisine_type")
generator = func.generator()

if st.button("generate"):
    if text == "":
        st.write("> please specify cruisine type")
    else:
        st.write(generator.get_ans(text))