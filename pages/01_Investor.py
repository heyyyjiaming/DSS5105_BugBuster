import streamlit as st
import os
import glob
import pandas as pd
from llama_parse import LlamaParse
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig
from uniflow.flow.config import OpenAIModelConfig
from uniflow.op.prompt import PromptTemplate, Context
from utils.extract import convert_pdf_to_text, convert_text_to_xlsx, extract_esg_contents, convert_xlsx_to_summary
# from models_test.scoring import ESGModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px



st.title("Investor's ESG Lens 👓")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)

with st.sidebar:
    input_openai_api_key = st.text_input("OpenAI API Key", type="password")
    input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")


if input_openai_api_key:
    os.environ["OPENAI_API_KEY"] = input_openai_api_key
if input_llama_api_key:
    os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api_key





if not (input_openai_api_key and input_llama_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key on the left to continue.", icon="🗝️")
else:
    with st.sidebar:
        company_name = st.text_input("Please enter the name of company you want to analyze")
        st.session_state.uploaded_file = st.file_uploader("Upload a your ESG report(PDF) 📎", type=("pdf"))

    if not st.session_state.uploaded_file:
        st.warning("⬅️ Please upload a PDF file to continue 👻")
    if st.session_state.uploaded_file:
        st.write("File uploaded successfully! 🎈")

        with st.spinner("Extracting text from the PDF file..."):
            st.session_state.doc_parsed = convert_pdf_to_text(st.session_state.uploaded_file)
            
        with st.spinner("Extracting ESG information..."): 
            st.session_state.df_info = convert_text_to_xlsx(st.session_state.doc_parsed)
            # st.dataframe(st.session_state.df_info)
            st.session_state.df_summary = convert_xlsx_to_summary(st.session_state.df_info, company_name)
            st.dataframe(st.session_state.df_summary)
            
        if st.session_state.df_summary:  
            st.markdown("### ESG Summary")
            st.write("Here is the summary of the ESG information extracted from the report.")
            
        
            

   
    







