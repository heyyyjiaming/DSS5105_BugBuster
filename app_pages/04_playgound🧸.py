import streamlit as st
import os
import glob
import pandas as pd
from llama_parse import LlamaParse
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig
from uniflow.flow.config import OpenAIModelConfig
from uniflow.op.prompt import PromptTemplate, Context
from utils.extract import convert_pdf_to_text, convert_text_to_xlsx, extract_esg_contents, convert_xlsx_to_summary, append_to_summary
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import plotly.express as px



def init_session():
    if 'pdf_texts' not in st.session_state:
        st.session_state.pdf_texts = None
    if 'df_info' not in st.session_state:
        st.session_state.df_info = None
    if 'df_summary' not in st.session_state:
        st.session_state.df_summary = None
        
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


st.title("🪀Playground for ESG Analysis Tool")
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
        st.session_state.uploaded_path = st.file_uploader("Upload a your ESG report(PDF) 📎", type=("pdf"), accept_multiple_files=True)

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
        st.session_state.uploaded_path = st.file_uploader("Upload a your ESG report(PDF) 📎", type=("pdf"), accept_multiple_files=True)
    if not st.session_state.uploaded_path:
        st.warning("⬅️ Please upload a PDF file to continue 👻")
    else:
        st.write("File uploaded successfully! 🎉")
        init_session()
        
    if not st.session_state.uploaded_path:
        st.warning("⬅️ Please upload a PDF file to continue 👻")
    else:
        st.write("File uploaded successfully! 🎉")
        init_session()
        
        with st.form(key='extraction_form'):
            # st.session_state.df_info = None
            # st.session_state.df_summary = None
            start = st.form_submit_button("Strat to analyze")
            # st.session_state.df_summary = None
            if start:
                progress_text = "Processing PDF file "
                my_bar = st.progress(0, text=progress_text)
                for i in range(len(st.session_state.uploaded_path)):
                    my_bar.progress(1/len(st.session_state.uploaded_path)*i, text=progress_text+f"{i+1}")
                    
                    with st.spinner("Converting PDF file into text..."):
                        cur_doc_parsed, cur_pdf_texts = convert_pdf_to_text(st.session_state.uploaded_path[i])
                    with st.spinner("Extracting ESG information..."):
                        df_info = convert_text_to_xlsx(cur_doc_parsed)
                        st.dataframe(df_info)
                        # df_summary = convert_xlsx_to_summary(df_info, company_name)
                    
                    with st.spinner("Extracting ESG information..."):
                        df_info = convert_text_to_xlsx(cur_doc_parsed)
                        st.dataframe(df_info)
                    if st.session_state.df_info is not None:
                        st.session_state.df_info = df_info
                    else:
                        st.session_state.df_info = pd.concat([st.session_state.df_info, df_info])
                        st.dataframe(st.session_state.df_info)
                my_bar.empty()
                            
                            # if (st.session_state.df_summary is not None) and (df_summary is not None):
                            #     st.session_state.df_summary = df_summary
                            # else:
                            #     st.session_state.df_summary = append_to_summary(st.session_state.df_summary, df_summary)
                            # st.dataframe(st.session_state.df_summary)
                            
                                
                            # st.session_state.df_info = convert_text_to_xlsx(st.session_state.doc_parsed)
                            # st.session_state.df_summary = convert_xlsx_to_summary(st.session_state.df_info, company_name)
                        # st.session_state.doc_parsed, st.session_state.pdf_texts = convert_pdf_to_text(st.session_state.uploaded_file)
                        
                    # with st.spinner("Extracting ESG information..."): 
                    #     st.session_state.df_info = convert_text_to_xlsx(st.session_state.doc_parsed)
                    #     st.session_state.df_summary = convert_xlsx_to_summary(st.session_state.df_info, company_name) 


        if st.session_state.df_summary is not None:
            st.markdown("#### Summary of Extracted ESG Related Data:")      
            st.dataframe(st.session_state.df_summary)
            df_summary_csv= convert_df(st.session_state.df_summary)
            st.download_button(
                label="Download Extracted ESG Data Summary",
                data=df_summary_csv,
                file_name=f"{company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
                mime="text/csv",)
        
            df_info_button = st.toggle("View Raw Extracted ESG Data 👇🏻")
            if df_info_button:
                st.markdown("**Below is the raw ESG data extracted from the report:**")
                st.dataframe(st.session_state.df_info)
                df_info_csv= convert_df(st.session_state.df_info)
                st.download_button(
                    label="Download Raw Extracted ESG Data as CSV",
                    data=df_info_csv,
                    file_name=f"{company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
                    mime="text/csv",)
                
            pdf_texts_button = st.toggle("View Converted ESG Report Texts 👇🏻")
            if pdf_texts_button:
                st.markdown("**Texts of uploaded ESG report:**")
                with st.container(height=600):
                    st.markdown(st.session_state.pdf_texts)
 

                    
        if st.session_state.df_summary is not None:
            st.markdown("### ESG Summary")
            st.write("Here is the summary of the ESG information extracted from the report.")
                    
            st.markdown("#### You could find more ESG related reports from the following sources:")
            input_serp_api_key = st.text_input("Serp API Key", type="password")
                    
            if not input_serp_api_key:
                st.info("Please add your Serp API key to continue.", icon="🗝️")
            else:
                os.environ["SERP_API_KEY"] = input_serp_api_key
                st.session_state.news_df = get_esg_news(company_name, input_serp_api_key)
                st.dataframe(st.session_state.news_df, 
                             column_config={"link": st.column_config.LinkColumn()})
