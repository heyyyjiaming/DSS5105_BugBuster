import streamlit as st
import os
# import glob
import pandas as pd
from llama_parse import LlamaParse
from utils.extract import convert_pdf_to_text, convert_text_to_xlsx, convert_xlsx_to_summary, append_to_summary
from utils.external import get_esg_news
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import plotly.express as px


st.cache_data.clear()
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

os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]
os.environ["SERP_API_KEY"] = st.secrets["SERP_API_KEY"]


with st.sidebar:
    input_openai_api_key = st.text_input("OpenAI API Key", type="password")
    # input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")

if input_openai_api_key:
    os.environ["OPENAI_API_KEY"] = input_openai_api_key
# if input_llama_api_key:
#     os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api_key



# if not (input_openai_api_key and input_llama_api_key):
if not (input_openai_api_key):
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
        
        with st.form(key='extraction_form'):
            start = st.form_submit_button("Strat to analyze")

            if start:
                st.session_state.df_info = None
                st.session_state.df_summary = None
                
                progress_text = "Processing PDF file "
                my_bar = st.progress(0, text=progress_text)
                for i in range(len(st.session_state.uploaded_path)):
                    my_bar.progress(1/len(st.session_state.uploaded_path)*i, text=progress_text+f"{i+1} ...")
                    
                    with st.spinner("Converting PDF file into text..."):
                        cur_doc_parsed, cur_pdf_texts = convert_pdf_to_text(st.session_state.uploaded_path[i])
                        # st.markdown(cur_pdf_texts)
                    
                    with st.spinner("Extracting ESG information..."):
                        df_info = convert_text_to_xlsx(cur_doc_parsed)
                        df_summary = convert_xlsx_to_summary(df_info, company_name)
                        # st.dataframe(df_info)
                    if st.session_state.df_info is None:
                        st.session_state.df_info = df_info
                    else:
                        existing_df_info = st.session_state.df_info.copy()
                        st.session_state.df_info = pd.concat([existing_df_info, df_info], axis=0)
                        # st.dataframe(st.session_state.df_info)
                    if st.session_state.df_summary is None:
                        st.session_state.df_summary = df_summary
                    else:
                        existing_df_summary = st.session_state.df_summary.copy()
                        st.session_state.df_summary = append_to_summary(existing_df_summary, df_summary)
                my_bar.empty()
                
                # st.markdown("ESG Data Extracted From the Uploaded Reports:")
                # st.dataframe(st.session_state.df_info)
                # st.markdown(f"ESG Data Summary of {company_name}:")
                # st.dataframe(st.session_state.df_summary)                    



        if st.session_state.df_summary is not None:
            st.markdown("ESG Data Extracted From the Uploaded Reports:")
            st.dataframe(st.session_state.df_info)
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
                
            # pdf_texts_button = st.toggle("View Converted ESG Report Texts 👇🏻")
            # if pdf_texts_button:
            #     st.markdown("**Texts of uploaded ESG report:**")
            #     with st.container(height=600):
            #         st.markdown(st.session_state.pdf_texts)
 

                    
        if st.session_state.df_summary is not None:
            st.markdown("### ESG Summary")
            st.write("Here is the summary of the ESG information extracted from the report.")
            
            
            ############################## Summary ###############################
                    
            st.markdown("#### You could find more ESG related reports from the following sources:")
            # input_serp_api_key = st.text_input("Serp API Key", type="password")
                    
            # if not input_serp_api_key:
            #     st.info("Please add your Serp API key to continue.", icon="🗝️")
            # else:
            #     os.environ["SERP_API_KEY"] = input_serp_api_key
            #     st.session_state.news_df = get_esg_news(company_name, input_serp_api_key)
            #     st.dataframe(st.session_state.news_df, 
            #                  column_config={"link": st.column_config.LinkColumn()})

            input_serp_api_key = os.environ["SERP_API_KEY"]
            st.session_state.news_df = get_esg_news(company_name, input_serp_api_key)
            st.dataframe(st.session_state.news_df, 
                        column_config={"link": st.column_config.LinkColumn()})
