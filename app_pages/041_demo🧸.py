import streamlit as st
import os
# import glob
import pandas as pd
import requests
from llama_parse import LlamaParse
from utils.extract import convert_pdf_to_text, convert_text_to_xlsx, extract_esg_contents, convert_xlsx_to_summary, append_to_summary
from utils.external import get_stock_data,  get_esg_news
from io import StringIO, BytesIO
import time
import pickle
from model.scoring import ESG_trend, ESG_trend_plot, company_scoring
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
import plotly.express as px


def load_github_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text), header=0)
    else:
        st.text(response.status_code)
        st.error("Failed to load data from GitHub.")
    return data

os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]
os.environ["SERP_API_KEY"] = st.secrets["SERP_API_KEY"]


def load_github_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.load(BytesIO(response.content))
    else:
        st.error(f"{response.status_code}Failed to load model from GitHub.")
        
    return model


st.cache_data.clear()
def init_session():
    if 'pdf_texts' not in st.session_state:
        st.session_state.pdf_texts = None
    if 'df_info' not in st.session_state:
        st.session_state.df_info = None
    if 'df_summary' not in st.session_state:
        st.session_state.df_summary = None
    if 'esg_score' not in st.session_state:
        st.session_state.esg_score = None
        
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


singtel_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/data/Singtel_ESG_test.xlsx"
response_singtel = requests.get(singtel_url)
if response_singtel.status_code == 200:
    singtel_data = pd.read_excel(BytesIO(response_singtel.content), engine='openpyxl', header=0)
else:
    st.error("Failed to load data from GitHub.")




st.title("ESGeunius 🪄")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)
init_session()

with st.sidebar:
    input_openai_api_key = st.text_input("OpenAI API Key", type="password")
    # input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")

if input_openai_api_key:
    os.environ["OPENAI_API_KEY"] = input_openai_api_key
# if input_llama_api_key:
#     os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api_key



if not input_openai_api_key:
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
                
                progress_text = "Processing report  "
                my_bar = st.progress(0, text=progress_text)
                for i in range(len(st.session_state.uploaded_path)):
                    my_bar.progress(1/len(st.session_state.uploaded_path)*i, text=progress_text+f"{i+1} ...")
                    
                    with st.spinner("Converting PDF file into text..."):
                        time.sleep(3)
                        # cur_doc_parsed, cur_pdf_texts = convert_pdf_to_text(st.session_state.uploaded_path[i])
                        # st.markdown(cur_pdf_texts)
                    
                    with st.spinner("Extracting ESG information..."):
                        time.sleep(5)
                        # df_info = convert_text_to_xlsx(cur_doc_parsed)
                        # df_summary = convert_xlsx_to_summary(df_info, company_name)
                        # st.dataframe(df_info)
                    # if st.session_state.df_info is None:
                    #     st.session_state.df_info = df_info
                    # else:
                    #     existing_df_info = st.session_state.df_info.copy()
                    #     st.session_state.df_info = pd.concat([existing_df_info, df_info], axis=0)
                    #     # st.dataframe(st.session_state.df_info)
                    # if st.session_state.df_summary is None:
                    #     st.session_state.df_summary = df_summary
                    # else:
                    #     existing_df_summary = st.session_state.df_summary.copy()
                    #     st.session_state.df_summary = append_to_summary(existing_df_summary, df_summary)
                st.session_state.df_summary = singtel_data
                my_bar.empty()
                
                # st.markdown("ESG Data Extracted From the Uploaded Reports:")
                # st.dataframe(st.session_state.df_info)
                # st.markdown(f"ESG Data Summary of {company_name}:")
                # st.dataframe(st.session_state.df_summary)                    


        
if st.session_state.df_summary is not None:
    # st.markdown("ESG Data Extracted From the Uploaded Reports:")
    # st.dataframe(st.session_state.df_info)
    st.markdown("##### Summary of Extracted ESG Related Data:")      
    st.dataframe(st.session_state.df_summary)
    df_summary_csv= convert_df(st.session_state.df_summary)
    st.download_button(
        label="Download Extracted ESG Data Summary",
        data=df_summary_csv,
        file_name=f"{company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
        mime="text/csv",)

         
         
                    
if st.session_state.df_summary is not None:
    st.header("ESG Summary")
    st.write("Here is the summary of the ESG information extracted from the report.")
    
    
    ############################## Summary ###############################
    
    scored_esg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/scored_tech_industry_esg_data.csv"   
    scored_tech_esg = load_github_csv(scored_esg_url)    
    # response_tech = requests.get(scored_esg_url)
    # if response_tech.status_code == 200:
    #     scored_tech_esg = pd.read_csv(StringIO(response_tech.text), header=0)
    # else:
    #     st.text(response_tech.status_code)
    #     st.error("Failed to load data from GitHub.")




    ESG_score_trend, esg_industry_plot_data = ESG_trend(scored_tech_esg)
    fig_esg_trend = ESG_trend_plot(esg_industry_plot_data)
    st.markdown("##### Trend of ESG Performance in Tech Industry")    
    st.plotly_chart(fig_esg_trend)
    
    
    # Load Reg Model
    # with open("../model/cluster_model.pkl", "rb") as f:
    #     cluster_model = pickle.load(f)
    # with open("../model/scoring_model.pkl", "rb") as f:
    #     scoring_model = pickle.load(f)
    # st.session_state.esg_socre = scoring_model.predict(st.session_state.df_summary)
    
    esg_cluster_centers_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/tech_esg_cluster_centers.csv"
    esg_cluster_centers = load_github_csv(esg_cluster_centers_url)
    
    cluster_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/cluster_model.pkl"
    cluster_model = load_github_model(cluster_url)
        
    reg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/scoring_model.pkl"
    scoring_model = load_github_model(reg_url)

    
    compare_fig = company_scoring(scored_tech_esg, st.session_state.df_summary, cluster_model, esg_cluster_centers, scoring_model, esg_industry_plot_data, ESG_score_trend)
    # st.text(company_scoring(scored_tech_esg, st.session_state.df_summary, cluster_model, esg_cluster_centers, scoring_model, esg_industry_plot_data, ESG_score_trend))
    st.plotly_chart(compare_fig)
    
    
    ## External Data        
    st.markdown("#### You could find more ESG related reports from the following sources:")
    input_serp_api_key = os.environ["SERP_API_KEY"]

    if not input_serp_api_key:
        st.info("Please add your Serp API key to continue.", icon="🗝️")
        input_serp_api_key = st.text_input("Serp API Key", type="password")
        st.session_state.news_df = get_esg_news(company_name, input_serp_api_key)
        st.dataframe(st.session_state.news_df, 
                        column_config={"link": st.column_config.LinkColumn()})
    else:
        input_serp_api_key = os.environ["SERP_API_KEY"]
        st.session_state.news_df = get_esg_news(company_name, input_serp_api_key)
        st.dataframe(st.session_state.news_df, 
                        column_config={"link": st.column_config.LinkColumn()})

    st.header("Financial Analysis")
    company_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/tests/FinancialData/company_ticker_mapping.csv"
    response = requests.get(company_url)
    if response.status_code == 200:
        company_name_mapping = pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load mapping table of company name from GitHub.")

    stock_price = get_stock_data(company_name, company_name_mapping)
    if stock_price is not None:
        fig = px.line(stock_price, x='Date', y='Adj Close', title=f"{company_name} Stock Price")
        st.plotly_chart(fig)
    else:
        st.warning("No available stock price data. 🙁")