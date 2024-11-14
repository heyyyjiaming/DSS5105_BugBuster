import streamlit as st
import os
import glob
import pandas as pd
from llama_parse import LlamaParse
# from uniflow.flow.client import TransformClient
# from uniflow.flow.config import TransformOpenAIConfig
# from uniflow.flow.config import OpenAIModelConfig
# from uniflow.op.prompt import PromptTemplate, Context
from utils.extract import convert_pdf_to_text, convert_text_to_xlsx, convert_xlsx_to_summary, append_to_summary
from utils.external import get_stock_data,  get_esg_news
from model.scoring import ESG_trend, ESG_trend_plot, company_scoring
from model.finance_eval import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
from yahooquery import search
import requests
import pickle
from io import StringIO, BytesIO

st.set_page_config(layout="wide")


st.cache_data.clear()
def init_session():
    if 'company_name' not in st.session_state:
        st.session_state.company_name = None
    if 'pdf_texts' not in st.session_state:
        st.session_state.pdf_texts = None
    if 'df_info' not in st.session_state:
        st.session_state.df_info = None
    if 'df_summary' not in st.session_state:
        st.session_state.df_summary = None
    if 'esg_score' not in st.session_state:
        st.session_state.esg_score = None
        
def load_github_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text), header=0)
    else:
        st.text(response.status_code)
        st.error("Failed to load data from GitHub.")
    return data


def load_github_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.load(BytesIO(response.content))
    else:
        st.error(f"{response.status_code}Failed to load model from GitHub.")
        
    return model

        
os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]
os.environ["SERP_API_KEY"] = st.secrets["SERP_API_KEY"]
        
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

init_session()
st.title("ESGenius 🪄")
st.header("Investor's ESG Lens 👓")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)

with st.sidebar:
    input_openai_api_key = st.text_input("OpenAI API Key", type="password")
    # input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")

if input_openai_api_key:
    os.environ["OPENAI_API_KEY"] = input_openai_api_key
# if input_llama_api_key:
#     os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api_key


# if not input_openai_api_key:
#     st.info("Please add your OpenAI API key on the left to continue.", icon="🗝️")
# else:
#     with st.sidebar:
#         company_name = st.text_input("Please enter the name of company you want to analyze")
#         st.session_state.uploaded_file = st.file_uploader("Upload a your ESG report(PDF) 📎", type=("pdf"))

#     if not st.session_state.uploaded_file:
#         st.warning("⬅️ Please upload a PDF file to continue 👻")
#     if st.session_state.uploaded_file:
#         st.write("File uploaded successfully! 🎉")
#         init_session()
        
#         with st.form(key='extraction_form'):
#             # st.session_state.df_info = None
#             # st.session_state.df_summary = None
#             start = st.form_submit_button("Strat to analyze")
#             if start:
#                 with st.spinner("Converting PDF file into text..."):
#                     st.session_state.doc_parsed, st.session_state.pdf_texts = convert_pdf_to_text(st.session_state.uploaded_file)
                    
#                 with st.spinner("Extracting ESG information..."): 
#                     st.session_state.df_info = convert_text_to_xlsx(st.session_state.doc_parsed)
#                     st.session_state.df_summary = convert_xlsx_to_summary(st.session_state.df_info, company_name) 


#         if st.session_state.df_summary is not None:
#             st.markdown("#### Summary of Extracted ESG Related Data:")      
#             st.dataframe(st.session_state.df_summary)
#             df_summary_csv= convert_df(st.session_state.df_summary)
#             st.download_button(
#                 label="Download Extracted ESG Data Summary",
#                 data=df_summary_csv,
#                 file_name=f"{company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
#                 mime="text/csv",)
        
#             df_info_button = st.toggle("View Raw Extracted ESG Data 👇🏻")
#             if df_info_button:
#                 st.markdown("**Below is the raw ESG data extracted from the report:**")
#                 st.dataframe(st.session_state.df_info)
#                 df_info_csv= convert_df(st.session_state.df_info)
#                 st.download_button(
#                     label="Download Raw Extracted ESG Data as CSV",
#                     data=df_info_csv,
#                     file_name=f"{company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
#                     mime="text/csv",)
                
#             pdf_texts_button = st.toggle("View Converted ESG Report Texts 👇🏻")
#             if pdf_texts_button:
#                 st.markdown("**Texts of uploaded ESG report:**")
#                 with st.container(height=600):
#                     st.markdown(st.session_state.pdf_texts)

if not (input_openai_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key on the left to continue.", icon="🗝️")
else:
    with st.sidebar:
        st.session_state.company_name = st.text_input("Please enter the name of company you want to analyze")
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
                        df_summary = convert_xlsx_to_summary(df_info, st.session_state.company_name)
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
                file_name=f"{st.session_state.company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
                mime="text/csv",)
        
            df_info_button = st.toggle("View Raw Extracted ESG Data 👇🏻")
            if df_info_button:
                st.markdown("**Below is the raw ESG data extracted from the report:**")
                st.dataframe(st.session_state.df_info)
                df_info_csv= convert_df(st.session_state.df_info)
                st.download_button(
                    label="Download Raw Extracted ESG Data as CSV",
                    data=df_info_csv,
                    file_name=f"{st.session_state.company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
                    mime="text/csv",)
                
            # pdf_texts_button = st.toggle("View Converted ESG Report Texts 👇🏻")
            # if pdf_texts_button:
            #     st.markdown("**Texts of uploaded ESG report:**")
            #     with st.container(height=600):
            #         st.markdown(st.session_state.pdf_texts)
 
                    
        if st.session_state.df_summary is not None:
            st.markdown("### ESG Summary")
            st.subheader("ESG Analysis")

            scored_esg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/scored_tech_industry_esg_data.csv"   
            scored_tech_esg = load_github_csv(scored_esg_url)    
            ESG_score_trend, esg_industry_plot_data = ESG_trend(scored_tech_esg)
            fig_esg_trend = ESG_trend_plot(esg_industry_plot_data)
            st.markdown("##### Trend of ESG Performance in Tech Industry")    
            st.plotly_chart(fig_esg_trend)

            esg_cluster_centers_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/tech_esg_cluster_centers.csv"
            esg_cluster_centers = load_github_csv(esg_cluster_centers_url)

            cluster_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/cluster_model.pkl"
            cluster_model = load_github_model(cluster_url)
                
            reg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/scoring_model.pkl"
            scoring_model = load_github_model(reg_url)

            compare_fig = company_scoring(scored_tech_esg, st.session_state.df_summary, cluster_model, esg_cluster_centers, scoring_model, esg_industry_plot_data, ESG_score_trend)
            st.plotly_chart(compare_fig)
                    
            st.markdown("#### You could find more ESG related reports from the following sources:")
            input_serp_api_key = os.environ["SERP_API_KEY"]
            if not input_serp_api_key:
                st.info("Please add your Serp API key to continue.", icon="🗝️")
                input_serp_api_key = st.text_input("Serp API Key", type="password")
                st.session_state.news_df = get_esg_news(st.session_state.company_name, input_serp_api_key)
                st.dataframe(st.session_state.news_df, 
                                column_config={"link": st.column_config.LinkColumn()})
            else:
                input_serp_api_key = os.environ["SERP_API_KEY"]
                st.session_state.news_df = get_esg_news(st.session_state.company_name, input_serp_api_key)
                st.dataframe(st.session_state.news_df, 
                                column_config={"link": st.column_config.LinkColumn()})
        
                
                # st.header("Financial Analysis")
                # company_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/tests/FinancialData/company_ticker_mapping.csv"
                # response = requests.get(company_url)
                # if response.status_code == 200:
                #     company_name_mapping = pd.read_csv(StringIO(response.text))
                # else:
                #     st.error("Failed to load mapping table of company name from GitHub.")

                # stock_price = get_stock_data(company_name, company_name_mapping)
                # if stock_price is not None:
                #     fig = px.line(stock_price, x='Date', y='Adj Close', title=f"{company_name} Stock Price")
                #     st.plotly_chart(fig)
                # else:
                #     st.warning("No available stock price data. 🙁")
                    


            st.subheader("Financial Analysis")
            company_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/tests/FinancialData/company_ticker_mapping.csv"
            response = requests.get(company_url)
            if response.status_code == 200:
                company_name_mapping = pd.read_csv(StringIO(response.text))
            else:
                st.error("Failed to load mapping table of company name from GitHub.")

            stock_price = get_stock_data(st.session_state.company_name, company_name_mapping)
            if stock_price is not None:
                fig = px.line(stock_price, x='Date', y='Adj Close', title=f"{st.session_state.company_name} Stock Price")
                st.plotly_chart(fig)
                
                stock_price = stock_data_manipulation(stock_price)
                fig_volatility_risk = rolling_vol_plot(stock_price)
                st.plotly_chart(fig_volatility_risk)
                    
                # fig_volatility_pred = volatility_pred(stock_price)
                # st.plotly_chart(fig_volatility_pred)
                
                time_step = 60
                with st.spinner("Predicting your future trend of stock..."):
                    stock_price, scaled_data, model, features, scaler = stock_pred_model(stock_price, time_step)
                    fig_pred, future_df = stock_pred(stock_price, scaled_data, features, model, time_step, scaler)
                    st.plotly_chart(fig_pred)
            else:
                st.warning("Oops... No available stock price data. 🙁")
                
                
            fin_data_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/data/financial_data.csv"
            fin_sub_mean_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/data/financial_sub_mean.csv"
            fin_mean_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/data/financial_mean.csv"
            financial_database = load_github_csv(fin_data_url)
            fin_sub_mean = load_github_csv(fin_sub_mean_url)
            fin_mean = load_github_csv(fin_mean_url)
            fin_df = fin_data_manipulate(financial_database, fin_sub_mean, fin_mean, st.session_state.company_name)
            fin_plots = plot_financial_data(fin_df)
            
            company_fin_data = financial_database[financial_database['Company'].str.lower() == st.session_state.company_name.lower()]
            fin_metrics = regulator_analyze_financial_metrics(company_fin_data, fin_mean)
            # st.markdown('\n\n'.join(fin_metrics))
            
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
            #     for i in range(0, len(fin_plots), 3):
            #         st.plotly_chart(fin_plots[i])
            #         st.markdown(fin_metrics[i])
            # with col2:
            #     for i in range(1, len(fin_plots), 3):
            #         st.plotly_chart(fin_plots[i])
            #         st.markdown(fin_metrics[i])
            # with col3:
            #     for i in range(2, len(fin_plots), 3):
            #         st.plotly_chart(fin_plots[i])
            #         st.markdown(fin_metrics[i])
            
            col1, col2 = st.columns(2)
            with col1:
                for i, (metric, result) in enumerate(fin_metrics.items()):
                    if i % 2 == 0:
                        st.plotly_chart(fin_plots[i])
                        st.markdown(result['Analysis'])
                        if result['Recommendation']:
                            st.markdown(f"Recommendation: {result['Recommendation']}")

            with col2:
                for i, (metric, result) in enumerate(fin_metrics.items()):
                    if i % 2 != 0:
                        st.plotly_chart(fin_plots[i])
                        st.markdown(result['Analysis'])
                        if result['Recommendation']:
                            st.markdown(f"Recommendation: {result['Recommendation']}")
                
            st.balloons()
                


        
            
        
            

   
    







