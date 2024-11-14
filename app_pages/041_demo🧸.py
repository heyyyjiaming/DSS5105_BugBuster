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
# from model.finance_eval import stock_data_manipulation, rolling_vol_plot, volatility_pred, stock_pred_model, stock_pred, fin_data_manipulate, plot_financial_data
from model.finance_eval import *
from arch import arch_model
import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")

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




st.title("ESGenius 🪄")
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
    st.info("Please add your OpenAI API key on the left to continue.", icon="🗝️")
else:
    with st.sidebar:
        st.session_state.company_name = st.text_input("Please enter the name of company you want to analyze")
        st.session_state.uploaded_path = st.file_uploader("Upload a your ESG report(PDF) 📎", type=("pdf"), accept_multiple_files=True)
        translate = st.checkbox("🙋🏻‍♀️ The report is **NON-ENGLISH**")
    if not st.session_state.uploaded_path:
        st.warning("⬅️ Please upload a PDF file to continue 👻")
    else:
        st.write("File uploaded successfully! 🎉")
        risk_pref = st.selectbox("Please select your preferred investment risk level", ["Middle", "High", "Low"])
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
        file_name=f"{st.session_state.company_name.replace(" ", "_")}_raw_extracted_esg_ata.csv",
        mime="text/csv",)

         
############################## ESG Summary ###############################
                    
# if st.session_state.df_summary is not None:
    st.header("ESG Summary")
    
    st.subheader("🌱 ESG Analysis")
    scored_esg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/scored_tech_industry_esg_data.csv"   
    scored_tech_esg = load_github_csv(scored_esg_url)    
    
    esg_cluster_centers_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/tech_esg_cluster_centers.csv"
    esg_cluster_centers = load_github_csv(esg_cluster_centers_url)
    
    cluster_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/cluster_model.pkl"
    cluster_model = load_github_model(cluster_url)
        
    reg_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/scoring_model.pkl"
    scoring_model = load_github_model(reg_url)
    
# PART 1
    ESG_score_trend, esg_industry_plot_data = ESG_trend(scored_tech_esg)
    fig_esg_trend = ESG_trend_plot(esg_industry_plot_data)
    st.markdown("##### Trend of ESG Performance in Tech Industry")    
    st.plotly_chart(fig_esg_trend)
    
    esg_weights = scoring_model.coef_
    esg_bottom3_idx = esg_weights.argsort()[:3]
    esg_cols = st.session_state.df_summary.columns[2:]
    st.markdown("⚠️ **Top 3 ESG indicators that might impair your score**")
    for idx,col in enumerate(esg_cols[esg_bottom3_idx]):
        st.markdown(f"Top{idx+1}: {col}")
        # st.markdown(f"Top{idx+1} ESG Weights: {col}, {esg_weights[esg_bottom3_idx][idx]:.4f}")
            
    
    
# PART 2
    
    compare_fig = company_scoring(scored_tech_esg, st.session_state.df_summary, cluster_model, esg_cluster_centers, scoring_model, esg_industry_plot_data, ESG_score_trend)
    st.markdown("\n\n")
    st.markdown("\n\n")
    st.markdown("\n")
    st.markdown("##### Trend of ESG Performance in Tech Industry") 
    st.plotly_chart(compare_fig)
    
    fin_top3_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/top_features_df.csv"
    fin_bott3_url = "https://raw.githubusercontent.com/heyyyjiaming/DSS5105_BugBuster/refs/heads/main/model/data/bottom_features_df.csv"
    fin_top3 = load_github_csv(fin_top3_url)
    fin_bott3 = load_github_csv(fin_bott3_url)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("✅ Top 3 ESG indicators **most** related to Finance")
        for idx,col in enumerate(fin_top3['Feature']):
            st.markdown(f"Top{idx+1}: {col}")
            # st.markdown(f"Top{idx+1} ESG Weights: {col}, {esg_weights[esg_bottom3_idx][idx]:.4f}")
    with col2:
        st.markdown("❎ Top 3 ESG indicators **least** related to Finance")
        for idx,col in enumerate(fin_bott3['Feature']):
            st.markdown(f"Top{idx+1}: {col}")
            # st.markdown(f"Top{idx+1} ESG Weights: {col}, {esg_weights[esg_bottom3_idx][idx]:.4f}")
    
    
    ## External Data        
    st.markdown(f"#### You could find more **{st.session_state.company_name}**'s real-time reports related to ESG from the following sources:")
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


############################## Finance Summary ###############################
    st.subheader("💰 Financial Analysis")
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
        volatility, fig_volatility_risk = rolling_vol_plot(stock_price)
        st.plotly_chart(fig_volatility_risk)
        st.markdown(volatility_analysis_invest(volatility))
            
        # fig_volatility_pred = volatility_pred(stock_price)
        # st.plotly_chart(fig_volatility_pred)
        
        
        time_step = 60
        with st.spinner("Predicting your future trend of stock..."):
            stock_price, scaled_data, model, features, scaler = stock_pred_model(stock_price, time_step)
            fig_pred, future_df = stock_pred(stock_price, scaled_data, features, model, time_step, scaler)
            st.plotly_chart(fig_pred)    
            
            confidence_level = 0.95
            VaR, CVaR, fig_var = var_calculate(future_df, confidence_level)
            st.plotly_chart(fig_var)
            
            match_con, VaR_analysis = risk_analysis(risk_pref, VaR)
            st.markdown(f"Confidence Interval {confidence_level * 100}%, **VaR**: {VaR:.4f}")
            # st.markdown(f"Confidence Interval {confidence_level * 100}%, CVaR: {CVaR:.4f}")
            
            st.markdown(match_con)
            st.markdown(VaR_analysis)
            
        
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
    fin_metrics = investor_analyze_financial_metrics(company_fin_data, fin_mean)
    # fin_metrics = regulator_analyze_financial_metrics(company_fin_data, fin_mean)
    # st.markdown('\n\n'.join(fin_metrics))
    
    col3, col4 = st.columns(2)
    
    with col3:
        for i, (metric, result) in enumerate(fin_metrics.items()):
            if i % 2 == 0:
                st.plotly_chart(fin_plots[i])
                st.markdown(result['Analysis'])
                if result['Recommendation']:
                    st.markdown(f"Recommendation: {result['Recommendation']}")
        # for i in range(0, len(fin_plots), 2):
        #     st.plotly_chart(fin_plots[i])
        #     st.markdown(fin_metrics[i])
    with col4:
        for i, (metric, result) in enumerate(fin_metrics.items()):
            if i % 2 != 0:
                st.plotly_chart(fin_plots[i])
                st.markdown(result['Analysis'])
                if result['Recommendation']:
                    st.markdown(f"Recommendation: {result['Recommendation']}")
        # for i in range(1, len(fin_plots), 2):
        #     st.plotly_chart(fin_plots[i])
        #     st.markdown(fin_metrics[i])
        
    st.balloons()
        
    

        
