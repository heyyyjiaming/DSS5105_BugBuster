import streamlit as st
# import streamlit_scrollable_textbox as stx
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
# import matplotlib.pyplot as plt
import plotly.express as px


st.title("ESG Analysis Tool 📊")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)

input_openai_api_key = st.text_input("OpenAI API Key", type="password")
input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")

if input_openai_api_key:
    os.environ["OPENAI_API_KEY"] = input_openai_api_key
if input_llama_api_key:
    os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api_key


identify_prompt = PromptTemplate(
    instruction="""Extract and directly copy any text-based content or tables specifically containing ESG information that could be used for a data analysis. Focus on capturing content that is comprehensive.
    """,
    few_shot_prompt=[
        Context(
            context="The company reported a total of 10,001 promtCO2e of Scope 1 emissions in 2020."""
    )]
)


standardize_prompt = PromptTemplate(
    instruction="""Standardize the ESG contents or tables into a structured data frame that includes: 'label' , 'metric', 'unit', 'year' and 'value' (numerical value). 
    Here is the reference for 'label', 'metric' and 'unit': 
    {
"Label": {
    "Greenhouse Gas Emissions": [
    {
        "metric": "Total","Scope 1","Scope 2","Scope 3"
        "unit": "tCO2e"
    },
    {
        "metric": "Emission intensities of total","Emission intensities of Scope 1","Emission intensities of Scope 2","Emission intensities of Scope 3"
        "unit": "tCO2e"
    }
    ],
    "Energy Consumption": [
    {
        "metric": "Total energy consumption",
        "unit": "MWhs", "GJ"
    },
    {
        "metric": "Energy consumption intensity",
        "unit": "MWhs", "GJ"
    }
    ],
    "Water Consumption": [
    {
        "metric": "Total water consumption",
        "unit": "ML", "m³"
    },
    {
        "metric": "Water consumption intensity",
        "unit": "ML", "m³"
    }
    ],
    "Waste Generation": {
    "metric": "Total waste generated",
    "unit": "t"
    },
    "Gender Diversity": [
    {
        "metric": "Current employees by gender",
        "unit": "Male Percentage (%)","Female Percentage (%)","Others Percentage (%)"
    },
    {
        "metric": "New hires and turnover by gender",
        "unit": "Male Percentage (%)","Female Percentage (%)","Others Percentage (%)"
    }
    ],
    "Age-Based Diversity": [
    {
        "metric": "Current employees by age groups",
        "unit": "Baby Boomers (%)","Gen Xers (%)","Millennials (%)","Gen Z (%)"
    },
    {
        "metric": "New hires and turnover by age groups",
        "unit": "Baby Boomers (%)","Gen Xers (%)","Millennials (%)","Gen Z (%)"
    }
    ],
    "Employment": [
    {
        "metric": "Total employee turnover",
        "unit": "Number", "Percentage (%)"
    },
    {
        "metric": "Total number of employees",
        "unit": "Number"
    }
    ],
    "Development & Training": [
    {
        "metric": "Average training hours per employee",
        "unit": "Hours/No. of employees"
    },
    {
        "metric": "Average training hours per employee by gender",
        "unit": "Male Hours/No. of employees", "Female Hours/No. of employees"
    }
    ],
    "Occupational Health & Safety": [
    {
        "metric": "Fatalities",
        "unit": "Number of cases"
    },
    {
        "metric": "High-consequence injuries",
        "unit": "Number of cases"
    },
    {
        "metric": "Recordable injuries",
        "unit": "Number of cases"
    }
    ],
    "Recordable work-related illnesses": {
    "metric": "Number of recordable work-related illnesses or health conditions",
    "unit": "Number of cases"
    },
    "Board Composition": [
    {
        "metric": "Board independence",
        "unit": "Percentage (%)"
    },
    {
        "metric": "Women on the board",
        "unit": "Percentage (%)"
    }
    ],
    "Management Diversity": {
    "metric": "Women in the management team",
    "unit": "Percentage (%)"
    },
    "Ethical Behaviour": [
    {
        "metric": "Anti-corruption disclosures",
        "unit": "Discussion and number"
    },
    {
        "metric": "Anti-corruption training for employees",
        "unit": "Number and Percentage (%)"
    }
    ],
    "Certifications": {
    "metric": "List of relevant certifications",
    "unit": "List"
    },
    "Alignment with Frameworks": {
    "metric": "Alignment with frameworks and disclosure practices"
    },
    "Assurance": {
    "metric": "Assurance of sustainability report",
    "unit": "Internal","External","None"
    }
}
}
    Return the standardized data frame for analysis.
    """,
    few_shot_prompt=[
        Context(
            label="Greenhouse Gas Emissions""",
            metrics="Scope 1""",
            unit="tCO2e""",
            year="2020""",
            value=10001
    )]
)

#Set AI config
identify_config = TransformOpenAIConfig(
    prompt_template=identify_prompt,
    model_config=OpenAIModelConfig(
        model_name = 'gpt-4o-mini',
        response_format={"type": "json_object"}
    ),
)

standardize_config = TransformOpenAIConfig(
    prompt_template=standardize_prompt,
    model_config=OpenAIModelConfig(
        model_name = 'gpt-4o-2024-08-06',
        response_format={"type": "json_object"}
    ),
)


# def convert_text_to_xlsx(documents):
#     identify_client = TransformClient(identify_config)
#     standardize_client = TransformClient(standardize_config)

#     #store the extracted esg contents as a dictionary
#     ESG_contents = {}

#     for idx, doc in enumerate(documents):
#         input_page = [
#             Context(
#                 context=doc.text,
#             )]
#         ESG_contents[idx] = identify_client.run(input_page)

#     # Restructure the extracted esg contents as a list
#     extracted_contents = extract_esg_contents(ESG_contents)
#     # run step 2 and store the json output in a dictionary 
#     output = {}

#     for idx, item in enumerate(extracted_contents):
#         sentence = [
#             Context(
#                 context=item
#             )
#             ]

#         output[idx] = standardize_client.run(sentence)

#     # transform the json output into a DataFrame
#     unit = []
#     label = []
#     year =[]
#     metric = []
#     value = []
#     for out in output.values():  
#         for item in out:
#             for i in item.get('output', []):
#                 for response in i.get('response', []):
#                     for key in response:
#                                     if isinstance(response[key], list) and len(response[key]) > 0:
#                                         for res in response[key]:  
#                                             if all(k in res for k in [ 'unit','label', 'year', 'metric', 'value']):
#                                                 unit.append(res['unit'])
#                                                 label.append(res['label'])
#                                                 year.append(res['year'])
#                                                 metric.append(res['metric'])
#                                                 value.append(res['value'])
                        
#     df = pd.DataFrame({
#         'label': label,
#         'metric': metric,
#         'unit' : unit,
#         'year':year,
#         'value' :value
#     })

#     # Delet the example data
#     df_filtered = df[df['value'] != 10001]
#     return df_filtered


if not (input_openai_api_key and input_llama_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key to continue.", icon="🗝️")
else:
    company_name = st.text_input("Please enter the name of company you want to analyze")
    st.session_state.uploaded_file = st.file_uploader("Upload a document (PDF)", type=("pdf"))

    if st.session_state.uploaded_file:
        st.write("File uploaded successfully! 🎈")
        

        with st.spinner("Extracting text from the PDF file..."):
            st.session_state.doc_parsed = convert_pdf_to_text(st.session_state.uploaded_file)
        # content_on = st.toggle("Show the content of the PDF file 📄", False)
        # if content_on:
        #     with st.spinner("Loading the content..."):
        #         st.markdown(st.session_state.contents)
           
            
        with st.spinner("Extracting ESG information..."): 
            st.session_state.df_info = convert_text_to_xlsx(st.session_state.doc_parsed)
            st.dataframe(st.session_state.df_info)
            
        with st.spinner("Summarizing ESG information..."):
            st.session_state.df_summary = convert_xlsx_to_summary(st.session_state.df_info, company_name)
            st.dataframe(st.session_state.df_summary)
   


# e_company_info = st.session_state.df_summary[['Company Name', 'Year']]
# e_company_numeric = st.session_state.df_summary.drop(columns=['Company Name', 'Year', 'GHG Emissions (Total)'])

# # Handle missing values(Implement techniques to handle missing data and ensure fair comparisons across
# # companies.)
# imputer = SimpleImputer(strategy='median')
# e_company_imputed = pd.DataFrame(imputer.fit_transform(e_company_numeric), columns=e_company_numeric.columns)

# # Step 3: Standardize Data
# scaler = StandardScaler()
# e_company_scaled = pd.DataFrame(scaler.fit_transform(e_company_imputed), columns=e_company_numeric.columns)
            
# company_e_scaled_data = e_company_scaled.values  # 标准化后的数据作为特征输入

# # 使用 KMeans 模型进行聚类预测
# optimal_clusters = 3
# kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
# company_e_clusters = kmeans.predict(company_e_scaled_data)
# st.session_state.df_summary['Cluster'] = company_e_clusters



# # 使用新的聚类结果为 Company 1 分配 Performance Category
# st.session_state.df_summary['Performance Category'] = st.session_state.df_summary['Cluster'].apply(categorize_performance_by_cluster)


# # Step 2: 使用线性回归公式计算每年的 ESG 得分
# # 假设 reg.coef_ 已经存储了模型的权重 (weights)，并且 intercept_b 已经存储了截距 (b)
# e_weights = reg.coef_
# e_intercept_b = reg.intercept_

# # 将权重转换为NumPy数组，以便于矩阵运算
# e_weights = np.array(e_weights)         







