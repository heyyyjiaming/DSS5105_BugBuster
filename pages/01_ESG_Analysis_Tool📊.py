import streamlit as st
# import streamlit_scrollable_textbox as stx
import os
import glob
import pandas as pd
from utils.extract import convert_pdf_to_text, extract_esg_contents
# from extract import convert_text_to_xlsx  
from utils.tools import read_pdf
from llama_parse import LlamaParse
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig
from uniflow.flow.config import OpenAIModelConfig
from uniflow.op.prompt import PromptTemplate, Context



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
identify_prompt = PromptTemplate(
    instruction="""Extract and directly copy any text-based content or tables specifically containing ESG information that could be used for a data analysis. Focus on capturing content that is comprehensive.
    """,
    few_shot_prompt=[
        Context(
            context="The company reported a total of 10,001 promtCO2e of Scope 1 emissions in 2020."""
    )]
)

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


if not (input_openai_api_key and input_llama_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key to continue.", icon="🗝️")
else:
    st.session_state.uploaded_file = st.file_uploader("Upload a document (PDF)", type=("pdf"))

    if st.session_state.uploaded_file:
        st.write("File uploaded successfully!")
        # Parse the PDF file
        st.session_state.doc_parsed = LlamaParse(result_type="markdown").\
                                    load_data(st.session_state.uploaded_file, extra_info={"file_name": "_"})

        all_text = []
        for doc in st.session_state.doc_parsed:
            all_text.append(doc.text)

        merged_doc = '\n\n'.join(all_text)
        # st.markdown("**Here is the content of the PDF file 📄:**")
        # stx.scrollableTextbox(context,height = 500)
        content_on = st.toggle("Show the content of the PDF file 📄", False)
        if content_on:
            st.markdown(merged_doc)

        ### Extract ESG Information
        identify_client = TransformClient(identify_config)
        standardize_client = TransformClient(standardize_config)
        #store the extracted esg contents as a dictionary
        ESG_contents = {}

        for idx, doc in enumerate(st.session_state.doc_parsed):
            input_page = [
                Context(context=doc.text,
                )]

            ESG_contents[idx] = identify_client.run(input_page)

        # Restructure the extracted esg contents as a list
        extracted_contents = extract_esg_contents(ESG_contents)

        # run step 2 and store the json output in a dictionary 
        output = {}

        for idx, item in enumerate(extracted_contents):
            sentence = [Context(context=item)]
            output[idx] = standardize_client.run(sentence)

        # transform the json output into a DataFrame
        unit = []
        label = []
        year =[]
        metric = []
        value = []
        for out in output.values():  
            for item in out:
                for i in item.get('output', []):
                    for response in i.get('response', []):
                        for key in response:
                                        if isinstance(response[key], list) and len(response[key]) > 0:
                                            for res in response[key]:  
                                                if all(k in res for k in [ 'unit','label', 'year', 'metric', 'value']):
                                                    unit.append(res['unit'])
                                                    label.append(res['label'])
                                                    year.append(res['year'])
                                                    metric.append(res['metric'])
                                                    value.append(res['value'])
                            
        df = pd.DataFrame({
            'label': label,
            'metric': metric,
            'unit' : unit,
            'year':year,
            'value' :value
        })

        # Delet the example data
        df_filtered = df[df['value'] != 10001]
        st.dataframe(df_filtered)
        

            
            



#Current dir
# thisfile_dir = os.path.dirname(os.path.abspath(__file__))
# dir_cur = os.path.join(thisfile_dir, '..')
# print("Target Directory:", dir_cur)

# pdf_directory = os.path.join(dir_cur, 'data/Reports')
# txt_output_path = os.path.join(dir_cur, 'outputs/llama_parsed')
# xlsx_directory = os.path.join(dir_cur, 'outputs/extracted_data')

# # Get all PDF 
# pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

# # select pdf
# selected_pdf = st.selectbox("Please choose a PDF file", pdf_files)
# base_name = os.path.basename(selected_pdf)

# # 检查选择的文件是否已经有对应的 XLSX 文件
# selected_basename = os.path.basename(selected_pdf)
# xlsx_file = os.path.join(xlsx_directory, f'{selected_basename}.txt')

# if os.path.exists(xlsx_file):
#     st.success(f"已存在对应的 XLSX 文件: {xlsx_file}")
# else:
#     st.info("没有找到对应的 XLSX 文件。")

# # 文件转换按钮
# if st.button("转换为 XLSX"):
#     if (input_openai_api_key and input_llama_api_key):
#         if not os.path.exists(xlsx_file):
#             convert_pdf_to_text(selected_pdf, xlsx_file, input_llama_api_key)
#             st.success("PDF 已成功转换为 XLSX。")
#         else:
#             st.warning("XLSX 文件已经存在，不需要重新转换。")
#     else:
#         st.error("请提供有效的 API 网址。")



