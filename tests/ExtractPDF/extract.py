# %%
import os
import pandas as pd
import nest_asyncio
import sys
from llama_parse import LlamaParse
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig
from uniflow.flow.config import OpenAIModelConfig
from uniflow.op.prompt import PromptTemplate, Context
from dotenv import load_dotenv

# Load the .env file
dotenv_path = 'D:/apikeys/.env'  
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the API key from the environment
openAI_API = os.getenv('OPENAI_API_KEY')
llama_API = os.getenv('LLAMA_API_KEY')

# %%
# Set dir
sys.path.append(".")
sys.path.append("..")

thisfile_dir = os.path.dirname(os.path.abspath(__file__))#if put this file in tests/ExtractPDF
dir_cur = os.path.join(thisfile_dir, '..', '..')
print("Target Directory:", dir_cur)

# %%
def convert_pdf_to_text(input_file, output_path, input_llama_api):
    
    # Get the base name of the file
    base_name = os.path.basename(input_file)

    # Transform PDF
    nest_asyncio.apply()

    if input_llama_api == '':
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_API
    else:
        os.environ["LLAMA_CLOUD_API_KEY"] = input_llama_api

    documents = LlamaParse(result_type="markdown").load_data(input_file)

    # Merge all the text into one str
    all_text = []
    for doc in documents:
        all_text.append(doc.text)

    merged_doc = '\n\n'.join(all_text)

    # Save as txt
    txt_output_path = output_path
    with open(txt_output_path, 'w', encoding='utf-8') as file:
        file.write(merged_doc)
        
    return documents

# %% 
pdf_file = "test.pdf"
input_file = os.path.join(f"{dir_cur}/data/Reports", pdf_file)
base_name = os.path.basename(input_file)
txt_output_path = os.path.join(dir_cur, 'outputs/llama_parsed', f'{base_name}.txt')
input_llama_api = ''

documents = convert_pdf_to_text(input_file, txt_output_path, input_llama_api)

# %%
### Set configs
# Set prompt and example
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

# Restructure the extracted esg contents as a list
def extract_esg_contents(esg_contents):
    extracted_responses = []

    try:
        # Iterate through the keys of the ESG_contents dictionary
        for key in esg_contents:
            items = esg_contents[key]
            
            # Iterate through each item in the list associated with the current key
            for item in items:
                output_list = item.get('output', [])
                
                # Iterate through each output item
                for output_item in output_list:
                    response_list = output_item.get('response', [])
                    
                    # Append each response item to the extracted_responses list
                    for response_item in response_list:
                        extracted_responses.append(response_item)
    
    except Exception as e:
        print(f"Error extracting response content: {e}")

    return extracted_responses

# %%    
def convert_text_to_xlsx(documents, output_path, input_openai_api):
    load_dotenv()
    if input_openai_api == '':
        os.environ["OPENAI_API_KEY"] = openAI_API
    else:
        os.environ["OPENAI_API_KEY"] = input_openai_api
        
    identify_client = TransformClient(identify_config)
    standardize_client = TransformClient(standardize_config)

    #store the extracted esg contents as a dictionary
    ESG_contents = {}

    for idx, doc in enumerate(documents):
        input_page = [
            Context(
                context=doc.text,
            )]

        ESG_contents[idx] = identify_client.run(input_page)

    # Restructure the extracted esg contents as a list
    extracted_contents = extract_esg_contents(ESG_contents)

    # run step 2 and store the json output in a dictionary 
    output = {}

    for idx, item in enumerate(extracted_contents):
        sentence = [
            Context(
                context=item
            )
            ]

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

    # Save DataFrame as Excel file
    excel_output_path = output_path
    df_filtered.to_excel(excel_output_path)


# %% 
input_openai_api = ''
output_path = os.path.join(dir_cur, 'outputs/extracted_data', f'{base_name}.xlsx')
convert_text_to_xlsx(documents, output_path, input_openai_api)