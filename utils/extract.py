import os
import pandas as pd
import nest_asyncio
import sys
from llama_parse import LlamaParse
from uniflow.flow.client import TransformClient
from uniflow.flow.config import TransformOpenAIConfig
from uniflow.flow.config import OpenAIModelConfig
from uniflow.op.prompt import PromptTemplate, Context



def convert_pdf_to_text(input_file):
    
    nest_asyncio.apply()
    documents = LlamaParse(result_type="markdown").load_data(input_file, extra_info={"file_name": "_"})

    # Merge all the text into one str
    all_text = []
    for doc in documents:
        all_text.append(doc.text)

    merged_doc = '\n\n'.join(all_text)

    return documents, merged_doc




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


    
def convert_text_to_xlsx(documents): 
    identify_prompt = PromptTemplate(
        instruction="""Extract and directly copy any text-based content or tables specifically containing ESG information that could be used for a data analysis. Focus on capturing content that is comprehensive.
        """,
        few_shot_prompt=[
            Context(
                context="The company reported a total of 10,001 promtCO2e of Scope 1 emissions in 2020."""
    )])
    
    identify_config = TransformOpenAIConfig(
        prompt_template=identify_prompt,
        model_config=OpenAIModelConfig(
            model_name = 'gpt-4o-mini',
            response_format={"type": "json_object"}),
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
    
    standardize_config = TransformOpenAIConfig(
        prompt_template=standardize_prompt,
        model_config=OpenAIModelConfig(
            model_name = 'gpt-4o-2024-08-06',
            response_format={"type": "json_object"}
        ),
    )
    
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
    # print(extracted_contents)

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
    
    return df_filtered
    
    
    
exchange_rates = {
    'EUR': 1.1,   # 1 EUR = 1.1 USD
    'GBP': 1.3,   
    'CAD': 0.8,   
    'CNY': 0.139,  
    'SGD': 0.754,
    'JPY': 0.0091, 
    'AUD': 0.7,
    'USD': 1  
}

## Change whole func
def modify_units(row):
        # Check if the 'year' field is not null
    if pd.notnull(row['year']):
        year_str = str(row['year'])
        
        # Initialize variables for finding the year sequence
        potential_year = ''
        year_found = False
        
        for char in year_str:
            if char.isdigit():
                potential_year += char
            else:
                # Reset if the sequence goes beyond four digits
                potential_year = ''

            # Once four consecutive digits are found, validate
            if len(potential_year) == 4:
                year_int = int(potential_year)
                # Define the valid year range
                min_year, max_year = 2015, 2025
                if min_year <= year_int <= max_year:
                    row['year'] = year_int
                    year_found = True
                    break
                else:
                    # Reset and continue searching if invalid
                    potential_year = ''

        if not year_found:
            row['year'] = None
    else:
        row['year'] = None
    
    # Existing unit modifications
    if row['unit'] == 'GJ':
        row['unit'] = 'MWhs'
        row['value'] = row['value'] * 0.277778  # GJ to MWh
    elif row['unit'] == 'm³':
        row['unit'] = 'ML'
        row['value'] = row['value'] * 0.001  # m3 to ML
    elif row['unit'] == 'm3':
        row['unit'] = 'ML'
        row['value'] = row['value'] * 0.001  # m3 to ML
    elif row['unit'] == 'kg':
        row['unit'] = 't'
        row['value'] = row['value'] / 1000  # kg to t
    elif row['unit'] in exchange_rates:
        row['value'] = row['value'] * exchange_rates[row['unit']]
        row['unit'] = 'USD' 
    elif row['unit'] == 'Million dollars':
        row['value'] = row['value'] * 1000000
        row['unit'] = 'USD' 

    return row

## change whole
def custom_agg(values):
    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
    string_values = values[numeric_values.index.difference(values.index)]

    if not numeric_values.empty: 
        return numeric_values.max() 
    else:
        return ', '.join(string_values.unique()) 
    
def fill_esg_data(df, new_df, column_name, label=None, metric=None, unit=None):
    # condition = pd.Series([True] * len(df))
    # if label is not None:
    #     condition &= (df['label'].str.contains(label, regex=False))
    # if metric is not None:
    #     condition &= (df['metric'].str.contains(metric, regex=False))
    # if unit is not None:
    #     condition &= (df['unit'].str.contains(unit, regex=False))
    
    # filtered_df = df[condition]
    
    # max_values_by_year = filtered_df.groupby('year')['value'].max().reset_index()
    # max_values_by_year = max_values_by_year.set_index('year')

    # if column_name not in new_df.columns:
    #     new_df[column_name] = ''

    # for year, row in max_values_by_year.iterrows():
    #     new_df.loc[year, column_name] = row['value']
    condition = pd.Series([True] * len(df))
    if label is not None:
        condition &= (df['label'].str.contains(label, regex=False))
    if metric is not None:
        condition &= (df['metric'].str.contains(metric, regex=False))
    if unit is not None:
        condition &= (df['unit'].str.contains(unit, regex=False))
    
    filtered_df = df[condition]

    # def custom_agg(values):
    #     numeric_values = pd.to_numeric(values, errors='coerce').dropna()
    #     string_values = values[numeric_values.index.difference(values.index)]

    #     if not numeric_values.empty: 
    #         return numeric_values.max() 
    #     else:
    #         return ', '.join(string_values.unique()) 

    max_values_by_year = filtered_df.groupby('year')['value'].agg(custom_agg).reset_index()
    max_values_by_year = max_values_by_year.set_index('year')

    if column_name not in new_df.columns:
        new_df[column_name] = ''

    for year, row in max_values_by_year.iterrows():
        new_df.loc[year, column_name] = row['value']




## change whole
def restructure(df,company_name):
    # new_df = pd.DataFrame(columns=['year'])
    # new_df.set_index('year', inplace=True)
    
    # fill_esg_data(df, new_df, 'GHG Emissions (Scope 1) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 1', unit = 'tCO2e')
    # fill_esg_data(df, new_df, 'GHG Emissions (Scope 2) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 2', unit='tCO2e')
    # fill_esg_data(df, new_df, 'GHG Emissions (Scope 3) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 3', unit='tCO2e')
    # fill_esg_data(df, new_df, 'GHG Emissions (Total) (tCO2e)', label='Greenhouse Gas Emissions', metric='Total', unit='tCO2e')
    # fill_esg_data(df, new_df, 'Total Energy Consumption (MWhs)', label='Energy Consumption', metric='Total energy consumption', unit='MWhs')
    # fill_esg_data(df, new_df, 'Total Water Consumption (ML)', label='Water Consumption', metric='Total water consumption', unit='ML')
    # fill_esg_data(df, new_df, 'Total Waste Generated (t)', label='Waste Generation', metric='Total waste generated', unit='t')
    # fill_esg_data(df, new_df, 'Current Employees by Gender (Female %)', label='Gender Diversity', metric='Current employees by gender', unit='Female Percentage (%)')
    # fill_esg_data(df, new_df, 'New Hires and Turnover by Gender (Female %)', label='Gender Diversity', metric='New hires and turnover by gender', unit='Female Percentage (%)')
    # # fill_esg_data(df, new_df, 'Current Employees by Age Groups (Millennials %)', label='Age-Based Diversity', metric='Current employees by age groups', unit='Millennials (%)')
    # # fill_esg_data(df, new_df, 'New Hires and Turnover by Age Groups (Millennials %)', label='Age-Based Diversity', metric='New hires and turnover by age groups', unit='Millennials (%)')
    # fill_esg_data(df, new_df, 'Total Turnover (%)', label='Employment', metric='Total employee turnover')
    # fill_esg_data(df, new_df, 'Total Number of Employees', label='Employment', metric='Total number of employees')
    # fill_esg_data(df, new_df, 'Average Training Hours per Employee', label='Development & Training', metric='Average training hours per employee', unit='Hour')
    # fill_esg_data(df, new_df, 'Fatalities', metric='Fatalities')
    # fill_esg_data(df, new_df, 'High-consequence injuries', metric='High-consequence injuries', unit='Number')
    # fill_esg_data(df, new_df, 'Recordable injuries', metric='Recordable injuries', unit='Number')
    # fill_esg_data(df, new_df, 'Recordable work-related ill health cases', metric='Number of recordable work-related illnesses or health conditions', unit='Number')
    # fill_esg_data(df, new_df, 'Board Independence (%)', label='Board Composition', metric='Board independence')
    # fill_esg_data(df, new_df, 'Women on the Board (%)', label='Board Composition', metric='Women on the board')
    # fill_esg_data(df, new_df, 'Women in Management Team (%)', label='Management Diversity', metric='Women in the management team')
    # # fill_esg_data(df, new_df, 'Anti-Corruption Disclosures', metric='Anti-corruption disclosures')
    # fill_esg_data(df, new_df, 'Anti-Corruption Training for Employees (%)', label='Ethical Behaviour', metric='Anti-corruption training for employees',unit='Number')
    # # fill_esg_data(df, new_df, 'List of Relevant Certifications', label='Certifications', metric='List of relevant certifications')
    # # fill_esg_data(df, new_df, 'Alignment with Frameworks and Disclosure Practices', label='Alignment with Frameworks', metric='Alignment with frameworks and disclosure practices')
    # # fill_esg_data(df, new_df, 'Assurance of Sustainability Report', label='Assurance', metric='Assurance of sustainability report')

    # new_df.insert(0, 'Company Name', company_name)
    # new_df.rename_axis('Year', inplace=True)
    # new_df.reset_index(inplace=True)
    # new_df.fillna('', inplace=True)
    
    new_df = pd.DataFrame(columns=['year'])
    new_df.set_index('year', inplace=True)
    
    fill_esg_data(df, new_df, 'GHG Emissions (Scope 1) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 1', unit = 'tCO2e')
    fill_esg_data(df, new_df, 'GHG Emissions (Scope 2) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 2', unit='tCO2e')
    fill_esg_data(df, new_df, 'GHG Emissions (Scope 3) (tCO2e)', label='Greenhouse Gas Emissions', metric='Scope 3', unit='tCO2e')
    fill_esg_data(df, new_df, 'GHG Emissions (Total) (tCO2e)', label='Greenhouse Gas Emissions', metric='Total', unit='tCO2e')
    fill_esg_data(df, new_df, 'Total Energy Consumption (MWhs)', label='Energy Consumption', metric='Total energy consumption', unit='MWhs')
    fill_esg_data(df, new_df, 'Total Water Consumption (ML)', label='Water Consumption', metric='Total water consumption', unit='ML')
    fill_esg_data(df, new_df, 'Total Waste Generated (t)', label='Waste Generation', metric='Total waste generated', unit='t')
    fill_esg_data(df, new_df, 'Current Employees by Gender (Female %)', label='Gender Diversity', metric='Current employees by gender', unit='Female Percentage (%)')
    fill_esg_data(df, new_df, 'New Hires and Turnover by Gender (Female %)', label='Gender Diversity', metric='New hires and turnover by gender', unit='Female Percentage (%)')
    # fill_esg_data(df, new_df, 'Current Employees by Age Groups (Millennials %)', label='Age-Based Diversity', metric='Current employees by age groups', unit='Millennials (%)')
    # fill_esg_data(df, new_df, 'New Hires and Turnover by Age Groups (Millennials %)', label='Age-Based Diversity', metric='New hires and turnover by age groups', unit='Millennials (%)')
    fill_esg_data(df, new_df, 'Total Turnover (%)', label='Employment', metric='Total employee turnover')
    fill_esg_data(df, new_df, 'Total Number of Employees', label='Employment', metric='Total number of employees')
    fill_esg_data(df, new_df, 'Average Training Hours per Employee', label='Development & Training', metric='Average training hours per employee', unit='Hour')
    fill_esg_data(df, new_df, 'Fatalities', metric='Fatalities')
    fill_esg_data(df, new_df, 'High-consequence injuries', metric='High-consequence injuries', unit='Number')
    fill_esg_data(df, new_df, 'Recordable injuries', metric='Recordable injuries', unit='Number')
    fill_esg_data(df, new_df, 'Recordable work-related ill health cases', metric='Number of recordable work-related illnesses or health conditions', unit='Number')
    fill_esg_data(df, new_df, 'Board Independence (%)', label='Board Composition', metric='Board independence')
    fill_esg_data(df, new_df, 'Women on the Board (%)', label='Board Composition', metric='Women on the board')
    fill_esg_data(df, new_df, 'Women in Management Team (%)', label='Management Diversity', metric='Women in the management team')
    # fill_esg_data(df, new_df, 'Anti-Corruption Disclosures', metric='Anti-corruption disclosures')
    fill_esg_data(df, new_df, 'Anti-Corruption Training for Employees (%)', label='Ethical Behaviour', metric='Anti-corruption training for employees',unit='Number')
    # fill_esg_data(df, new_df, 'List of Relevant Certifications', label='Certifications', metric='List of relevant certifications')
    # fill_esg_data(df, new_df, 'Alignment with Frameworks and Disclosure Practices', label='Alignment with Frameworks', metric='Alignment with frameworks and disclosure practices')
    # fill_esg_data(df, new_df, 'Assurance of Sustainability Report', label='Assurance', metric='Assurance of sustainability report')

    new_df.insert(0, 'Company Name', company_name)
    new_df.rename_axis('Year', inplace=True)
    new_df.reset_index(inplace=True)
    new_df.fillna('', inplace=True)


    return new_df

def append_to_summary(existing_df, new_df):

    # Ensure 'Year' and 'Company Name' are present in new_df
    if 'Year' not in new_df.columns or 'Company Name' not in new_df.columns:
        raise ValueError("new_df must contain 'Year' and 'Company Name' columns")

    # Go through each row in new_df
    for index, new_row in new_df.iterrows():
        # Check if there is an existing row that matches the 'Year' and 'Company Name'
        match = (existing_df['Year'] == new_row['Year']) & (existing_df['Company Name'] == new_row['Company Name'])
        
        if existing_df[match].empty:
            # If there is no matching row, concatenate the new row
            existing_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            for col in new_df.columns:
                existing_value = existing_df.loc[match, col].values[0]
                new_value = new_row[col]
                
                if pd.isna(existing_value):
                    existing_df.loc[match, col] = new_value
                else:
                    # Check if both values are numeric or strings before comparing
                    if pd.notna(new_value):  # Only compare if new_value is not NaN
                        if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
                            existing_df.loc[match, col] = max(existing_value, new_value)
                        else:
                            existing_df.loc[match, col] = existing_value  # Keep existing value if types differ

#     # Write the updated dataframe back to the same Excel file
#     existing_df.to_excel(summary_table_path, sheet_name='E', index=False)
    
    
def convert_xlsx_to_summary(data_df, company_name):
    data_df = data_df.apply(modify_units, axis=1)
    new_df = restructure(data_df, company_name)
    # append_to_summary(summary_table_path, new_df)  
        # Ensure 'Year' and 'Company Name' are present in new_df
    # if 'Year' not in new_df.columns or 'Company Name' not in new_df.columns:
    #     raise ValueError("new_df must contain 'Year' and 'Company Name' columns")

    # # Go through each row in new_df
    # for index, new_row in new_df.iterrows():
    #     # Check if there is an existing row that matches the 'Year' and 'Company Name'
    #     match = (data_df['year'] == new_row['Year']) & (data_df['Company Name'] == new_row['Company Name'])
        
    #     if data_df[match].empty:
    #         # If there is no matching row, concatenate the new row
    #         data_df = pd.concat([data_df, pd.DataFrame([new_row])], ignore_index=True)
    #     else:
    #         # If there is a matching row, update the values
    #         data_df.loc[match, new_df.columns] = new_row
    return new_df



# %%

# # This block retrieves the directory where this script is located and the parent directory
# thisfile_dir = os.path.dirname(os.path.abspath(__file__))
# dir_cur = os.path.join(thisfile_dir, '..', '..')
# reports_dir = os.path.join(dir_cur, 'data', 'Reports')
# summary_path = os.path.join(dir_cur, 'outputs', 'Summary_table.xlsx')

# print("Target Directory:", dir_cur)

# # Iterate over all PDF files in the Reports directory
# for pdf_file in glob.glob(os.path.join(reports_dir, '*.pdf')):
#     base_name = os.path.basename(pdf_file)
    
#     # Define paths for the output files
#     txt_path = os.path.join(dir_cur, 'outputs', 'llama_parsed', f'{base_name}.txt')
#     xlsx_path = os.path.join(dir_cur, 'outputs', 'extracted_data', f'{base_name}.xlsx')
    
#     # Check if the PDF is already processed by checking if the output files exist
#     if os.path.exists(txt_path) and os.path.exists(xlsx_path):
#         print(f"Skipping {base_name}, already processed.")
#         continue
    
#     # Put the company name here, assuming it is derived from the PDF file name
#     company_name = base_name

#     # Process the PDF
#     print(f"Processing {base_name}...")
#     documents = convert_pdf_to_text(pdf_file, txt_path, input_llama_api)
#     convert_text_to_xlsx(documents, xlsx_path, input_openai_api)
#     convert_xlsx_to_summary(xlsx_path, summary_path, company_name)

# print("Processing complete.")