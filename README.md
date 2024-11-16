# DSS5105_BugBuster

## Team Member (in alphabetic order):
Bi Ying, Chen Zhujun, Ding Jiaming, Huang Yuxin, Li Jingming, Niu Muyuan, Zhang yi

## Project Description
This an automatic system that extracts ESG information from unstructured reports and provides a comprehensive analysis of ESG performance within selected industries. By incorporating advanced natural language processing (NLP) techniques and data analysis, the tool helps streamline the ESG data extraction process, improve data quality, and offer valuable insights into corporate sustainability practices.

## Streamlit Online App

Welcome to the **[ESG Data Extraction Tool](https://dss5105bugbuster-vungkn2vd4zbfouiwsr55u.streamlit.app/)**! This user-friendly web application is designed to help you easily extract and analyze ESG (Environmental, Social, and Governance) data from PDF reports.

![](https://github.com/heyyyjiaming/DSS5105_BugBuster/blob/main/image/app.png)


### How to Use
To get started with the ESG Data Extraction Tool, please refer to our detailed instructions available here: [Getting Started Guide](https://github.com/heyyyjiaming/DSS5105_BugBuster/wiki/6.-Getting-Started). This guide will walk you through the necessary steps, including how to input your API key, upload files, and review your results.

## :computer: Installation (For running code locally)

1. Create a conda environment on your terminal using:
    ```
    conda create -n extractESG python=3.10 -y
    conda activate extractESG  # some OS requires `source activate extractESG`
    ```

2. Install the compatible pytorch based on your OS.
    - If you are on a GPU, install [pytorch based on your cuda version](https://pytorch.org/get-started/locally/). You can find your CUDA version via `nvcc -V`.
        ```
        pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121  # cu121 means cuda 12.1
        ```
    - If you are on a CPU instance,
        ```
        pip3 install torch
        ```

3. Install `uniflow`:
    ```
    pip install uniflow
    ```
4. Install `llama-parse`:
    ```
    pip install llama-parse
    ```
5. Get API keys:
 - Login and get a Llama model api-key from [**https://cloud.llamaindex.ai/api-key ↗**](https://cloud.llamaindex.ai/api-key). (For converting PDF to text)
 - Login and get an Open AI api-key from [**https://platform.openai.com/settings/organization/api-keys ↗**](https://platform.openai.com/settings/organization/api-keys). (For extracting and standarding data)
 - Login and get a Serper api-key from [**https://serper.dev/login ↗**](https://serper.dev/login). (For collecting reports)
   
- (Optional) If you are running on your computer, you can set up your OpenAI API key first. To do so, create a `.env` file in `D:/apikeys/.env` Then add the following line to the `.env` file:
     ```
     OPENAI_API_KEY = sk-.....
     LLAMA_API_KEY = llx-....
     ```
## Data Extraction Process
By following our structured approach to ESG information extraction, we ensure that our data is not only accurate but also organized in a way that supports comprehensive analysis and reporting. Below are the main components of our ESG information extraction process:

![](https://github.com/heyyyjiaming/DSS5105_BugBuster/blob/main/image/5105.jpg)


## Output Examples

Once the script runs successfully, it will generate several output files:

- **One `.txt` file** in the directory `/outputs/llama_parsed`, which contains the text extracted from your PDFs.
  - **Example Output**: 
    ```
    # Our Impact from Climate Change

    Global warming leading to national disasters impacting our business and infrastructure


    # Environment

    # Climate Change and Carbon

    As it is important to understand our initiatives in the past and how they interrelate to our current focus and strategy going forward, we have         summarised the major actions, insights and milestones in our climate strategy and journey (see Table 1).

    |YEAR|INITIATIVES/|RELEVANT INSIGHTS INTO|OUR ACTIONS|REPORT|
    |---|---|---|---|---|
    |2010|Progressive depth and breadth of carbon disclosure, reporting and external assurance|Understanding our carbon footprint and drivers.|Continuous         refinement and validation to achieve a comprehensive view of our carbon footprint.|Sustainability Reports and CDP|
    |2013|Founding member of the Australian Business Roundtable for Disaster Resilience and Safer Communities (ABR)|Research and insights into social and     economic impact of natural disasters.|Five research reports.|ABR Reports from 2013 to November 2017|
    |2015|Stakeholder and Materiality Assessment|Climate change emerged as a topic of moderate importance and moderate impact.|Mid-term energy and carbon     intensity targets set for 2020 and 2030: improvement by 30% and 50% respectively. Widen depth and scope of carbon reporting to CDP for Singapore and Australia.|SR2016|
    |2016|Life Cycle Assessment|Climate change and carbon were the most material environmental issues of concern. Almost two-thirds of our carbon footprint were in our supply chain. This was highly relevant in setting science-based Scope 3 carbon reduction targets.|Environment strategy updated to     strengthen focus on climate change and carbon. Climate risk and carbon assessment updated for our Sustainable Supply Chain Management framework.|SR2016|

    ```

- **One `.xlsx` file** in the directory `/outputs/extracted_data`, which contains the extracted ESG data organized into five key features.
  - **Example Output**: 


    | label                     | metric                       | unit   | year | value |
    --------------------------|------------------------------|--------|------|-------|
    | Greenhouse Gas Emissions  | Scope 2                     | tCO2e  | 2021 | 8500  |
    | Energy Consumption        | Total energy consumption     | MWhs   | 2021 | 120000|
    | Water Consumption         | Total water consumption      | ML     | 2021 | 1500  |

    This table can be combined with the previous example to create a comprehensive view of the ESG data. If you need them merged or formatted 
differently, just let me know!

    This table can be included in the output section of your documentation to provide clear and structured information about the extracted ESG data.

- **A summary of the data** will be appended in the `Summary_table.xlsx` located in the `/outputs` directory.
  - **Example Output**: 

    | Company Name | Year | GHG Emissions (Scope 1) (tCO2e) | GHG Emissions (Scope 2) (tCO2e) | GHG Emissions (Scope 3) (tCO2e) | GHG Emissions (Total) (tCO2e) | Total Energy Consumption (MWhs) | Total Water Consumption (ML) | Total Waste Generated (t) | Current Employees by Gender (Female %) | New Hires and Turnover by Gender (Female %) | Total Turnover (%) | Total Number of Employees | Average Training Hours per Employee | Fatalities | High-consequence injuries | Recordable injuries | Recordable work-related ill health cases | Board Independence (%) | Women on the Board (%) | Women in Management Team (%) | Anti-Corruption Training for Employees (%) |
    |--------------|------|----------------------------------|----------------------------------|----------------------------------|--------------------------------|----------------------------------|------------------------------|--------------------------|-----------------------------------------|----------------------------------------------|--------------------|--------------------------|-----------------------------------|------------|--------------------------|-------------------|----------------------------------------------|----------------------|------------------------|-----------------------------|----------------------------------------------|
    | Singtel      | 2020 | 5000                             | 168679                           | 7000                             | 162566                         | 917090.7337                     | 8646.465                     | 8541                     | 45                                      | 4.6                                          | 39.6               | 24000                    | 45000000                          | 0          |                          | 14                |                                              | 28                   |                        |                             |                                              |
    | Singtel      | 2021 | 5749                             | 110292                           | 7809                         | 3613093                       | 808072.8687                     | 1500                        | 4921                     | 45                                      | 10.5                                         | 13.2               | 20078                    | 48.3                              | 0          | 17                       | 10                | 60                                           | 30                   | 40                     | 80                          |                                              |

These output files provide a comprehensive view of the ESG data extracted from the reports, making it easy to analyze and report on sustainability metrics.
