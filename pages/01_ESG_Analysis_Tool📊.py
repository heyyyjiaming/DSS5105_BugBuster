import streamlit as st
from llama_parse import LlamaParse


# def convert_pdf_to_txt(input_file, txt_output_path):
#     documents = LlamaParse(result_type="markdown").load_data(input_file)
#     # Merge all the text into one str
#     all_text = []
#     for doc in documents:
#         all_text.append(doc.text)

#     merged_doc = '\n\n'.join(all_text)
#     # Save as txt
#     txt_output_path = os.path.join(dir_cur, 'outputs/llama_parsed', f'{base_name}.txt')
#     with open(txt_output_path, 'w', encoding='utf-8') as file:
#         file.write(merged_doc)




st.title("ESG Analysis Tool 📊")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
llama_api_key = st.text_input("Llama Cloud API Key", type="password")
if not (openai_api_key and llama_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key to continue.", icon="🗝️")
else:

    uploaded_file = st.file_uploader(
        "Upload a document (PDF)", type=("pdf")
    )

    if uploaded_file:
        st.write("File uploaded successfully!")

