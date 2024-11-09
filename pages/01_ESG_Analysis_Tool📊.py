import streamlit as st
# import streamlit_scrollable_textbox as stx
import os
import glob
from utils.extract import convert_pdf_to_text
# from extract import convert_text_to_xlsx  
from utils.tools import read_pdf



st.title("ESG Analysis Tool 📊")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide some API keys below. "
)

input_openai_api_key = st.text_input("OpenAI API Key", type="password")
input_llama_api_key = st.text_input("Llama Cloud API Key", type="password")

if not (input_openai_api_key and input_llama_api_key):
    st.info("Please add your OpenAI & Llama Cloud API key to continue.", icon="🗝️")
else:
    uploaded_file = st.file_uploader("Upload a document (PDF)", type=("pdf"))

    if uploaded_file:
        st.write("File uploaded successfully!")
        ## Test importing pkg
        # context = read_pdf(uploaded_file)
        context = convert_pdf_to_text(uploaded_file, input_llama_api_key)
        st.markdown("**Here is the content of the PDF file 📄:**")
        # stx.scrollableTextbox(context,height = 500)
        st.write(context)



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



