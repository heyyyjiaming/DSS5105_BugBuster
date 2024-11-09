import PyPDF2

# class read_pdf:
#     def __init__(self, path):
#         self.path = path

#     def pdf_to_text(self):
#         # Open the PDF file in read-binary mode
#         with open(self.path, 'rb') as pdf_file:
#             # Create a PdfReader object instead of PdfFileReader
#             pdf_reader = PyPDF2.PdfReader(pdf_file)

#             # Initialize an empty string to store the text
#             text = ''
#             for page_num in range(len(pdf_reader.pages)):
#                 page = pdf_reader.pages[page_num]
#                 text += page.extract_text()
                
#         return text

# def read_pdf(pdf_path):
#     # Open the PDF file in read-binary mode
#     with open(pdf_path, 'rb') as pdf_file:
#         # Create a PdfReader object instead of PdfFileReader
#         pdf_reader = PyPDF2.PdfReader(pdf_file)

#         # Initialize an empty string to store the text
#         text = ''
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
        
#     print("PDF converted to text successfully!")

#     return text

def read_pdf(pdf_file):
    # Open the PDF file in read-binary mode

    # Create a PdfReader object instead of PdfFileReader
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Initialize an empty string to store the text
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
        
    print("PDF converted to text successfully!")

    return text

