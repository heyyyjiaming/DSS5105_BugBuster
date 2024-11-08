import streamlit as st

st.title("ESG Analysis Tool 📊")
st.write(
    "Upload an ESG report below and see how well the company performs! "
    "To use this app, you may need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:


    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("pdf")
    )

    if uploaded_file:
        st.write("File uploaded successfully!")

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        # messages = [
        #     {
        #         "role": "user",
        #         "content": f"Here's a document: {document} \n\n---\n\n {question}",
        #     }
        # ]

        # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     stream=True,
        # )

        # Stream the response to the app using `st.write_stream`.
        # st.write_stream(stream)