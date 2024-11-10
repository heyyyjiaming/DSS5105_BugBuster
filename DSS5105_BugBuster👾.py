import streamlit as st

pages = {
    "Overview 👀": [
        st.Page("app_pages/00_Welcome.py", title="DSS5105 BugBuster 👾")
    ],
    "Your Role 🧑🏻‍💼👩🏻‍💼": [
        st.Page("app_pages/01_Investor.py", title="Investor"),
        st.Page("app_pages/02_Regulator.py", title="Regulator")
    ],
    "Test Area 🔨": [
        st.Page("app_pages/03_ESG_Analysis_Tool.py", title="Template 📊"),
        st.Page("app_pages/04_playgound🧸.py", title="playground🧸")
    ]
}
pg = st.navigation(pages)
pg.run()
