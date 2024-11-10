import streamlit as st

pages = {
    "Overview 🎈": [
        st.Page("pages/00_Welcome.py", title="DSS5105 BugBuster 👾")
    ],
    "Your Role 🧑🏻‍💼👩🏻‍💼": [
        st.Page("pages/01_Investor.py", title="Investor"),
        st.Page("pages/02_Regulator.py", title="Regulator")
    ],
    "Test Area 🔨": [
        st.Page("pages/03_ESG_Analysis_Tool📊.py", title="Template 📊"),
        st.Page("pages/04_playgound🧸.py", title="playground🧸")
    ]
}
pg = st.navigation(pages)
pg.run()
