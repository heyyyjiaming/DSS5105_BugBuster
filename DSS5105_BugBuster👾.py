import streamlit as st

pages = {
    "👀 Overview": [
        st.Page("app_pages/00_Welcome.py", title="DSS5105 BugBuster 👾"),
        st.Page("app_pages/041_demo🧸.py", title="ESGenius demo 🧸")
    ],
    "🧑🏻‍💼👩🏻‍💼 I am ...": [
        st.Page("app_pages/01_Investor.py", title="Investor"),
        st.Page("app_pages/02_Regulator.py", title="Regulator")
    ],
    # "🔨 Test Area": [
    #     st.Page("app_pages/03_MultiLingual.py", title="ESGenius_MultiLingual📊"),
    #     st.Page("app_pages/04_playgound🪀.py", title="playground🪀"),
    #     st.Page("app_pages/05_backup.py", title="backup📑")
    # ]
}
pg = st.navigation(pages)
pg.run()
