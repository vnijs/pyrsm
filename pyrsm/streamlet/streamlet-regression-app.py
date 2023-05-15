import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import pyrsm as rsm
from datetime import datetime


def view_df(df):
    st.write("### Data:")
    st.write(df)


# @st.cache
def run_regression(df):
    st.write("### Select variables for linear Regression:")
    y = st.selectbox("Response variable:", df.columns)
    X = st.multiselect("Explanatory variables:", rsm.setdiff(df.columns, y))

    if X and y:
        form = f"{y} ~ {' + '.join(X)}"
        model = smf.ols(form, data=df).fit()

        st.write("### Summary of Linear regression (OLS):")
        st.write(model.summary())
        dt_string = datetime.now.strftime("%d/%m/%Y %H:%M:%S")
        st.write(f"Date and time: {dt_string}")


def sample_df(df):
    frac = st.slider("Random sample (%)", 1, 100, 100)

    if frac < 100:
        df = df.sample(frac=frac / 100, random_state=1234)

    return df


@st.cache
def get_df(file):
    extension = file.name.split(".")[-1].upper()
    if extension == "CSV":
        df = pd.read_csv(file)
    elif extension in ["XLS", "XLSX"]:
        df = pd.read_excel(file, engine="openpyxl")
    elif extension in ["PKL", "PICKLE"]:
        df = pd.read_pickle(file)
    return df


def main():
    st.write("## Regression model - Streamlit app")

    file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "pickle"])
    if not file:
        st.sidebar.write("Upload a .csv, .xlsx or .pickle file to get started")
        # need return and a 'main' function to avoid actions the
        # rest of the code is not ready for yet
        return None

    df = get_df(file)
    df = sample_df(df)

    st.sidebar.title("Visualization Selector")
    select_status = st.sidebar.radio("Function", ("View Data", "Regression(OLS)"))

    if select_status == "View Data":
        view_df(df)
    else:
        run_regression(df)


main()

# to run this app use the below from a terminal
# then open a browser at the url localhost:8501

# streamlit run streamlit-regression-app.py
