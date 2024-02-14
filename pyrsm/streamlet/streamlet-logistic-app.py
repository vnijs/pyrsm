import streamlit as st
import pandas as pd
import pyrsm as rsm
import io
from contextlib import redirect_stdout
from datetime import datetime

st.set_page_config(layout="wide")
st.set_option("deprecation.showPyplotGlobalUse", False)

tab1, tab2, tab3 = st.tabs(["Data", "Summary", "Plot"])


def view_df(df):
    st.dataframe(df)


def describe_df(df):
    if hasattr(df, "description"):
        st.markdown(df.description, unsafe_allow_html=True)


@st.cache_resource
def logistic_regression(df, X, y):
    return rsm.model.logistic(dataset=df, rvar=y, evar=X)


# @st.cache_data
def logistic_summary(_lr):
    out = io.StringIO()
    with redirect_stdout(out):
        _lr.summary()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    st.write(f"Date and time: {dt_string}")
    return out.getvalue()


# @st.cache_data
def logistic_plot(_lr, plots="or"):
    return _lr.plot(plots=plots)


@st.cache_data
def get_df(file):
    fname = file.name.split(".")[0]
    extension = file.name.split(".")[-1].upper()
    if extension == "CSV":
        df = pd.read_csv(file)
        code = f"""{fname} = pd.read_csv({file.name})"""
    elif extension in ["XLS", "XLSX"]:
        df = pd.read_excel(file, engine="openpyxl")
        code = f"""{fname} = pd.read_excel("{file.name}", engine="openpyxl")"""
    elif extension in ["PKL", "PICKLE"]:
        df = pd.read_pickle(file)
        code = f"""{fname} = pd.read_pickle("{file.name}")"""
    return df, fname, code


def main():

    file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx", "pkl"])
    if not file:
        st.sidebar.write("Upload a .csv, .xlsx or .pkl file to get started")
        # need return and a 'main' function to avoid actions the
        # rest of the code is not ready for yet
        return None

    df, fname, code = get_df(file)

    with tab1:
        view_df(df)
        describe_df(df)

    df_cols = df.columns.tolist()
    rvar = st.sidebar.selectbox("Response variable:", df.columns)
    df_cols.remove(rvar)
    evar = st.sidebar.multiselect("Explanatory variables:", df_cols)

    if len(evar) > 0:
        lr = logistic_regression(df, evar, rvar)
    else:
        lr = None

    with tab2:
        if lr:
            # st.text(logistic_summary(lr))
            st.code(logistic_summary(lr))
            st.code(
                f"""{code}\nlr = rsm.model.logistic(dataset={fname}, rvar="{rvar}", evar={evar})\nlr.summary()"""
            )

    with tab3:
        plot_types = {
            "Odds ratio": "or",
            "Prediction": "pred",
            "Variable important": "vimp",
        }
        plots = st.sidebar.selectbox("Plot types:", plot_types)
        if lr:
            st.pyplot(logistic_plot(lr, plots=plot_types[plots]))
            st.code(
                f"""{code}\nlr = rsm.logitic(dataset={fname}, rvar="{rvar}", evar={evar})\nlr.plot(plots="{plot_types[plots]}")"""
            )


main()

# to run this app use the below from a terminal
# then open a browser at the url localhost:8501
# streamlit run streamlit-logistic-app.py
