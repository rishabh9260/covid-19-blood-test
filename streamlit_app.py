import streamlit as st
import pickle
import numpy
import xgboost

st.title("Covid-19 Prediction using Blood Test")

with st.form("blood_test_form"):
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        hcm = st.number_input(label="HCM", min_value=0.0, max_value=1000.0, value=0.0)
        hemoglobin = st.number_input(label="Hemoglobin", min_value=0.0, max_value=1000.0, value=0.0)
        mchc = st.number_input(label="MCHC", min_value=0.0, max_value=1000.0, value=0.0)
        rdw_cv = st.number_input(label="RDW-CV", min_value=0.0, max_value=1000.0, value=0.0)
    with col2:
        rdw_sd = st.number_input(label="RDW-SD", min_value=0.0, max_value=1000.0, value=0.0)
        vcm = st.number_input(label="VCM", min_value=0.0, max_value=1000.0, value=0.0)
        basophils = st.number_input(label="Basophils", min_value=0.0, max_value=1000.0, value=0.0)
        eosinophils = st.number_input("Eosinophils", min_value=0.0, max_value=1000.0, value=0.0)
    with col3:
        erythroblasts = st.number_input(label="Erythroblasts", min_value=0.0, max_value=1000.0, value=0.0)
        erythrocytes = st.number_input(label="Erythrocytes", min_value=0.0, max_value=1000.0, value=0.0)
        hematocrit = st.number_input(label="Hematocrit", min_value=0.0, max_value=1000.0, value=0.0)
        leukocytes = st.number_input(label="Leukocytes", min_value=0.0, max_value=1000.0, value=0.0)
    with col4:
        lymphocytes = st.number_input(label="Lymphocytes", min_value=0.0, max_value=1000.0, value=0.0)
        monocytes = st.number_input(label="Monocytes", min_value=0.0, max_value=1000.0, value=0.0)
        neutrophils = st.number_input(label="Neutrophils", min_value=0.0, max_value=1000.0, value=0.0)
        DAY_DIFFERENCE = st.number_input(label="Days Since Symptoms", min_value=0, max_value=14, value=0)

    NRS_values = {
        "Anisocitose +": 0,
        "Anisocitose ++": 0,
        "Anisocitose +++": 0,
        "Erythrocytes normal in size": 0,
        "Macrocitose +": 0,
        "Macrocitose ++": 0,
        "Macrocitose +++": 0,
        "Microcitose +": 0,
        "Microcitose ++": 0,
        "Microcitose +++": 0,
    }
    NRS_option = st.selectbox("Note Red Series observation", list(NRS_values.keys()))
    NRS_values[NRS_option] = 1

    submitted = st.form_submit_button("Submit")

predict_data = [hcm, hemoglobin, mchc, rdw_cv, rdw_sd, vcm, basophils,
                eosinophils, erythroblasts, erythrocytes, hematocrit,
                leukocytes, lymphocytes, monocytes, neutrophils,
                DAY_DIFFERENCE, NRS_values['Anisocitose +'],
                NRS_values['Anisocitose ++'],
                NRS_values['Anisocitose +++'],
                NRS_values['Erythrocytes normal in size'],
                NRS_values['Macrocitose +'],
                NRS_values['Macrocitose ++'],
                NRS_values['Macrocitose +++'],
                NRS_values['Microcitose +'],
                NRS_values['Microcitose ++'],
                NRS_values['Microcitose +++']]


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    return pickle.load(open("covid-xgb.pickle.dat", "rb"))


@st.cache(show_spinner=False)
def predict(data):
    with st.spinner('Predicting...'):
        classifier = load_model()
        probability = classifier.predict_proba(numpy.array(data).reshape((1, -1)))
    return probability


if submitted:
    prediction_proba = predict(predict_data)
    if prediction_proba[0][0] > 0.5:
        st.success("Covid-19 Negative")
        st.info(f"Probability: {(prediction_proba[0][0] * 100):.2f}")
    else:
        st.error("Covid-19 Positive")
        st.info(f"Probability: {(prediction_proba[0][1] * 100):.2f}")
else:
    st.write('Fill the values and click on "Submit" to predict covid-19 status.')
