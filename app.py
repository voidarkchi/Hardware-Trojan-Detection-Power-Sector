import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hardware Trojan Detection",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ Hardware Trojan Detection System")

st.write(
    "Machine-learning based detection of hardware Trojans "
    "using circuit-level structural and power features."
)


# =========================================================
# LOAD TRAINED MODEL AND CLEAN BASELINES
# =========================================================

MODEL_FILE = "hardware_trojan_rf.pkl"
BASELINE_FILE = "clean_baselines.pkl"

try:

    model = joblib.load(MODEL_FILE)
    baselines = joblib.load(BASELINE_FILE)

    # Feature names used during training
    feature_cols = baselines.columns.tolist()

except Exception as e:

    st.error(f"Could not load model files: {e}")
    st.stop()


# =========================================================
# DATA PREPROCESSING
# =========================================================

def preprocess_dataset(uploaded_file):

    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    if uploaded_file.name.lower().endswith(".xlsx"):

        full = pd.read_excel(uploaded_file)

    elif uploaded_file.name.lower().endswith(".csv"):

        full = pd.read_csv(uploaded_file)

    else:

        return None, 0, 0, 0, 0


    # Original number of records
    original_count = len(full)


    # -----------------------------------------------------
    # CLEAN LABEL
    # -----------------------------------------------------

    full["Label"] = (
        full["Label"]
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
    )


    # -----------------------------------------------------
    # CLEAN CIRCUIT NAME
    # -----------------------------------------------------

    full["Circuit"] = (
        full["Circuit"]
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
    )


    # -----------------------------------------------------
    # REMOVE EXACT DUPLICATE RECORDS
    # -----------------------------------------------------

    full = full.drop_duplicates().reset_index(drop=True)

    unique_count = len(full)


    # -----------------------------------------------------
    # HANDLE MISSING NUMERICAL VALUES
    # -----------------------------------------------------

    numeric_cols = (
        full
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    for col in numeric_cols:

        if full[col].isna().any():

            full[col] = full[col].fillna(
                full[col].median()
            )


    # -----------------------------------------------------
    # EXTRACT CIRCUIT FAMILY
    #
    # Example:
    #
    # S38417-T100  →  S38417
    # AES-T200     →  AES
    #
    # -----------------------------------------------------

    full["Family"] = (
        full["Circuit"]
        .str.replace(
            r"-T\d+$",
            "",
            regex=True
        )
        .str.upper()
    )


    # -----------------------------------------------------
    # FAMILIES FOR WHICH WE HAVE A CLEAN BASELINE
    # -----------------------------------------------------

    valid_families = baselines.index.tolist()


    # Number of records excluded because their family
    # does not have a Trojan-Free baseline

    before_family_filter = len(full)


    full = full[
        full["Family"].isin(valid_families)
    ].copy()


    excluded_count = (
        before_family_filter - len(full)
    )


    usable_count = len(full)


    return (
        full,
        original_count,
        unique_count,
        excluded_count,
        usable_count
    )


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📂 Upload Hardware Circuit Dataset")

uploaded_file = st.file_uploader(
    "Upload HEROdata2.xlsx",
    type=["xlsx", "csv"],
    accept_multiple_files=False
)


# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_file:

    (
        data,
        original_count,
        unique_count,
        excluded_count,
        usable_count
    ) = preprocess_dataset(uploaded_file)


    if data is None or len(data) == 0:

        st.error(
            "No valid data found in the uploaded file."
        )

        st.stop()


    # =====================================================
    # DATASET PREPROCESSING SUMMARY
    # =====================================================

    st.subheader("📊 Dataset After Preprocessing")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Original Records",
            original_count
        )


    with col2:

        st.metric(
            "Unique Records",
            unique_count
        )


    with col3:

        st.metric(
            "Excluded",
            excluded_count
        )


    with col4:

        st.metric(
            "Usable Records",
            usable_count
        )


    st.info(
        "Duplicate records are removed first. "
        "Circuit families without a Trojan-Free baseline "
        "are then excluded before prediction."
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    predictions = []
    probabilities = []


    for _, row in data.iterrows():

        family = row["Family"]


        # -------------------------------------------------
        # GET CLEAN BASELINE FOR THIS CIRCUIT FAMILY
        # -------------------------------------------------

        baseline = baselines.loc[family]


        # -------------------------------------------------
        # GET MODEL FEATURES
        # -------------------------------------------------

        circuit_features = (
            row[feature_cols]
            .astype(float)
        )


        # -------------------------------------------------
        # BASELINE NORMALIZATION
        #
        # z = (x - baseline) / baseline
        #
        # -------------------------------------------------

        normalized = (
            (circuit_features - baseline)
            /
            baseline.replace(
                0,
                np.nan
            )
        ).fillna(0)


        # -------------------------------------------------
        # TROJAN PROBABILITY
        # -------------------------------------------------

        probability = model.predict_proba(
            normalized.to_frame().T
        )[0, 1]


        # -------------------------------------------------
        # DECISION THRESHOLD
        # -------------------------------------------------

        threshold = 0.70


        if probability >= threshold:

            prediction = "Trojan Infected"

        else:

            prediction = "Trojan Free"


        predictions.append(prediction)

        probabilities.append(probability)


    # =====================================================
    # ADD RESULTS TO DATAFRAME
    # =====================================================

    data["Prediction"] = predictions

    data["Trojan Probability"] = probabilities


    # =====================================================
    # RESULT COUNTS
    # =====================================================

    infected_count = (
        data["Prediction"]
        == "Trojan Infected"
    ).sum()


    clean_count = (
        data["Prediction"]
        == "Trojan Free"
    ).sum()


    # =====================================================
    # DETECTION RESULTS
    # =====================================================

    st.subheader("🔍 Detection Results")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Circuits",
            len(data)
        )


    with col2:

        st.metric(
            "Trojan Infected",
            infected_count
        )


    with col3:

        st.metric(
            "Trojan Free",
            clean_count
        )


    # =====================================================
    # RESULT TABLE
    # =====================================================

    st.subheader("📋 Circuit-Level Predictions")


    result_columns = [
        "Circuit",
        "Family",
        "Prediction",
        "Trojan Probability"
    ]


    result_df = data[
        result_columns
    ].copy()


    result_df["Trojan Probability"] = (
        result_df["Trojan Probability"]
        .round(4)
    )


    st.dataframe(
        result_df,
        use_container_width=True,
        height=500
    )


    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================

    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Predictions",
        data=csv_data,
        file_name="final_trojan_predictions.csv",
        mime="text/csv"
    )


    # =====================================================
    # FINAL STATUS
    # =====================================================

    st.success(
        f"Analysis completed successfully — "
        f"{usable_count} unique usable circuits analyzed."
    )


else:

    st.info(
        "Upload HEROdata2.xlsx to begin analysis."
    )