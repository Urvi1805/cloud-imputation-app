import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Data Imputation Tool", layout="wide")

if "imputation_results" not in st.session_state:
    st.session_state.imputation_results = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

st.title("🧠 Cloud Data Imputation System")
st.write("Upload a dataset and compare imputation techniques (Mean vs KNN)")


# ---------------------------------
# FILE UPLOAD
# ---------------------------------
file = st.file_uploader("cardio.csv", type=["csv"])

if file:
    upload_id = f"{getattr(file, 'name', 'upload')}_{getattr(file, 'size', 0)}"
    if st.session_state.last_upload_id != upload_id:
        st.session_state.imputation_results = None
        st.session_state.last_upload_id = upload_id

    # Semicolon-separated CSVs (common in EU) parse as one column with default comma sep — no numeric cols.
    df = pd.read_csv(file, sep=None, engine="python")
    if df.shape[1] == 1:
        file.seek(0)
        df = pd.read_csv(file, sep=";")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------
    # MISSING VALUES
    # ---------------------------------
    st.subheader("🔍 Missing Values Summary")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])

    # Create copy
    df_missing = df.copy()

    # ---------------------------------
    # IMPUTATION
    # ---------------------------------
    st.subheader("⚙️ Apply Imputation")

    method = st.selectbox(
        "Choose Method",
        ["Mean Imputation", "KNN Imputation"]
    )

    if st.button("Run Imputation"):

        num_cols = df.select_dtypes(include=np.number)
        if num_cols.empty:
            st.error(
                "No numeric columns found. If your file uses `;` as separator, re-upload after saving "
                "as comma-separated, or the parser failed — check the preview above."
            )
            st.stop()

        # Scale data for KNN
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(num_cols),
            columns=num_cols.columns,
        )

        # ---------------------------------
        # MEAN IMPUTATION
        # ---------------------------------
        mean_imputer = SimpleImputer(strategy="mean")
        mean_imputed = pd.DataFrame(
            mean_imputer.fit_transform(df_scaled),
            columns=df_scaled.columns,
        )

        # ---------------------------------
        # KNN IMPUTATION
        # ---------------------------------
        knn_imputer = KNNImputer(n_neighbors=5)
        knn_imputed = pd.DataFrame(
            knn_imputer.fit_transform(df_scaled),
            columns=df_scaled.columns,
        )

        # RMSE only where original values were observed (flattened — reliable vs DataFrame boolean mask)
        mask = df_scaled.isnull().to_numpy()
        observed = ~mask.ravel()
        orig_flat = np.asarray(df_scaled, dtype=float).ravel()
        mean_flat = np.asarray(mean_imputed, dtype=float).ravel()
        knn_flat = np.asarray(knn_imputed, dtype=float).ravel()

        def calculate_rmse(orig, imp):
            if not observed.any():
                return float("nan")
            return float(
                np.sqrt(mean_squared_error(orig[observed], imp[observed]))
            )

        results = {
            "Mean": calculate_rmse(orig_flat, mean_flat),
            "KNN": calculate_rmse(orig_flat, knn_flat),
        }
        results_df = pd.DataFrame(list(results.items()), columns=["Method", "RMSE"])

        st.session_state.imputation_results = {
            "results_df": results_df,
            "df_missing": df_missing.copy(),
            "mean_imputed": mean_imputed,
            "knn_imputed": knn_imputed,
            "columns": list(df_scaled.columns),
        }

    # Persisted after button reruns (selectbox / downloads still work)
    res = st.session_state.imputation_results
    if res is not None:
        st.subheader("📈 Comparison of Imputation Techniques (RMSE)")
        st.dataframe(res["results_df"], use_container_width=True)
        
        # Heatmap: full grid is too large for tens of thousands of rows — sample for display
        st.subheader("🔥 Missing Data Heatmap")
        miss = res["df_missing"]
        heat_n = min(500, len(miss))
        miss_sample = miss.isnull().iloc[:heat_n]
        st.caption(
            f"Showing first {heat_n:,} of {len(miss):,} rows (full-row heatmap would be unreadable)."
        )
        fig2, ax2 = plt.subplots(figsize=(min(16, 8 + heat_n / 200), 4.5))
        sns.heatmap(
            miss_sample.T,
            cbar=True,
            ax=ax2,
            cmap="YlOrRd",
            yticklabels=True,
            xticklabels=False,
        )
        ax2.set_xlabel("Row index (sample)")
        ax2.set_ylabel("Column")
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)

        st.subheader("📊 Distribution Comparison")

        column = st.selectbox("Select column", res["columns"], key="dist_col")

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        m_series = res["mean_imputed"][column]
        k_series = res["knn_imputed"][column]
        if m_series.nunique() <= 1 and k_series.nunique() <= 1:
            ax3.bar(["Mean", "KNN"], [float(m_series.iloc[0]), float(k_series.iloc[0])])
            ax3.set_ylabel("Value")
        else:
            sns.kdeplot(m_series, label="Mean", ax=ax3)
            sns.kdeplot(k_series, label="KNN", ax=ax3)
            ax3.legend()
        ax3.set_title(f"Column: {column}")
        fig3.tight_layout()
        st.pyplot(fig3, clear_figure=True)

        st.subheader("⬇️ Download Results")

        out = res["mean_imputed"] if method == "Mean Imputation" else res["knn_imputed"]
        csv = out.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Imputed Data",
            csv,
            "imputed_data.csv",
            "text/csv",
            key="dl_imputed",
        )
