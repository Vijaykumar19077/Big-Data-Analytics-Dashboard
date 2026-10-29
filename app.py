
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Big Data Explorer",
    page_icon="📊",
    layout="wide"
)

# ---------------- SAMPLE DATASET ----------------
@st.cache_data
def create_sample_data():
    rng = np.random.default_rng(42)
    n = 25000

    dates = pd.date_range(
        start="2024-01-01",
        end="2025-12-31",
        freq="h"
    )

    selected_dates = rng.choice(dates, size=n)

    units = rng.integers(1, 10, size=n)
    prices = np.round(
        rng.lognormal(mean=3.2, sigma=0.65, size=n), 2
    )

    df = pd.DataFrame({
        "order_id": np.arange(100001, 100001 + n),
        "order_date": pd.to_datetime(selected_dates),
        "category": rng.choice(
            ["Electronics", "Clothing", "Home", "Books", "Sports"],
            size=n
        ),
        "region": rng.choice(
            ["North", "South", "East", "West", "Central"],
            size=n
        ),
        "units_sold": units,
        "unit_price": prices,
        "sales": np.round(units * prices, 2),
        "customer_rating": np.round(rng.uniform(1, 5, n), 1),
        "payment_method": rng.choice(
            ["UPI", "Card", "Cash", "Net Banking"],
            size=n
        )
    })

    # Add missing values and duplicate records for cleaning practice
    for col in ["category", "region", "customer_rating"]:
        missing_indices = rng.choice(
            df.index, size=40, replace=False
        )
        df.loc[missing_indices, col] = np.nan

    df = pd.concat(
        [df, df.sample(30, random_state=42)],
        ignore_index=True
    )

    return df


# ---------------- APPLICATION HEADER ----------------
st.title("📊 Big Data Explorer")
st.markdown(
    "### Big Data Processing and Analytics Dashboard"
)
st.write(
    "Upload, clean, analyze, visualize, and export datasets "
    "using interactive analytics tools."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Dashboard Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data cleaning")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows", value=True
)
fill_missing = st.sidebar.checkbox(
    "Fill missing values", value=True
)

# ---------------- LOAD DATA ----------------
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()
else:
    raw_df = create_sample_data()

if raw_df.empty or len(raw_df.columns) == 0:
    st.warning("The uploaded dataset is empty.")
    st.stop()

df = raw_df.copy()

original_rows = len(df)
original_columns = len(df.columns)
duplicate_count = int(df.duplicated().sum())
missing_count = int(df.isna().sum().sum())

# ---------------- DATA CLEANING ----------------
if remove_duplicates:
    df = df.drop_duplicates()

if fill_missing:
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                df[col] = df[col].fillna(
                    median_value if pd.notna(median_value) else 0
                )
            else:
                mode_value = df[col].mode(dropna=True)
                replacement = (
                    mode_value.iloc[0] if not mode_value.empty
                    else "Unknown"
                )
                df[col] = df[col].fillna(replacement)

# ---------------- KEY METRICS ----------------
st.subheader("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows Loaded", f"{original_rows:,}")
col2.metric("Total Columns", f"{original_columns:,}")
col3.metric("Duplicate Rows", f"{duplicate_count:,}")
col4.metric("Missing Cells", f"{missing_count:,}")

st.caption(
    f"After cleaning: {len(df):,} rows × {len(df.columns):,} columns"
)

# ---------------- DATA PREVIEW ----------------
st.subheader("🔍 Dataset Preview")

with st.expander("View dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption("Showing up to the first 100 rows.")

# ---------------- FILTERS ----------------
st.subheader("🎛️ Filter Your Data")

filtered_df = df.copy()

categorical_cols = [
    col for col in df.columns
    if (
        pd.api.types.is_object_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    )
]

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

date_cols = [
    col for col in df.columns
    if (
        pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in col.lower()
        or "time" in col.lower()
    )
]

filter_left, filter_right = st.columns(2)

with filter_left:
    if categorical_cols:
        category_col = st.selectbox(
            "Choose a category column",
            ["None"] + categorical_cols
        )

        if category_col != "None":
            available_values = sorted(
                df[category_col].dropna().astype(str).unique().tolist()
            )

            selected_values = st.multiselect(
                "Choose values",
                available_values,
                default=available_values
            )

            filtered_df = filtered_df[
                filtered_df[category_col].astype(str).isin(
                    selected_values
                )
            ]

with filter_right:
    valid_date_cols = []

    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            valid_date_cols.append(col)

    if valid_date_cols:
        date_col = st.selectbox(
            "Choose a date column",
            ["None"] + valid_date_cols
        )

        if date_col != "None":
            parsed_dates = pd.to_datetime(
                df[date_col], errors="coerce"
            )

            min_date = parsed_dates.min().date()
            max_date = parsed_dates.max().date()

            date_range = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_dates = pd.to_datetime(
                    filtered_df[date_col], errors="coerce"
                )

                filtered_df = filtered_df[
                    filtered_dates.notna()
                    & (filtered_dates.dt.date >= start_date)
                    & (filtered_dates.dt.date <= end_date)
                ]

st.info(f"Records matching your filters: {len(filtered_df):,}")

# ---------------- STATISTICS ----------------
st.subheader("📈 Statistical Analysis")

if numeric_cols and not filtered_df.empty:
    selected_numeric = st.selectbox(
        "Select a numeric column",
        numeric_cols
    )

    series = pd.to_numeric(
        filtered_df[selected_numeric], errors="coerce"
    ).dropna()

    if not series.empty:
        stat1, stat2, stat3, stat4 = st.columns(4)

        stat1.metric("Average", f"{series.mean():,.2f}")
        stat2.metric("Median", f"{series.median():,.2f}")
        stat3.metric("Minimum", f"{series.min():,.2f}")
        stat4.metric("Maximum", f"{series.max():,.2f}")

        with st.expander("Full statistical summary"):
            st.dataframe(
                series.describe().to_frame("Value"),
                use_container_width=True
            )
else:
    st.info("No numeric data available for statistical analysis.")

# ---------------- VISUALIZATIONS ----------------
st.subheader("📊 Interactive Visualizations")

if not filtered_df.empty:
    chart1, chart2 = st.columns(2)

    with chart1:
        if categorical_cols:
            chart_category = st.selectbox(
                "Category for bar chart",
                categorical_cols,
                key="bar_category"
            )

            counts = (
                filtered_df[chart_category]
                .astype(str)
                .value_counts()
                .head(20)
                .rename_axis(chart_category)
                .reset_index(name="Record Count")
            )

            fig_bar = px.bar(
                counts,
                x=chart_category,
                y="Record Count",
                color=chart_category,
                title=f"Records by {chart_category}"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        elif numeric_cols:
            fig_bar = px.histogram(
                filtered_df,
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with chart2:
        if numeric_cols:
            chart_numeric = st.selectbox(
                "Numeric column for histogram",
                numeric_cols,
                key="hist_numeric"
            )

            fig_hist = px.histogram(
                filtered_df,
                x=chart_numeric,
                nbins=30,
                title=f"Distribution of {chart_numeric}"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.markdown("#### Correlation Heatmap")

        correlation = filtered_df[numeric_cols].corr(
            numeric_only=True
        )

        fig_heatmap = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Relationships Between Numeric Variables",
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("No records match your filters. Change the filters to continue.")

# ---------------- DATA QUALITY ----------------
st.subheader("🧹 Data Quality Report")

quality_report = pd.DataFrame({
    "Column": raw_df.columns,
    "Data Type": [str(raw_df[c].dtype) for c in raw_df.columns],
    "Missing Values": [int(raw_df[c].isna().sum()) for c in raw_df.columns],
    "Missing Percentage": [
        round(raw_df[c].isna().mean() * 100, 2)
        for c in raw_df.columns
    ],
    "Unique Values": [int(raw_df[c].nunique(dropna=True)) for c in raw_df.columns]
})

st.dataframe(quality_report, use_container_width=True)

# ---------------- DOWNLOADS ----------------
st.subheader("⬇️ Download Your Results")

download_left, download_right = st.columns(2)

with download_left:
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Processed Dataset",
        data=csv_data,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

with download_right:
    if numeric_cols and not filtered_df.empty:
        summary_data = (
            filtered_df[numeric_cols]
            .describe()
            .to_csv()
            .encode("utf-8")
        )

        st.download_button(
            "Download Statistical Summary",
            data=summary_data,
            file_name="statistical_summary.csv",
            mime="text/csv"
        )

# ---------------- PROJECT INFORMATION ----------------
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    **Objective:** Demonstrate a data analytics workflow through an
    interactive dashboard.

    **Technologies:** Python, Streamlit, Pandas, NumPy and Plotly.

    **Processing stages:**
    1. Data ingestion from CSV files
    2. Data cleaning and preprocessing
    3. Descriptive statistical analysis
    4. Interactive visualization
    5. Data quality assessment
    6. Export of processed results

    **Important limitation:** This version processes data on a single
    machine. It demonstrates Big Data analytics concepts but does not
    implement distributed processing with Hadoop or Apache Spark.
    A PySpark integration can be added as a future enhancement.
    """)

st.divider()
st.caption(
    "Big Data Explorer | BSc IT College Project | Python + Streamlit"
)
