from shiny import App, ui, render, reactive
import duckdb
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
matplotlib.use("Agg")  # Use non-interactive backend for Shiny


# UI: Define the user interface
app_ui = ui.page_fluid(
    ui.panel_title("Exploratory Data Analysis App"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Controls"),
            ui.input_file("file", "Upload CSV File", accept=[".csv"], multiple=False),
            ui.output_text("file_info"),
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Dataset Overview",
                ui.h4("Dataset Overview"),
                ui.output_data_frame("dataset_overview"),
            ),
            ui.nav_panel(
                "Column Details",
                ui.h4("Column Details"),
                ui.output_data_frame("column_details"),
            ),
            ui.nav_panel(
                "Data Visualization",
                ui.h4("Numerical Column Distributions"),
                ui.output_ui("numerical_plots"),
            ),
            ui.nav_panel(
                "Correlation",
                ui.h4("Correlation"),
                ui.output_ui("correlation_plots"),
            ),
        ),
    ),
)


# Server: Define the application logic
def server(input, output, session):
    @reactive.Calc
    def load_data():
        """Load data once and cache it"""
        if input.file() is None:
            return pl.DataFrame()

        file_infos = input.file()
        con = duckdb.connect(":memory:")

        # Load data with DuckDB and convert to Polars
        df = con.execute(
            f"SELECT * FROM read_csv_auto('{file_infos[0]['datapath']}')"
        ).pl()
        con.close()

        return df

    @output
    @render.text
    def file_info():
        """Display information about the uploaded file"""
        if input.file() is None:
            return "No data available. Please upload a CSV file."

        file_infos = input.file()
        return f"File uploaded: {file_infos[0]['name']} ({file_infos[0]['size']} bytes)"

    @output
    @render.data_frame
    def dataset_overview():
        """Display overall dataset metrics including shape, duplicates, and data type breakdown"""
        df = load_data()

        # Check if DataFrame has data
        if len(df.columns) == 0:
            return pl.DataFrame(
                {"Message": ["No data available. Please upload a CSV file."]}
            )

        # Calculate duplicate rows
        duplicate_count = df.shape[0] - df.unique().shape[0]

        # Calculate data type breakdown by grouping actual dtypes
        dtype_counts = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        # Build metrics and counts lists
        metrics = ["Number of Rows", "Number of Columns", "Duplicate Rows"]
        counts = [df.shape[0], df.shape[1], duplicate_count]

        # Add dtype breakdown
        for dtype, count in sorted(dtype_counts.items()):
            metrics.append(f"{dtype} Columns")
            counts.append(count)

        shape_df = pl.DataFrame({"Metric": metrics, "Count": counts})

        return shape_df

    @output
    @render.data_frame
    def column_details():
        """Display detailed column statistics including datatype, counts, missing values, mixed type detection, and example values"""
        df = load_data()

        # Check if DataFrame has data
        if len(df.columns) == 0:
            return pl.DataFrame(
                {"Message": ["No data available. Please upload a CSV file."]}
            )

        total_rows = df.shape[0]
        column_stats = []

        for col in df.columns:
            stats = _get_column_stats(df[col], col, total_rows)
            column_stats.append(stats)

        # Create DataFrame with column information
        return pl.DataFrame(column_stats)

    def _get_column_stats(series: pl.Series, col_name: str, total_rows: int) -> dict:
        """Calculate statistics for a single column"""
        null_count = series.null_count()
        dtype = str(series.dtype)

        # Determine if column is numerical or categorical
        col_type = "Numerical" if series.dtype in pl.NUMERIC_DTYPES else "Categorical"

        # Calculate missing percentage
        missing_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0

        # Calculate cardinality (distinct values as percentage of non-null count)
        non_null_count = series.count()
        distinct_count = series.n_unique()
        cardinality_pct = (
            round((distinct_count / non_null_count * 100), 2)
            if non_null_count > 0
            else 0
        )

        # Check for mixed types (only relevant for string columns)
        mixed_type = _detect_mixed_types(series, dtype)

        # Get example value (first non-null value)
        example_value = _get_example_value(series)

        # Calculate numerical statistics for numerical columns
        if series.dtype in pl.NUMERIC_DTYPES:
            min_val = series.min()
            max_val = series.max()
            mean_val = round(series.mean(), 3)
            var_val = round(series.var(), 3)
            std_val = round(series.std(), 3)
            q25 = series.quantile(0.25)
            q50 = series.quantile(0.50)
            q75 = series.quantile(0.75)
            # Imbalance check not applicable for numerical columns
            is_imbalanced = ""
        else:
            # For categorical columns, get min/max by sorting
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                sorted_vals = non_null.sort()
                min_val = str(sorted_vals[0])
                max_val = str(sorted_vals[-1])
            else:
                min_val = max_val = None
            mean_val = var_val = std_val = None
            q25 = q50 = q75 = None

            # Check for imbalance using majority-class dominance (90% threshold)
            if non_null_count > 0:
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    max_count = value_counts["count"].max()
                    majority_pct = (max_count / non_null_count) * 100
                    is_imbalanced = "Yes" if majority_pct >= 90 else "No"
                else:
                    is_imbalanced = "No"
            else:
                is_imbalanced = ""

        return {
            "Column": col_name,
            "Data Type": dtype,
            "Type": col_type,
            "Mixed Types": mixed_type,
            "Count": non_null_count,
            "Distinct": distinct_count,
            "Cardinality %": cardinality_pct,
            "Imbalanced": is_imbalanced,
            "Missing": null_count,
            "Missing %": missing_pct,
            "Mean": mean_val,
            "Variance": var_val,
            "Std Deviation": std_val,
            "Min": min_val,
            "Q25": q25,
            "Q50": q50,
            "Q75": q75,
            "Max": max_val,
            "Example Value": example_value,
        }

    def _detect_mixed_types(series: pl.Series, dtype: str) -> str:
        """Detect if a string column contains mixed types"""
        if dtype not in ["Utf8", "String"]:
            return "No"

        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return "No"

        try:
            numeric_cast = non_null.cast(pl.Float64, strict=False)
            null_after_cast = numeric_cast.null_count()
            # If some values became null after casting and some didn't, we have mixed types
            if 0 < null_after_cast < len(non_null):
                return "Yes"
        except Exception:
            pass

        return "No"

    def _get_example_value(series: pl.Series) -> str:
        """Get the first non-null value as an example"""
        non_null = series.drop_nulls()
        if len(non_null) > 0:
            return str(non_null[0])
        return "(all null)"

    def _create_histogram(data, col_name):
        """Create a histogram for a numerical column using Freedman-Diaconis rule"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins='fd', edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="plain", axis="both")
        plt.tight_layout()
        return fig

    def _create_boxplot(data, col_name):
        """Create a boxplot for a numerical column"""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, vert=True)
        ax.set_ylabel("Value")
        ax.set_title(f"Boxplot")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="plain", axis="y")
        plt.tight_layout()
        return fig

    def _create_correlation_heatmap(df, method='pearson', title=''):
        """Create a correlation heatmap for numerical columns"""
        # Filter numerical columns
        numerical_cols = [
            col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES
        ]

        if len(numerical_cols) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No numerical columns available for correlation",
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig

        # Convert to numpy for correlation calculation
        df_numeric = df.select(numerical_cols)

        # Calculate correlation matrix using Polars
        # Convert to pandas for easier correlation calculation
        corr_data = {}
        for col in numerical_cols:
            corr_data[col] = df_numeric[col].drop_nulls().to_numpy()

        # Create numpy array (fill missing values with column mean for correlation)
        data_arrays = []
        for col in numerical_cols:
            col_data = df_numeric[col].to_numpy()
            # Replace None/NaN with column mean
            col_mean = np.nanmean(col_data)
            col_data = np.where(np.isnan(col_data.astype(float)), col_mean, col_data)
            data_arrays.append(col_data)

        data_matrix = np.column_stack(data_arrays)

        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = np.corrcoef(data_matrix, rowvar=False)
        else:  # spearman
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data_matrix, axis=0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(20, 14))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            xticklabels=numerical_cols,
            yticklabels=numerical_cols,
            ax=ax,
            vmin=-1,
            vmax=1
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def numerical_plots():
        """Generate histograms and boxplots for all numerical columns"""
        df = load_data()

        # Check if DataFrame has data
        if len(df.columns) == 0:
            return ui.p("No data available. Please upload a CSV file.")

        # Get numerical columns
        numerical_cols = [
            col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES
        ]

        if len(numerical_cols) == 0:
            return ui.p("No numerical columns found in the dataset.")

        # Generate plots for each numerical column
        plot_elements = []

        for col in numerical_cols:
            # Get non-null values for plotting
            data = df[col].drop_nulls().to_numpy()

            if len(data) == 0:
                continue

            # Add section header for the column
            plot_elements.append(ui.h5(f"Column: {col}"))

            # Create unique IDs for the plots
            hist_id = f"hist_{col.replace(' ', '_').replace('.', '_')}"
            box_id = f"box_{col.replace(' ', '_').replace('.', '_')}"

            # Add plots side-by-side using row and column layout
            plot_elements.append(
                ui.div(
                    ui.row(
                        ui.column(6, ui.output_plot(hist_id)),
                        ui.column(6, ui.output_plot(box_id)),
                    ),
                    style="text-align: left;"
                )
            )
            plot_elements.append(ui.hr())

            # Register the plot renderers dynamically
            def make_hist_renderer(col_name, col_data):
                @output(id=f"hist_{col_name.replace(' ', '_').replace('.', '_')}")
                @render.plot
                def _():
                    return _create_histogram(col_data, col_name)

            def make_box_renderer(col_name, col_data):
                @output(id=f"box_{col_name.replace(' ', '_').replace('.', '_')}")
                @render.plot
                def _():
                    return _create_boxplot(col_data, col_name)

            make_hist_renderer(col, data)
            make_box_renderer(col, data)

        return ui.div(*plot_elements)

    @output
    @render.ui
    def correlation_plots():
        """Generate correlation heatmaps for Pearson and Spearman methods"""
        df = load_data()

        # Check if DataFrame has data
        if len(df.columns) == 0:
            return ui.p("No data available. Please upload a CSV file.")

        # Get numerical columns
        numerical_cols = [
            col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES
        ]

        if len(numerical_cols) == 0:
            return ui.p("No numerical columns found in the dataset.")

        if len(numerical_cols) < 2:
            return ui.p("At least 2 numerical columns are required for correlation analysis.")

        # Create plot elements
        plot_elements = []

        # Add Pearson correlation heatmap
        plot_elements.append(ui.h5("Pearson Correlation"))
        plot_elements.append(
            ui.div(
                ui.output_plot("pearson_heatmap", width="100%", height="800px"),
                style="text-align: left;"
            )
        )
        plot_elements.append(ui.hr())

        # Add Spearman correlation heatmap
        plot_elements.append(ui.h5("Spearman Correlation"))
        plot_elements.append(
            ui.div(
                ui.output_plot("spearman_heatmap", width="100%", height="800px"),
                style="text-align: left;"
            )
        )

        # Register the plot renderers
        @output(id="pearson_heatmap")
        @render.plot
        def _():
            return _create_correlation_heatmap(df, method='pearson', title='Pearson Correlation')

        @output(id="spearman_heatmap")
        @render.plot
        def _():
            return _create_correlation_heatmap(df, method='spearman', title='Spearman Correlation')

        return ui.div(*plot_elements)


# Create the Shiny app
app = App(app_ui, server)
