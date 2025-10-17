import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from agents import create_agent, create_multi_agent
from tools import DataRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.agents import Tool
from matplotlib.figure import Figure
import io
from ydata_profiling import ProfileReport

st.set_page_config(page_title="AI Data Science Assistant", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 AI-Powered Data Science Assistant")
st.caption("Multi-Agent System: Mistral (Reasoning) + CodeLLaMA (Code Gen) | LangChain + Advanced Analytics")

# -----------------------------
# Session State Initialization
# -----------------------------
# Agent version - increment this to force agent reload
AGENT_VERSION = "4.0"  # Multi-agent version

# Initialize agent system
agent_mode = st.sidebar.radio(
    "🤖 Agent Mode",
    ["Multi-Agent (Recommended)", "Single Agent (Mistral)"],
    help="Multi-Agent uses Mistral for reasoning and CodeLLaMA for code generation"
)

if "agent_version" not in st.session_state or st.session_state.agent_version != AGENT_VERSION:
    if agent_mode == "Multi-Agent (Recommended)":
        st.session_state.supervisor = create_multi_agent()
        st.session_state.ds_tools = st.session_state.supervisor.reasoning_agent.ds_tools
    else:
        st.session_state.agent, st.session_state.ds_tools = create_agent()
    st.session_state.agent_version = AGENT_VERSION
    st.session_state.agent_mode = agent_mode
    
if "agent" not in st.session_state and agent_mode == "Single Agent (Mistral)":
    st.session_state.agent, st.session_state.ds_tools = create_agent()
    
if "supervisor" not in st.session_state and agent_mode == "Multi-Agent (Recommended)":
    st.session_state.supervisor = create_multi_agent()
    st.session_state.ds_tools = st.session_state.supervisor.reasoning_agent.ds_tools
if "retriever_engine" not in st.session_state:
    st.session_state.retriever_engine = DataRetriever()
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "retriever_tool" not in st.session_state:
    st.session_state.retriever_tool = None
if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None

# -----------------------------
# Sidebar: Upload & Settings
# -----------------------------
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        if st.session_state.df is None:
            with open("uploaded.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.ds_tools.load_data("uploaded.csv")
            st.session_state.df = st.session_state.ds_tools.get_dataframe()
            st.session_state.df_original = st.session_state.df.copy()
            st.success("✅ Dataset loaded!")
    
    if st.session_state.df is not None:
        st.divider()
        st.info(f"📊 **Dataset**: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")

# -----------------------------
# Main Dashboard
# -----------------------------
if st.session_state.df is not None:
    # Main Dashboard Stats (Above Tabs)
    st.markdown("---")
    dash_col1, dash_col2, dash_col3, dash_col4, dash_col5 = st.columns(5)
    
    with dash_col1:
        st.metric("📊 Rows", f"{st.session_state.df.shape[0]:,}")
    
    with dash_col2:
        st.metric("📋 Columns", st.session_state.df.shape[1])
    
    with dash_col3:
        missing_count = int(st.session_state.df.isnull().sum().sum())
        st.metric("🔴 Missing", missing_count)
    
    with dash_col4:
        duplicate_count = int(st.session_state.df.duplicated().sum())
        st.metric("🔄 Duplicates", duplicate_count)
    
    with dash_col5:
        memory_mb = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
    
    st.markdown("---")
    
    # Quick Action Buttons
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🔄 Reset to Original", key="dashboard_reset", use_container_width=True):
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.ds_tools.df = st.session_state.df.copy()
            if "cleaner" in st.session_state:
                st.session_state.cleaner = None
            st.success("✅ Reset complete!")
            st.rerun()
    
    with action_col2:
        if st.button("🧹 Quick Clean (Standard)", key="dashboard_quick_clean", use_container_width=True):
            from tools import DataCleaner
            cleaner = DataCleaner(st.session_state.df)
            st.session_state.df = cleaner.auto_clean(level='standard')
            st.session_state.ds_tools.df = st.session_state.df.copy()
            st.success("✅ Data cleaned!")
            st.rerun()
    
    with action_col3:
        if st.button("📥 Download CSV", key="dashboard_download", use_container_width=True):
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Save File",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
    
    with action_col4:
        if st.button("📊 Generate EDA Report", key="dashboard_eda", use_container_width=True):
            from ydata_profiling import ProfileReport
            with st.spinner("Generating report..."):
                profile = ProfileReport(st.session_state.df, title="Data Report", minimal=True)
                st.download_button(
                    label="📥 Download Report",
                    data=profile.to_html(),
                    file_name="eda_report.html",
                    mime="text/html"
                )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🧹 Data Cleaning", "📈 Visualizations", "🤖 ML Models", "💬 AI Assistant", "📥 Export"])
    
    # ===== TAB 1: OVERVIEW =====
    with tab1:
        # Show comparison if data has been modified
        is_modified = not st.session_state.df.equals(st.session_state.df_original)
        
        if is_modified:
            st.success("✅ **Data has been modified** - Showing current state vs original")
            compare_col1, compare_col2 = st.columns(2)
            
            with compare_col1:
                st.subheader("📊 Current Data")
                st.metric("Rows", f"{st.session_state.df.shape[0]:,}")
                st.metric("Columns", st.session_state.df.shape[1])
                st.metric("Missing Values", int(st.session_state.df.isnull().sum().sum()))
                st.metric("Duplicates", int(st.session_state.df.duplicated().sum()))
            
            with compare_col2:
                st.subheader("📂 Original Data")
                st.metric("Rows", f"{st.session_state.df_original.shape[0]:,}")
                st.metric("Columns", st.session_state.df_original.shape[1])
                st.metric("Missing Values", int(st.session_state.df_original.isnull().sum().sum()))
                st.metric("Duplicates", int(st.session_state.df_original.duplicated().sum()))
            
            st.markdown("---")
        
        st.header("Dataset Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Info")
            st.write(f"**Shape:** {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns")
            st.write(f"**Memory:** {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.subheader("🔢 Data Types")
            dtype_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes.values.astype(str),
                'Non-Null': st.session_state.df.count().values,
                'Null': st.session_state.df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistics Dashboard")
            
            numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.metric("Numeric Columns", len(numeric_cols))
                st.metric("Categorical Columns", len(st.session_state.df.columns) - len(numeric_cols))
                st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
                st.metric("Duplicate Rows", st.session_state.df.duplicated().sum())
            
            # Missing value chart
            if st.session_state.df.isnull().sum().sum() > 0:
                st.subheader("🔴 Missing Values by Column")
                missing_df = st.session_state.df.isnull().sum()[st.session_state.df.isnull().sum() > 0]
                fig = px.bar(x=missing_df.index, y=missing_df.values, labels={'x': 'Column', 'y': 'Missing Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("👀 Data Preview")
        st.dataframe(st.session_state.df.head(100), use_container_width=True)
        
        st.subheader("📈 Numeric Summary Statistics")
        if len(numeric_cols) > 0:
            st.dataframe(st.session_state.df[numeric_cols].describe(), use_container_width=True)
        
        # Automated EDA Report
        st.subheader("📋 Generate Full EDA Report")
        if st.button("🚀 Generate Comprehensive Report (may take time)", key="eda_report"):
            with st.spinner("Generating detailed report..."):
                try:
                    profile = ProfileReport(st.session_state.df, title="Data Profiling Report", explorative=True, minimal=True)
                    st.success("✅ Report generated!")
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=profile.to_html(),
                        file_name="data_report.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    # ===== TAB 2: AUTOMATED DATA CLEANING =====
    with tab2:
        st.header("🤖 Automated Data Cleaning")
        st.info("💡 **One-click data cleaning** with intelligent strategies. No manual configuration needed!")
        
        from tools import DataCleaner
        
        # Initialize cleaner
        if "cleaner" not in st.session_state or st.session_state.get("cleaner_df_id") != id(st.session_state.df):
            st.session_state.cleaner = DataCleaner(st.session_state.df)
            st.session_state.cleaner_df_id = id(st.session_state.df)
        
        # Main content
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Cleaning Strategy")
            
            # Describe strategies
            st.markdown("""
            **Choose your cleaning level:**
            
            🌱 **Light** - Quick & Safe
            - Remove duplicates
            - Fill missing values
            - Best for: Exploration
            
            ⭐ **Standard** - Recommended
            - Light cleaning +
            - Handle outliers
            - Encode categories
            - Optimize memory
            - Best for: Most use cases
            
            🔥 **Aggressive** - Maximum
            - Standard cleaning +
            - Remove useless columns
            - Deep optimization
            - Best for: ML pipelines
            """)
            
            st.divider()
            
            # Select cleaning level
            auto_level = st.radio(
                "Select Cleaning Level:",
                ["light", "standard", "aggressive"],
                index=1,  # Default to standard
                help="Standard is recommended for most use cases"
            )
            
            st.divider()
            
            # Big clean button
            if st.button("🚀 Clean Data Now", key="cleaning_auto_clean", type="primary", use_container_width=True):
                with st.spinner(f"🧹 Cleaning data with {auto_level} strategy..."):
                    st.session_state.cleaner.auto_clean(level=auto_level)
                    st.session_state.df = st.session_state.cleaner.get_cleaned_dataframe()
                    st.session_state.ds_tools.df = st.session_state.df.copy()
                    st.success(f"✅ Data cleaned successfully with {auto_level} level!")
                    st.balloons()
                    st.rerun()
            
            st.divider()
            
            # Reset button
            if st.button("🔄 Reset to Original", key="cleaning_tab_reset", use_container_width=True):
                st.session_state.cleaner.reset()
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.ds_tools.df = st.session_state.df.copy()
                st.session_state.cleaner = DataCleaner(st.session_state.df)
                st.success("✅ Reset complete")
                st.rerun()
        
        with col2:
            # Before/After Comparison
            st.subheader("📊 Data Status")
            
            # Metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                original_rows = st.session_state.df_original.shape[0]
                current_rows = st.session_state.df.shape[0]
                delta_rows = int(current_rows - original_rows)
                st.metric("Rows", current_rows, delta=delta_rows if delta_rows != 0 else None)
            
            with metrics_col2:
                original_cols = st.session_state.df_original.shape[1]
                current_cols = st.session_state.df.shape[1]
                delta_cols = int(current_cols - original_cols)
                st.metric("Columns", current_cols, delta=delta_cols if delta_cols != 0 else None)
            
            with metrics_col3:
                missing = int(st.session_state.df.isnull().sum().sum())
                original_missing = int(st.session_state.df_original.isnull().sum().sum())
                delta_missing = int(missing - original_missing)
                st.metric("Missing", missing, delta=delta_missing if delta_missing != 0 else None, delta_color="inverse")
            
            with metrics_col4:
                memory_mb = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
                original_memory = st.session_state.df_original.memory_usage(deep=True).sum() / 1024**2
                delta_memory = memory_mb - original_memory
                st.metric("Memory (MB)", f"{memory_mb:.1f}", delta=f"{delta_memory:.1f}" if delta_memory != 0 else None, delta_color="inverse")
            
            st.divider()
            
            # Cleaning Report
            st.subheader("📋 Cleaning Log")
            report = st.session_state.cleaner.get_cleaning_report()
            
            if "No cleaning operations performed" in report:
                st.info("👉 Click 'Clean Data Now' to start automated cleaning")
            else:
                st.text_area("", report, height=350, label_visibility="collapsed")
            
            st.divider()
            
            # Data Preview
            st.subheader("👀 Data Preview")
            
            # Tabs for before/after
            preview_tab1, preview_tab2 = st.tabs(["After Cleaning", "Original Data"])
            
            with preview_tab1:
                st.dataframe(st.session_state.df.head(15), use_container_width=True)
            
            with preview_tab2:
                st.dataframe(st.session_state.df_original.head(15), use_container_width=True)
    
    # ===== TAB 3: VISUALIZATIONS =====
    with tab3:
        st.header("Data Visualizations")
        
        viz_col1, viz_col2 = st.columns([1, 2])
        
        with viz_col1:
            st.subheader("📊 Plot Configuration")
            
            plot_type = st.selectbox("Select Plot Type", [
                "Scatter Plot",
                "Line Plot",
                "Bar Plot",
                "Histogram",
                "Box Plot",
                "Violin Plot",
                "Heatmap (Correlation)",
                "Pair Plot",
                "3D Scatter",
                "Distribution Plot"
            ])
            
            numeric_cols = list(st.session_state.df.select_dtypes(include=['number']).columns)
            all_cols = list(st.session_state.df.columns)
            
            if plot_type in ["Scatter Plot", "Line Plot", "3D Scatter"]:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                if plot_type == "3D Scatter":
                    z_col = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
                color_col = st.selectbox("Color by (optional)", [None] + all_cols)
            
            elif plot_type in ["Histogram", "Box Plot", "Violin Plot", "Distribution Plot"]:
                selected_col = st.selectbox("Select Column", numeric_cols)
                if plot_type in ["Box Plot", "Violin Plot"]:
                    group_by = st.selectbox("Group by (optional)", [None] + all_cols)
            
            elif plot_type == "Bar Plot":
                x_col = st.selectbox("Category Column", all_cols)
                y_col = st.selectbox("Value Column (optional)", [None] + numeric_cols)
            
            elif plot_type == "Pair Plot":
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
            
            elif plot_type == "Heatmap (Correlation)":
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols)
            
            generate_plot = st.button("🎨 Generate Plot", key="viz_generate_plot", use_container_width=True)
        
        with viz_col2:
            if generate_plot:
                try:
                    if plot_type == "Scatter Plot":
                        fig = px.scatter(st.session_state.df, x=x_col, y=y_col, color=color_col, 
                                        title=f"{x_col} vs {y_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Line Plot":
                        fig = px.line(st.session_state.df, x=x_col, y=y_col, color=color_col,
                                     title=f"{x_col} vs {y_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Histogram":
                        fig = px.histogram(st.session_state.df, x=selected_col, 
                                          title=f"Distribution of {selected_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Box Plot":
                        if group_by and group_by != "None":
                            fig = px.box(st.session_state.df, x=group_by, y=selected_col,
                                        title=f"{selected_col} by {group_by}", height=500)
                        else:
                            fig = px.box(st.session_state.df, y=selected_col,
                                        title=f"Box Plot: {selected_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Violin Plot":
                        if group_by and group_by != "None":
                            fig = px.violin(st.session_state.df, x=group_by, y=selected_col,
                                           title=f"{selected_col} by {group_by}", height=500)
                        else:
                            fig = px.violin(st.session_state.df, y=selected_col,
                                           title=f"Violin Plot: {selected_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Bar Plot":
                        if y_col and y_col != "None":
                            fig = px.bar(st.session_state.df, x=x_col, y=y_col,
                                        title=f"{y_col} by {x_col}", height=500)
                        else:
                            value_counts = st.session_state.df[x_col].value_counts()
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                        labels={'x': x_col, 'y': 'Count'},
                                        title=f"Frequency of {x_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Heatmap (Correlation)":
                        if selected_cols:
                            corr = st.session_state.df[selected_cols].corr()
                            fig = px.imshow(corr, text_auto=True, aspect="auto",
                                           title="Correlation Heatmap", height=600)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Pair Plot":
                        if selected_cols:
                            fig = px.scatter_matrix(st.session_state.df[selected_cols],
                                                   title="Pair Plot", height=800)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "3D Scatter":
                        fig = px.scatter_3d(st.session_state.df, x=x_col, y=y_col, z=z_col,
                                           color=color_col, title=f"3D Scatter: {x_col}, {y_col}, {z_col}",
                                           height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif plot_type == "Distribution Plot":
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=st.session_state.df[selected_col], name="Histogram"))
                        fig.update_layout(title=f"Distribution of {selected_col}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
            else:
                st.info("👈 Configure plot settings and click 'Generate Plot'")
    
    # ===== TAB 4: ML MODELS =====
    with tab4:
        st.header("Machine Learning Models")
        
        ml_col1, ml_col2 = st.columns([1, 2])
        
        with ml_col1:
            st.subheader("🎯 Model Configuration")
            
            target_column = st.selectbox(
                "Select Target Column",
                st.session_state.df.columns,
                index=len(st.session_state.df.columns) - 1
            )
            
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            
            model_types = st.multiselect(
                "Select Models to Compare",
                ["Random Forest", "XGBoost", "Logistic Regression", "Linear Regression"],
                default=["Random Forest"]
            )
            
            run_model = st.button("🚀 Train Models", key="ml_train_models", use_container_width=True)
        
        with ml_col2:
            if run_model:
                st.subheader("📊 Model Results")
                
                with st.spinner("Training models..."):
                    try:
                        results = st.session_state.ds_tools.train_models(
                            st.session_state.df,
                            target_column,
                            model_types,
                            test_size
                        )
                        
                        for model_name, result in results.items():
                            with st.expander(f"📈 {model_name}", expanded=True):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Score", f"{result['score']:.4f}")
                                    st.metric("Task Type", result['task_type'].title())
                                with col_b:
                                    if 'training_time' in result:
                                        st.metric("Training Time", f"{result['training_time']:.2f}s")
                                
                                if 'plot' in result:
                                    st.pyplot(result['plot'])
                    
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
            
            # Quick Model Suggestion
            st.subheader("🤖 Quick Model Suggestion")
            if st.button("Get AI Suggestion", key="ml_ai_suggestion"):
                model_output = st.session_state.ds_tools.suggest_model(target_column)
                st.info(model_output)
    
    # ===== TAB 5: AI ASSISTANT =====
    with tab5:
        st.header("AI Assistant Chat")
        
        # Show current mode
        current_mode = st.session_state.get('agent_mode', 'Multi-Agent (Recommended)')
        if current_mode == "Multi-Agent (Recommended)":
            st.success("🤖 **Multi-Agent Mode**: Mistral (reasoning) + CodeLLaMA (code generation)")
        else:
            st.info("🤖 **Single Agent Mode**: Mistral only")
        
        st.markdown("""
        💡 **Examples:**
        - *"Summarize the data"* → Mistral will analyze
        - *"Show correlations"* → Mistral will compute
        - *"Create scatter plot of price vs duration"* → CodeLLaMA will generate code
        - *"Plot histogram of age"* → CodeLLaMA will generate code
        """)
        
        query = st.text_input(
            "Your question:",
            placeholder="e.g., 'Create a scatter plot' or 'Explain correlations'"
        )
        
        if st.button("🤖 Ask Assistant", key="ai_assistant_ask", use_container_width=True):
            if query:
                try:
                    # Update DataFrame in agents
                    if current_mode == "Multi-Agent (Recommended)":
                        st.session_state.supervisor.set_dataframe(st.session_state.df)
                        
                        with st.spinner("🤖 Routing query to appropriate agent..."):
                            result = st.session_state.supervisor.process_query(query)
                        
                        # Show which agent handled the query
                        if result['agent'] == 'CodeLLaMA':
                            st.success(f"✅ Complete! Handled by **CodeLLaMA** (Code Generation)")
                        else:
                            st.success(f"✅ Complete! Handled by **Mistral** (Reasoning)")
                        
                        # Display response with proper formatting
                        if "```python" in result['response']:
                            parts = result['response'].split("**Generated Code:**")
                            if len(parts) > 1:
                                st.write(parts[0])
                                st.code(parts[1].split("```python")[1].split("```")[0], language="python")
                                if "**Output:**" in result['response']:
                                    output = result['response'].split("**Output:**")[1]
                                    st.text(output)
                            else:
                                st.text_area("Response", result['response'], height=400)
                        else:
                            st.text_area("Response", result['response'], height=300)
                    else:
                        # Single agent mode
                        with st.spinner("🤖 Thinking..."):
                            response = st.session_state.agent.run(query)
                        st.success("✅ Complete!")
                        st.text_area("Response", response, height=300)
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Try rephrasing your question or use simpler queries")
            else:
                st.warning("Please enter a query.")
    
    # ===== TAB 6: EXPORT =====
    with tab6:
        st.header("Export Data & Reports")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.subheader("📥 Export Dataset")
            
            # Export cleaned data
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Export as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name="data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with export_col2:
            st.subheader("📊 Export Summary")
            
            # Generate summary report
            summary = f"""
# Data Analysis Summary

## Dataset Information
- Rows: {st.session_state.df.shape[0]}
- Columns: {st.session_state.df.shape[1]}
- Missing Values: {st.session_state.df.isnull().sum().sum()}

## Columns
{', '.join(st.session_state.df.columns)}

## Numeric Summary
{st.session_state.df.describe().to_string()}
            """
            
            st.download_button(
                label="📥 Download Summary Report (TXT)",
                data=summary,
                file_name="summary_report.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👈 Please upload a CSV file to get started!")
    
    # Show demo
    st.subheader("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Overview**
        - Automated statistics
        - Missing value analysis
        - Data type detection
        - Full EDA reports
        """)
    
    with col2:
        st.markdown("""
        **📈 Visualizations**
        - 10+ plot types
        - Interactive Plotly charts
        - Correlation heatmaps
        - 3D scatter plots
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI & ML**
        - Natural language queries
        - Multiple ML models
        - Model comparison
        - Local LLM (CodeLLaMA)
        """)
