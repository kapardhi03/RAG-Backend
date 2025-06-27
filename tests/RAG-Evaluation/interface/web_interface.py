# tests/RAG-Evaluation/interface/web_interface.py
"""
Simple web interface for RAG evaluation using Streamlit
"""
import streamlit as st
import asyncio
import pandas as pd
import json
import os
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# Streamlit doesn't handle async well, so we need this wrapper
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.base_evaluator import EvaluationDatapoint
from runner.evaluation_runner import RAGEvaluationRunner
from runner.batch_evaluator import BatchEvaluator
from llama_index.core.llms import OpenAI
from llama_index.core.embeddings import OpenAIEmbedding

class RAGEvaluationWebApp:
    """Streamlit web application for RAG evaluation"""
    
    def __init__(self):
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="RAG Evaluation Suite",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🎯 RAG Evaluation Suite")
        st.markdown("Comprehensive evaluation toolkit for Retrieval-Augmented Generation systems")
    
    @st.cache_resource
    def setup_runner(_self, api_key: str, llm_model: str, embedding_model: str):
        """Setup evaluation runner (cached)"""
        try:
            llm = OpenAI(model=llm_model, api_key=api_key, temperature=0.0)
            embedding = OpenAIEmbedding(model=embedding_model, api_key=api_key)
            
            runner = RAGEvaluationRunner(llm=llm, embedding_model=embedding)
            return runner, None
        except Exception as e:
            return None, str(e)
    
    def sidebar_config(self):
        """Sidebar configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # API Configuration
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Required for most evaluators"
        )
        
        llm_model = st.sidebar.selectbox(
            "LLM Model",
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        embedding_model = st.sidebar.selectbox(
            "Embedding Model",
            ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
            index=0
        )
        
        return api_key, llm_model, embedding_model
    
    def data_input_section(self):
        """Data input section"""
        st.header("📊 Data Input")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload file", "Manual input", "Sample data"]
        )
        
        datapoints = []
        
        if data_source == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload evaluation data",
                type=["json", "csv"],
                help="Upload a JSON or CSV file with evaluation data"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.json'):
                        data = json.load(uploaded_file)
                        datapoints = [
                            EvaluationDatapoint(
                                query=item["query"],
                                response=item.get("response", ""),
                                reference=item.get("reference"),
                                contexts=item.get("contexts"),
                                metadata=item.get("metadata")
                            )
                            for item in data
                        ]
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        for _, row in df.iterrows():
                            contexts = None
                            if 'contexts' in row and pd.notna(row['contexts']):
                                try:
                                    contexts = eval(row['contexts'])
                                except:
                                    contexts = [str(row['contexts'])]
                            
                            datapoints.append(EvaluationDatapoint(
                                query=row["query"],
                                response=row.get("response", ""),
                                reference=row.get("reference") if pd.notna(row.get("reference")) else None,
                                contexts=contexts,
                                metadata={"source": "uploaded_csv"}
                            ))
                    
                    st.success(f"✅ Loaded {len(datapoints)} datapoints")
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        elif data_source == "Manual input":
            st.subheader("Manual Data Entry")
            
            # Initialize session state for manual data
            if 'manual_datapoints' not in st.session_state:
                st.session_state.manual_datapoints = []
            
            with st.form("manual_input_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    query = st.text_area("Query", height=100)
                    response = st.text_area("Response", height=100)
                
                with col2:
                    reference = st.text_area("Reference (optional)", height=100)
                    contexts_text = st.text_area(
                        "Contexts (one per line)", 
                        height=100,
                        help="Enter each context on a separate line"
                    )
                
                submitted = st.form_submit_button("Add Datapoint")
                
                if submitted and query and response:
                    contexts = [ctx.strip() for ctx in contexts_text.split('\n') if ctx.strip()] if contexts_text else None
                    
                    datapoint = EvaluationDatapoint(
                        query=query,
                        response=response,
                        reference=reference if reference else None,
                        contexts=contexts,
                        metadata={"source": "manual_input"}
                    )
                    
                    st.session_state.manual_datapoints.append(datapoint)
                    st.success("✅ Datapoint added!")
                    st.experimental_rerun()
            
            # Show current manual datapoints
            if st.session_state.manual_datapoints:
                st.subheader(f"Current Datapoints ({len(st.session_state.manual_datapoints)})")
                
                for i, dp in enumerate(st.session_state.manual_datapoints):
                    with st.expander(f"Datapoint {i+1}: {dp.query[:50]}..."):
                        st.text(f"Query: {dp.query}")
                        st.text(f"Response: {dp.response}")
                        if dp.reference:
                            st.text(f"Reference: {dp.reference}")
                        if dp.contexts:
                            st.text(f"Contexts: {len(dp.contexts)} items")
                
                if st.button("Clear All Datapoints"):
                    st.session_state.manual_datapoints = []
                    st.experimental_rerun()
                
                datapoints = st.session_state.manual_datapoints
        
        elif data_source == "Sample data":
            sample_data = [
                EvaluationDatapoint(
                    query="What is the capital of France?",
                    response="The capital of France is Paris.",
                    reference="Paris is the capital and most populous city of France.",
                    contexts=[
                        "Paris is the capital and most populous city of France.",
                        "France is a country in Western Europe with Paris as its capital city."
                    ],
                    metadata={"category": "geography"}
                ),
                EvaluationDatapoint(
                    query="How does photosynthesis work?",
                    response="Photosynthesis converts sunlight, CO2, and water into glucose and oxygen using chlorophyll.",
                    reference="Photosynthesis is the process by which plants convert light energy into chemical energy.",
                    contexts=[
                        "Photosynthesis occurs in chloroplasts and involves converting carbon dioxide and water into glucose.",
                        "Plants use chlorophyll to capture light energy for photosynthesis."
                    ],
                    metadata={"category": "biology"}
                )
            ]
            
            st.info("Using sample data for demonstration")
            datapoints = sample_data
        
        return datapoints
    
    def evaluator_selection(self, runner):
        """Evaluator selection section"""
        st.header("🔧 Evaluator Selection")
        
        if not runner:
            st.warning("⚠️ Please configure API settings first")
            return []
        
        available_evaluators = list(runner.evaluators.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Evaluators")
            selected_evaluators = st.multiselect(
                "Select evaluators to run:",
                available_evaluators,
                default=available_evaluators[:3]
            )
        
        with col2:
            st.subheader("Custom Evaluators")
            
            # Guideline evaluator
            add_guideline = st.checkbox("Add Guideline Evaluator")
            if add_guideline:
                guidelines = st.text_area(
                    "Guidelines",
                    placeholder="Enter evaluation guidelines...",
                    height=100
                )
                if guidelines:
                    runner.add_guideline_evaluator("custom_guidelines", guidelines)
                    if "custom_guidelines" not in selected_evaluators:
                        selected_evaluators.append("custom_guidelines")
            
            # Retrieval evaluator
            add_retrieval = st.checkbox("Add Retrieval Evaluator")
            if add_retrieval:
                retrieval_metrics = st.multiselect(
                    "Retrieval metrics:",
                    ["hit_rate", "mrr", "ndcg"],
                    default=["hit_rate", "mrr"]
                )
                if retrieval_metrics:
                    runner.add_retrieval_evaluator(retrieval_metrics)
                    if "retrieval" not in selected_evaluators:
                        selected_evaluators.append("retrieval")
        
        return selected_evaluators
    
    async def run_evaluation_async(self, runner, datapoints, selected_evaluators):
        """Run evaluation asynchronously"""
        return await runner.run_evaluation(
            datapoints=datapoints,
            evaluator_names=selected_evaluators,
            save_results=False
        )
    
    def run_evaluation_section(self, runner, datapoints, selected_evaluators):
        """Evaluation execution section"""
        st.header("🚀 Run Evaluation")
        
        if not datapoints:
            st.warning("⚠️ No data available for evaluation")
            return None
        
        if not selected_evaluators:
            st.warning("⚠️ No evaluators selected")
            return None
        
        st.info(f"Ready to evaluate {len(datapoints)} datapoints with {len(selected_evaluators)} evaluators")
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    # Run evaluation
                    results = asyncio.run(
                        self.run_evaluation_async(runner, datapoints, selected_evaluators)
                    )
                    
                    st.success("✅ Evaluation completed!")
                    return results
                
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {e}")
                    return None
        
        return None
    
    def display_results(self, results):
        """Display evaluation results"""
        if not results:
            return
        
        st.header("📈 Results")
        
        # Convert results to DataFrame
        all_data = []
        for evaluator_name, evaluator_results in results.items():
            for result in evaluator_results:
                all_data.append({
                    "Evaluator": result.evaluator_name,
                    "Metric": result.metric_name,
                    "Score": result.score,
                    "Query": result.query[:100] + "..." if len(result.query) > 100 else result.query,
                    "Feedback": result.feedback[:200] + "..." if len(result.feedback) > 200 else result.feedback
                })
        
        if not all_data:
            st.warning("No results to display")
            return
        
        df = pd.DataFrame(all_data)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        summary = df.groupby(['Evaluator', 'Metric'])['Score'].agg(['count', 'mean', 'std']).round(3)
        st.dataframe(summary)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution by Evaluator")
            fig_box = px.box(df, x='Evaluator', y='Score', color='Evaluator')
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.subheader("Average Scores by Metric")
            avg_scores = df.groupby(['Evaluator', 'Metric'])['Score'].mean().reset_index()
            fig_bar = px.bar(avg_scores, x='Metric', y='Score', color='Evaluator', barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        # Filter options
        selected_evaluator = st.selectbox("Filter by evaluator:", ["All"] + list(df['Evaluator'].unique()))
        
        if selected_evaluator != "All":
            filtered_df = df[df['Evaluator'] == selected_evaluator]
        else:
            filtered_df = df
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download results
        st.subheader("💾 Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main application runner"""
        # Sidebar configuration
        api_key, llm_model, embedding_model = self.sidebar_config()
        
        # Setup runner
        runner = None
        if api_key:
            runner, error = self.setup_runner(api_key, llm_model, embedding_model)
            if error:
                st.sidebar.error(f"❌ Setup failed: {error}")
            else:
                st.sidebar.success("✅ Runner configured")
        else:
            st.sidebar.warning("⚠️ API key required")
        
        # Main content
        datapoints = self.data_input_section()
        selected_evaluators = self.evaluator_selection(runner)
        results = self.run_evaluation_section(runner, datapoints, selected_evaluators)
        
        if results:
            self.display_results(results)

def run_web_app():
    """Entry point for web application"""
    app = RAGEvaluationWebApp()
    app.run()

if __name__ == "__main__":
    run_web_app()