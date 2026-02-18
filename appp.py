import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import openpyxl
import requests
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================
# GEMINI API CONFIGURATION - USING SECRETS
# ============================================

# Get API key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Using gemini-2.5-flash which is currently available
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# ============================================
# Document Processor
# ============================================

class DocumentProcessor:
    def __init__(self):
        self.df = None
        self.document_text = ""
        self.file_name = ""
        self.file_type = ""
        self.data_summary = {}
        
    def process_file(self, uploaded_file):
        """Process uploaded file (Excel, CSV, PDF)"""
        self.file_name = uploaded_file.name
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension in ['xlsx', 'xls']:
                self.df = pd.read_excel(uploaded_file)
                self.file_type = 'excel'
                self._generate_summary()
                return True, f"✅ Excel loaded: {len(self.df)} rows, {len(self.df.columns)} columns"
            
            elif file_extension == 'csv':
                self.df = pd.read_csv(uploaded_file)
                self.file_type = 'csv'
                self._generate_summary()
                return True, f"✅ CSV loaded: {len(self.df)} rows, {len(self.df.columns)} columns"
            
            elif file_extension == 'pdf':
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                self.document_text = text
                self.file_type = 'pdf'
                return True, f"✅ PDF loaded: {len(pdf_reader.pages)} pages, {len(text)} characters"
            
            else:
                return False, "❌ Unsupported file type. Please upload Excel, CSV, or PDF."
                
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def _generate_summary(self):
        """Generate summary of dataframe"""
        if self.df is None:
            return
        
        self.data_summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
    
    def get_data_context(self):
        """Get context for Gemini API"""
        if self.file_type in ['excel', 'csv'] and self.df is not None:
            context = f"""FILE: {self.file_name}
TYPE: {self.file_type.upper()}
ROWS: {len(self.df)}
COLUMNS: {len(self.df.columns)}

COLUMN NAMES AND DATA TYPES:
"""
            for col in self.df.columns:
                dtype = "NUMERIC" if pd.api.types.is_numeric_dtype(self.df[col]) else "TEXT"
                context += f"- {col}: {dtype}\n"
            
            # Sample data
            context += f"\nFIRST 5 ROWS:\n"
            context += self.df.head().to_string()
            
            # Statistics for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                context += f"\n\nNUMERIC COLUMNS STATISTICS:\n"
                for col in numeric_cols:
                    context += f"\n{col}:\n"
                    context += f"  Min: {self.df[col].min():.2f}\n"
                    context += f"  Max: {self.df[col].max():.2f}\n"
                    context += f"  Mean: {self.df[col].mean():.2f}\n"
                    context += f"  Sum: {self.df[col].sum():.2f}\n"
            
            return context
        
        elif self.file_type == 'pdf':
            return f"""FILE: {self.file_name}
TYPE: PDF
CONTENT:
{self.document_text[:3000]}"""
        
        return "No data loaded"
    
    def calculate_direct(self, column, operation):
        """Direct calculation without AI"""
        if self.df is None or column not in self.df.columns:
            return None
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            if operation == "Count":
                return len(self.df[column].dropna())
            return None
        
        if operation == "Sum":
            return self.df[column].sum()
        elif operation == "Average":
            return self.df[column].mean()
        elif operation == "Min":
            return self.df[column].min()
        elif operation == "Max":
            return self.df[column].max()
        elif operation == "Count":
            return self.df[column].count()
        return None
    
    def execute_calculation(self, calculation_code):
        """Execute calculation code safely"""
        try:
            # Create safe namespace
            namespace = {
                'df': self.df,
                'pd': pd,
                'np': np,
                'sum': sum,
                'len': len,
                'min': min,
                'max': max,
                'mean': lambda x: sum(x)/len(x) if len(x) > 0 else 0
            }
            
            result = eval(calculation_code, {"__builtins__": {}}, namespace)
            return True, result
        except Exception as e:
            return False, str(e)

# ============================================
# Gemini AI Interface
# ============================================

class GeminiInterface:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def query(self, prompt, context=""):
        """Send query to Gemini AI"""
        
        full_prompt = f"""You are an expert data analyst. Analyze this data and answer the user's question.

DATA CONTEXT:
{context[:3000]}

USER QUESTION: {prompt}

INSTRUCTIONS:
1. If it's a calculation, provide the answer directly
2. If you need to show code, put it in ```python``` blocks
3. Be clear and concise

RESPONSE:"""

        try:
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "No response from Gemini"
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================
# Streamlit App
# ============================================

st.set_page_config(
    page_title="Gemini Document Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if 'gemini' not in st.session_state:
    st.session_state.gemini = GeminiInterface(GEMINI_API_KEY)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🤖 Gemini Document Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Upload any file - Ask anything - Get instant answers")

# Sidebar
with st.sidebar:
    st.markdown("## 🚀 Gemini AI")
    st.success("✅ Connected to Gemini 2.5 Flash")
    
    st.markdown("---")
    
    # File upload
    st.markdown("## 📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose Excel, CSV, or PDF",
        type=['xlsx', 'xls', 'csv', 'pdf']
    )
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            success, message = st.session_state.processor.process_file(uploaded_file)
            if success:
                st.success(message)
                
                # File stats
                st.markdown("### 📊 Statistics")
                if st.session_state.processor.file_type in ['excel', 'csv']:
                    st.metric("Rows", st.session_state.processor.data_summary['rows'])
                    st.metric("Columns", st.session_state.processor.data_summary['columns'])
                    
                    # Column list
                    with st.expander("📋 Columns"):
                        for col in st.session_state.processor.df.columns:
                            dtype = "🔢" if pd.api.types.is_numeric_dtype(st.session_state.processor.df[col]) else "📝"
                            st.write(f"{dtype} {col}")
            else:
                st.error(message)

# Main content
if st.session_state.processor.df is not None:
    # Data preview
    with st.expander("👀 Preview Data", expanded=True):
        st.dataframe(
            st.session_state.processor.df.head(10),
            use_container_width=True
        )
    
    # Quick calculations section
    st.markdown("## 🔢 Quick Calculations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_column = st.selectbox(
            "Select column",
            st.session_state.processor.df.columns
        )
    
    with col2:
        operation = st.selectbox(
            "Operation",
            ["Sum", "Average", "Min", "Max", "Count"]
        )
    
    with col3:
        if st.button("Calculate", use_container_width=True):
            result = st.session_state.processor.calculate_direct(selected_column, operation)
            if result is not None:
                st.success(f"**{operation} of {selected_column}:** {result:,.2f}")
            else:
                if operation == "Count":
                    count = len(st.session_state.processor.df[selected_column].dropna())
                    st.info(f"**Count of {selected_column}:** {count}")
                else:
                    st.warning(f"Cannot calculate {operation} for non-numeric column")

# Question input
st.markdown("---")
st.markdown("## 💭 Ask Gemini")

question = st.text_input(
    "What would you like to know?",
    placeholder="e.g., What is the sum of is_international column? or Summarize this data"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)

if ask_button and question:
    # Check if data is loaded
    has_data = (st.session_state.processor.df is not None) or (len(st.session_state.processor.document_text) > 0)
    
    if not has_data:
        st.warning("⚠️ Please upload a file first")
    else:
        with st.spinner("🤔 Gemini is thinking..."):
            # Get context
            context = st.session_state.processor.get_data_context()
            
            # Query Gemini
            response = st.session_state.gemini.query(question, context)
            
            # Display response
            st.markdown("### 📝 Answer")
            st.markdown(f"<div class='result-box'>{response}</div>", unsafe_allow_html=True)
            
            # Extract and execute Python code if present
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, response, re.DOTALL)
            
            if code_matches and st.session_state.processor.df is not None:
                st.markdown("### 🔢 Calculation Results")
                
                for i, code in enumerate(code_matches):
                    with st.expander(f"View Calculation"):
                        st.code(code, language='python')
                        
                        # Execute the code
                        success, result = st.session_state.processor.execute_calculation(code)
                        
                        if success:
                            st.success(f"**Result:** {result}")
                        else:
                            st.error(f"Error: {result}")
            
            # Save to history
            st.session_state.chat_history.append({
                'question': question,
                'response': response[:100] + "...",
                'time': datetime.now().strftime("%H:%M")
            })

# Chat history
if st.session_state.chat_history:
    with st.expander("📜 Recent Questions"):
        for chat in st.session_state.chat_history[-5:]:
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"*{chat['time']}*")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚀 Powered by Google Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)