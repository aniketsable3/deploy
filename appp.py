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
import io
import os

# ============================================
# Page configuration - MUST BE FIRST STREAMLIT COMMAND
# ============================================
st.set_page_config(
    page_title="AI Document Analyzer - Powered by Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .stApp {
        background-color: #fafafa;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Secure API Key Handling for Deployment
# ============================================

def get_api_key():
    """Get API key from environment variable or secrets"""
    # For Streamlit Cloud deployment, use secrets
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    
    # For local development with .env or environment variables
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        return api_key
    
    # Fallback for demo/testing - In production, always use secrets or env vars
    return None

# ============================================
# GEMINI API CONFIGURATION
# ============================================

GEMINI_MODELS = {
    "gemini-2.5-flash": "Fast & Efficient (Recommended)",
    "gemini-2.5-pro": "More Powerful (Slower)", 
    "gemini-3-flash-preview": "Latest Flash Preview",
    "gemini-3-pro-preview": "Latest Pro Preview",
}

# ============================================
# Document Processor Class
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
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
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
# Initialize Session State
# ============================================

if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if 'gemini' not in st.session_state:
    st.session_state.gemini = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = "gemini-2.5-flash"

# ============================================
# Header
# ============================================
st.markdown("<h1 class='main-header'>🤖 Gemini Document Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Upload any file - Ask anything - Get instant answers")

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    
    # API Key input (secure)
    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey",
        placeholder="AIza..."
    )
    
    if api_key:
        # Initialize Gemini with provided key
        if st.session_state.gemini is None or st.session_state.gemini.api_key != api_key:
            st.session_state.gemini = GeminiInterface(api_key, st.session_state.current_model)
            st.session_state.api_key_configured = True
            st.success("✅ API Key Configured")
    else:
        # Try to get from secrets/environment
        env_api_key = get_api_key()
        if env_api_key:
            st.session_state.gemini = GeminiInterface(env_api_key, st.session_state.current_model)
            st.session_state.api_key_configured = True
            st.success("✅ API Key Loaded from Environment")
        else:
            st.warning("⚠️ Please enter your Gemini API Key")
            st.info("🔑 Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
    st.markdown("---")
    
    # Model selection (only if API is configured)
    if st.session_state.api_key_configured:
        st.markdown("## 🚀 Model Selection")
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(GEMINI_MODELS.keys()),
            format_func=lambda x: GEMINI_MODELS[x],
            index=0
        )
        
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            if st.session_state.gemini:
                st.session_state.gemini.model_name = selected_model
                st.session_state.gemini.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
                st.success(f"✅ Switched to {selected_model}")
    
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
    
    # Deployment info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.7rem;'>
        <p>🚀 Deployed on Streamlit Cloud</p>
        <p>🔒 API key securely stored</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Main Content
# ============================================

# Check if API is configured
if not st.session_state.api_key_configured:
    st.warning("⚠️ Please configure your Gemini API Key in the sidebar to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        ### How to get your API Key:
        1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy and paste it in the sidebar
        """)
    
    # Stop execution here if no API key
    st.stop()

# Show data preview if available
if st.session_state.processor.df is not None:
    # Data preview
    with st.expander("👀 Preview Data", expanded=True):
        st.dataframe(
            st.session_state.processor.df.head(10),
            use_container_width=True,
            height=300
        )
    
    # Quick calculations section
    st.markdown("## 🔢 Quick Calculations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_column = st.selectbox(
            "Select column",
            st.session_state.processor.df.columns,
            key="quick_calc_column"
        )
    
    with col2:
        operation = st.selectbox(
            "Operation",
            ["Sum", "Average", "Min", "Max", "Count"],
            key="quick_calc_operation"
        )
    
    with col3:
        if st.button("Calculate", use_container_width=True, type="secondary", key="quick_calc_button"):
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
    placeholder="e.g., What is the sum of is_international column? or Summarize this data",
    key="question_input"
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    ask_button = st.button("🚀 Ask Gemini", type="primary", use_container_width=True, key="ask_button")

if ask_button and question:
    # Check if data is loaded
    has_data = (st.session_state.processor.df is not None) or (len(st.session_state.processor.document_text) > 0)
    
    if not has_data:
        st.warning("⚠️ Please upload a file first")
    else:
        with st.spinner("🤔 Gemini is analyzing your data..."):
            # Get context
            context = st.session_state.processor.get_data_context()
            
            # Query Gemini
            response = st.session_state.gemini.query(question, context)
            
            # Display response in a nice box
            st.markdown("### 📝 Answer")
            st.markdown(f"<div class='result-box'>{response}</div>", unsafe_allow_html=True)
            
            # Extract and execute Python code if present
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, response, re.DOTALL)
            
            if code_matches and st.session_state.processor.df is not None:
                st.markdown("### 🔢 Calculation Results")
                
                for i, code in enumerate(code_matches):
                    with st.expander(f"View Calculation #{i+1}"):
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
                'response': response[:100] + "..." if len(response) > 100 else response,
                'time': datetime.now().strftime("%H:%M")
            })

# Chat history
if st.session_state.chat_history:
    with st.expander("📜 Recent Questions"):
        for i, chat in enumerate(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"*{chat['time']}*")
            if i < len(st.session_state.chat_history[-5:]) - 1:  # Add separator except last
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    <p>🚀 Powered by Google Gemini AI | Deployed on Streamlit Cloud</p>
    <p>📊 Supports Excel, CSV, and PDF files</p>
</div>
""", unsafe_allow_html=True)