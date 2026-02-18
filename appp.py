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
import time

# ============================================
# GEMINI API CONFIGURATION
# ============================================

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Gemini Document Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Get API key from secrets (for Streamlit Cloud) or environment (for local)
def get_api_key():
    """Get Gemini API key from various sources"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets["GEMINI_API_KEY"]
    except:
        try:
            # Try environment variable (for local development)
            return os.environ.get("GEMINI_API_KEY", "")
        except:
            return ""

GEMINI_API_KEY = get_api_key()

# Show warning if no API key
if not GEMINI_API_KEY:
    st.error("""
    ⚠️ **Gemini API Key Not Found!**
    
    To use this app, you need to add your Gemini API key:
    
    **For Local Development:**
    - Create a `.env` file with: `GEMINI_API_KEY=your-key-here`
    - Or set environment variable: `set GEMINI_API_KEY=your-key-here`
    
    **For Streamlit Cloud:**
    - Go to app settings → Secrets
    - Add: `GEMINI_API_KEY = "your-key-here"`
    """)
    st.stop()

# Using confirmed working model
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

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
                # Excel file
                self.df = pd.read_excel(uploaded_file, engine='openpyxl')
                self.file_type = 'excel'
                self._generate_summary()
                return True, f"✅ Excel loaded: {len(self.df)} rows, {len(self.df.columns)} columns"
            
            elif file_extension == 'csv':
                # CSV file
                self.df = pd.read_csv(uploaded_file)
                self.file_type = 'csv'
                self._generate_summary()
                return True, f"✅ CSV loaded: {len(self.df)} rows, {len(self.df.columns)} columns"
            
            elif file_extension == 'pdf':
                # PDF file
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        else:
                            text += f"[Page {page_num+1}: No text extracted]\n"
                    except Exception as e:
                        text += f"[Page {page_num+1}: Error - {str(e)}]\n"
                
                self.document_text = text
                self.file_type = 'pdf'
                return True, f"✅ PDF loaded: {len(pdf_reader.pages)} pages, {len(text)} characters"
            
            else:
                return False, "❌ Unsupported file type. Please upload Excel, CSV, or PDF."
                
        except Exception as e:
            return False, f"❌ Error processing file: {str(e)}"
    
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
            for col in self.df.columns[:10]:  # Limit to first 10 columns
                dtype = "NUMERIC" if pd.api.types.is_numeric_dtype(self.df[col]) else "TEXT"
                missing = self.df[col].isnull().sum()
                context += f"- {col}: {dtype} (missing: {missing})\n"
            
            # Sample data
            context += f"\nFIRST 5 ROWS:\n"
            context += self.df.head().to_string()
            
            # Statistics for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                context += f"\n\nNUMERIC COLUMNS STATISTICS:\n"
                for col in numeric_cols[:5]:  # Limit to first 5
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
{self.document_text[:2000]}"""  # Limited to 2000 chars for PDFs
        
        return "No data loaded"
    
    def calculate_direct(self, column, operation):
        """Direct calculation without AI"""
        if self.df is None:
            return None, "No data loaded"
        
        if column not in self.df.columns:
            return None, f"Column '{column}' not found"
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            if operation == "Count":
                return len(self.df[column].dropna()), f"Count of {column}"
            return None, f"Column '{column}' is not numeric (cannot calculate {operation})"
        
        try:
            if operation == "Sum":
                return self.df[column].sum(), f"Sum of {column}"
            elif operation == "Average":
                return self.df[column].mean(), f"Average of {column}"
            elif operation == "Min":
                return self.df[column].min(), f"Minimum of {column}"
            elif operation == "Max":
                return self.df[column].max(), f"Maximum of {column}"
            elif operation == "Count":
                return self.df[column].count(), f"Count of {column}"
            elif operation == "Median":
                return self.df[column].median(), f"Median of {column}"
            elif operation == "Std Dev":
                return self.df[column].std(), f"Standard Deviation of {column}"
        except Exception as e:
            return None, f"Error calculating {operation}: {str(e)}"
        
        return None, "Unknown operation"
    
    def execute_calculation(self, calculation_code):
        """Execute calculation code safely"""
        try:
            # Create safe namespace with allowed functions
            namespace = {
                'df': self.df,
                'pd': pd,
                'np': np,
                'sum': sum,
                'len': len,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
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
        self.model = "gemini-1.5-flash"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
    
    def query(self, prompt, context="", max_retries=2):
        """Send query to Gemini AI with retry logic"""
        
        full_prompt = f"""You are an expert data analyst. Analyze this data and answer the user's question.

DATA CONTEXT:
{context[:2500]}

USER QUESTION: {prompt}

INSTRUCTIONS:
1. If it's a calculation, provide the answer directly
2. If you need to show code, put it in ```python``` blocks
3. Be clear and concise
4. If you can't find the answer, say so honestly

RESPONSE:"""

        for attempt in range(max_retries):
            try:
                payload = {
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                        "topP": 0.8,
                        "topK": 40
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}",
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'candidates' in data and len(data['candidates']) > 0:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        return "No response generated. Please try again."
                
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return "⚠️ Rate limit reached. Please wait a moment and try again."
                
                elif response.status_code == 403:
                    return "⚠️ Invalid API key. Please check your Gemini API key."
                
                elif response.status_code == 404:
                    # Model not found - try alternative
                    if attempt == 0:
                        self.model = "gemini-1.0-pro"
                        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                        continue
                    return f"⚠️ Model not available. Please try again later."
                
                else:
                    return f"⚠️ API Error ({response.status_code}). Please try again."
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return "⚠️ Request timed out. Please try again."
            
            except requests.exceptions.ConnectionError:
                return "⚠️ Connection error. Please check your internet connection."
            
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                return f"⚠️ Error: {str(e)}"
        
        return "⚠️ Failed after multiple attempts. Please try again."

# ============================================
# Initialize Session State
# ============================================

if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if 'gemini' not in st.session_state:
    st.session_state.gemini = GeminiInterface(GEMINI_API_KEY)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================
# Main App UI
# ============================================

# Header
st.markdown("<h1 class='main-header'>🤖 Gemini Document Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Upload any file - Ask anything - Get instant AI-powered answers")

# Sidebar
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_optimized.gif", width=250)
    st.markdown("## 🚀 Gemini AI Status")
    
    if GEMINI_API_KEY:
        st.success("✅ API Key Connected")
        st.info(f"Model: {GEMINI_MODEL}")
    else:
        st.error("❌ API Key Missing")
    
    st.markdown("---")
    
    # File upload section
    st.markdown("## 📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv', 'pdf'],
        help="Upload Excel, CSV, or PDF files (max 200MB)"
    )
    
    if uploaded_file:
        with st.spinner("📊 Processing file..."):
            success, message = st.session_state.processor.process_file(uploaded_file)
            if success:
                st.success(message)
                
                # File statistics
                st.markdown("### 📊 File Statistics")
                if st.session_state.processor.file_type in ['excel', 'csv']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", st.session_state.processor.data_summary.get('rows', 0))
                    with col2:
                        st.metric("Columns", st.session_state.processor.data_summary.get('columns', 0))
                    
                    # Column list in expander
                    with st.expander("📋 View All Columns"):
                        for col in st.session_state.processor.df.columns:
                            dtype = "🔢" if pd.api.types.is_numeric_dtype(st.session_state.processor.df[col]) else "📝"
                            missing = st.session_state.processor.data_summary['missing_values'][col]
                            st.write(f"{dtype} **{col}** (missing: {missing})")
                else:
                    st.metric("Pages", len(st.session_state.processor.document_text.split('\n')))
                    st.metric("Characters", len(st.session_state.processor.document_text))
            else:
                st.error(message)
    
    st.markdown("---")
    
    # Quick tips
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **For Excel/CSV:**
        - "What is the sum of [column]?"
        - "Show me the average of each column"
        - "Find the maximum value in [column]"
        - "Create a bar chart of sales by region"
        - "What's the trend over time?"
        
        **For PDF:**
        - "Summarize this document"
        - "Extract all dates mentioned"
        - "What are the key findings?"
        """)

# Main content area
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
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_column = st.selectbox(
            "Select column",
            st.session_state.processor.df.columns,
            key="calc_column_select"
        )
    
    with col2:
        operation = st.selectbox(
            "Operation",
            ["Sum", "Average", "Min", "Max", "Count", "Median", "Std Dev"],
            key="calc_op_select"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Calculate", key="calc_button", use_container_width=True):
            result, message = st.session_state.processor.calculate_direct(selected_column, operation)
            if result is not None:
                st.success(f"**{message}:** {result:,.2f}")
            else:
                st.warning(message)

# Question input section
st.markdown("---")
st.markdown("## 💭 Ask Gemini About Your Data")

question = st.text_area(
    "What would you like to know?",
    placeholder="e.g., What is the sum of column_name? or Summarize this data",
    height=100,
    key="question_input"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_button = st.button("🚀 Ask Gemini", type="primary", key="ask_button", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", key="clear_button", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Process question
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
            
            # Display response
            st.markdown("### 📝 Answer")
            st.markdown(f"<div class='result-box'>{response}</div>", unsafe_allow_html=True)
            
            # Extract and execute Python code if present
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, response, re.DOTALL)
            
            if code_matches and st.session_state.processor.df is not None:
                with st.expander("🔢 View Calculation Code"):
                    for i, code in enumerate(code_matches):
                        st.code(code, language='python')
                        
                        # Execute the code
                        success, result = st.session_state.processor.execute_calculation(code)
                        
                        if success:
                            st.success(f"**Result:** {result}")
                        else:
                            st.info(f"Note: {result}")
            
            # Save to history
            st.session_state.chat_history.append({
                'question': question,
                'response': response[:150] + "..." if len(response) > 150 else response,
                'time': datetime.now().strftime("%H:%M:%S")
            })

# Chat history display
if st.session_state.chat_history:
    with st.expander("📜 Recent Questions History"):
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A:** {chat['response']}")
            st.markdown(f"*{chat['time']}*")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🚀 Powered by Google Gemini 1.5 Flash</p>
    <p>Upload any file - Ask anything - Get instant answers</p>
    <p style='font-size: 0.8rem;'>Made with Streamlit • Free for everyone</p>
</div>
""", unsafe_allow_html=True)