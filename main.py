import streamlit as st
import os
import requests
from typing import TypedDict, Sequence
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# LangSmith Integration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI-Agent-Weather-RAG"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"  # Replace with your LangSmith key

# Configuration
GROQ_API_KEY = 'groq-api-key'  # Replace with your Groq API key

# ===========================
# State Definition
# ===========================
class AgentState(TypedDict):
    query: str
    messages: Sequence[BaseMessage]
    routing_decision: str
    final_answer: str
    tool_results: dict

# ===========================
# Weather Tool (wttr.in - No API Key Required)
# ===========================
def get_weather(city: str) -> dict:
    """Fetch weather data using wttr.in (no API key required)"""
    try:
        # Clean city name
        city = city.strip()
        
        # Try with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Check if response is valid JSON
            try:
                data = response.json()
            except:
                return {'success': False, 'error': 'Invalid response from weather service'}
            
            # Validate data structure
            if 'current_condition' not in data or not data['current_condition']:
                return {'success': False, 'error': 'Invalid weather data received'}
            
            current = data['current_condition'][0]
            
            return {
                'success': True,
                'city': city.title(),
                'temperature': current.get('temp_C', 'N/A'),
                'feels_like': current.get('FeelsLikeC', 'N/A'),
                'humidity': current.get('humidity', 'N/A'),
                'description': current.get('weatherDesc', [{'value': 'Unknown'}])[0]['value'],
                'wind_speed': current.get('windspeedMiles', 'N/A')
            }
        else:
            return {'success': False, 'error': f'City not found or service unavailable (Status: {response.status_code})'}
    except requests.Timeout:
        return {'success': False, 'error': 'Request timeout - please try again'}
    except requests.RequestException as e:
        return {'success': False, 'error': f'Network error: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Error: {str(e)}'}

# ===========================
# Initialize Components
# ===========================
@st.cache_resource
def initialize_components():
    """Initialize embeddings, vector store, and LLM"""
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="all-minilm")
    
    # Load PDF
    if not os.path.exists("LogicLoom_Company_Profile_Healthcare.pdf"):
        st.error("PDF file not found!")
        return None, None, None
    
    loader = PyMuPDFLoader("LogicLoom_Company_Profile_Healthcare.pdf")
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Initialize Qdrant (in-memory)
    client = QdrantClient(":memory:")
    collection_name = "pdf_knowledge_base"
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    # Add documents to vector store
    vector_store.add_documents(splits)
    
    # Initialize LLM with LangSmith tracing
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    
    return vector_store, llm, embeddings

# ===========================
# Agent Nodes
# ===========================
def router_node(state: AgentState) -> AgentState:
    """Route query to appropriate tool"""
    query = state['query'].lower()
    
    if any(word in query for word in ['weather', 'temperature', 'forecast', 'climate']):
        state['routing_decision'] = 'weather'
    else:
        state['routing_decision'] = 'pdf_rag'
    
    state['messages'] = [HumanMessage(content=f"Routing to: {state['routing_decision']}")]
    return state

def weather_node(state: AgentState) -> AgentState:
    """Handle weather queries"""
    query = state['query']
    
    # Extract city name
    words = query.split()
    city_indicators = ['weather', 'in', 'for', 'at']
    city = None
    
    for i, word in enumerate(words):
        if word.lower() in city_indicators and i + 1 < len(words):
            city = ' '.join(words[i+1:])
            break
    
    if not city:
        city = words[-1] if words else "London"
    
    # Get weather data
    weather_data = get_weather(city)
    
    if weather_data['success']:
        answer = f"""Weather in {weather_data['city']}:
- Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
- Condition: {weather_data['description']}
- Humidity: {weather_data['humidity']}%
- Wind Speed: {weather_data['wind_speed']} mph"""
    else:
        answer = f"Failed to fetch weather data: {weather_data['error']}"
    
    state['tool_results'] = {'weather': weather_data}
    state['final_answer'] = answer
    return state

def pdf_rag_node(state: AgentState, vector_store, llm) -> AgentState:
    """Handle PDF RAG queries"""
    
    # Create RAG prompt
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.
Be concise and accurate.

Context: {context}

Question: {input}

Answer:""")
    
    # Create RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get response
    response = retrieval_chain.invoke({"input": state['query']})
    
    state['tool_results'] = {'pdf_rag': response}
    state['final_answer'] = response['answer']
    return state

def route_decision(state: AgentState) -> str:
    """Conditional routing"""
    return state['routing_decision']

# ===========================
# Build Agent Graph
# ===========================
def build_agent_graph(vector_store, llm):
    """Build LangGraph agent workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("pdf_rag", lambda state: pdf_rag_node(state, vector_store, llm))
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "weather": "weather",
            "pdf_rag": "pdf_rag"
        }
    )
    
    # Add end edges
    workflow.add_edge("weather", END)
    workflow.add_edge("pdf_rag", END)
    
    return workflow.compile()

# ===========================
# Streamlit UI
# ===========================
def main():
    st.set_page_config(
        page_title="AI Agent - Weather & RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better aesthetics
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
    }
    .answer-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .answer-content {
        font-size: 1.1rem;
        line-height: 1.8;
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .tool-badge {
        background: linear-gradient(45deg, #ff6b6b, #ee5a6f);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🤖 AI Agent: Weather & PDF RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask about weather or questions from the PDF document</div>', unsafe_allow_html=True)
    
    # Initialize components
    with st.spinner("🔄 Loading components..."):
        vector_store, llm, embeddings = initialize_components()
        
        if vector_store is None:
            st.error("Failed to initialize components")
            return
        
        agent_app = build_agent_graph(vector_store, llm)
    
    st.success("✅ Agent Ready! LangSmith tracing enabled.")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Tools Available")
        st.markdown("""
        - 🌤️ **Weather Tool**: Real-time weather (wttr.in)
        - 📄 **PDF RAG**: Answer questions from PDF
        """)
        
        st.markdown("---")
        st.info("💡 LangSmith logs are being recorded for all interactions")
    
    # Chat interface
    query = st.text_area(
        "💭 Enter your query:",
        height=100,
        placeholder="Example: 'What's the weather in London?' or 'Tell me about the company'"
    )
    
    if st.button("🚀 Submit", type="primary"):
        if not query:
            st.warning("⚠️ Please enter a query")
            return
        
        with st.spinner("🤔 Processing..."):
            # Create initial state
            initial_state = {
                'query': query,
                'messages': [],
                'routing_decision': '',
                'final_answer': '',
                'tool_results': {}
            }
            
            # Run agent
            result = agent_app.invoke(initial_state)
            
            # Display results
            st.markdown("---")
            
            # Tool badge
            st.markdown(f"""
            <div class="tool-badge">
                🎯 Tool Used: {result['routing_decision'].upper().replace('_', ' ')}
            </div>
            """, unsafe_allow_html=True)
            
            # Answer box
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-header">
                    📝 Answer
                </div>
                <div class="answer-content">
                    {result['final_answer'].replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show tool results in expander
            with st.expander("🔍 View Tool Results"):
                st.json(result['tool_results'])

if __name__ == "__main__":
    main()