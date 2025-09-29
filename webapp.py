from flask import Flask, render_template, jsonify, request, session
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'medical-chatbot')

# Determine if we're in demo mode
if not PINECONE_API_KEY or PINECONE_API_KEY in ['your-pinecone-api-key-here', 'pc-demo-key-replace-with-your-actual-key']:
    print("Demo mode: Skipping Pinecone initialization")
    DEMO_MODE = True
else:
    DEMO_MODE = False

def download_huggingface_embedding():
    print("Loading HuggingFace embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print("HuggingFace embeddings loaded successfully")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# Initialize embeddings and document search
embeddings = None
docsearch = None

if not DEMO_MODE:
    print("Initializing embeddings...")
    embeddings = download_huggingface_embedding()
    if embeddings:
        try:
            docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
            print("Successfully connected to Pinecone index")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            docsearch = None
else:
    print("Demo mode: Skipping embeddings and Pinecone initialization")

# Enhanced medical prompt template
prompt_template = """
You are MediBot, an AI medical assistant designed to provide helpful, accurate, and evidence-based medical information. 

IMPORTANT GUIDELINES:
- Always emphasize that your advice is for informational purposes only
- Recommend consulting healthcare professionals for serious concerns
- Be empathetic and supportive in your responses
- If you detect emergency symptoms, advise immediate medical attention
- Provide clear, easy-to-understand explanations
- Include relevant medical context when appropriate

Context from medical literature: {context}

Patient Question: {question}

Please provide a comprehensive, helpful response that:
1. Addresses the patient's specific concern
2. Explains relevant medical concepts in simple terms
3. Suggests appropriate next steps or recommendations
4. Includes important disclaimers about seeking professional medical care

Response:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize LLM based on available API keys
def initialize_llm():
    if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key-here':
        print("Using OpenAI GPT model")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=OPENAI_API_KEY
        )
    elif DEMO_MODE:
        print("Demo mode: Skipping LLM initialization")
        return None
    else:
        print("No compatible LLM found, using demo mode")
        return None

llm = initialize_llm()

# Initialize Groq client separately for RAG
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != 'your-groq-api-key-here':
    print("Initializing Groq client for RAG")
    groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize QA chain
if docsearch and not DEMO_MODE and llm:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
else:
    print("Demo mode: Using direct LLM without retrieval")
    qa = None

# Emergency keywords detection
EMERGENCY_KEYWORDS = [
    'chest pain', 'heart attack', 'stroke', 'seizure', 'unconscious', 
    'severe bleeding', 'difficulty breathing', 'choking', 'overdose',
    'severe allergic reaction', 'anaphylaxis', 'suicide', 'self harm'
]

def detect_emergency(text):
    """Detect potential emergency situations in user input"""
    text_lower = text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def enhance_response(response, user_input):
    """Enhance the response with additional medical context and warnings"""
    enhanced = response
    
    # Add emergency warning if needed
    if detect_emergency(user_input):
        emergency_warning = "\n\n🚨 **EMERGENCY ALERT**: If this is a medical emergency, please call emergency services (911) immediately or go to the nearest emergency room. Do not rely on this chatbot for emergency medical situations."
        enhanced = emergency_warning + "\n\n" + enhanced
    
    # Add general disclaimer
    disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment."
    enhanced += disclaimer
    
    return enhanced



@app.route("/")
def index():
    # Initialize session for conversation history
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"User input: {msg}")
        
        # Handle different LLM types
        if groq_client and docsearch and not DEMO_MODE:
            # Use Groq with RAG
            try:
                # Get relevant documents
                docs = docsearch.similarity_search(msg, k=5)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Create prompt with context
                full_prompt = f"""You are MediBot, an AI medical assistant designed to provide helpful, accurate, and evidence-based medical information.

IMPORTANT GUIDELINES:
- Always emphasize that your advice is for informational purposes only
- Recommend consulting healthcare professionals for serious concerns
- Be empathetic and supportive in your responses
- If you detect emergency symptoms, advise immediate medical attention
- Provide clear, easy-to-understand explanations
- Include relevant medical context when appropriate

Context from medical literature: {context}

Patient Question: {msg}

Please provide a comprehensive, helpful response that:
1. Addresses the patient's specific concern
2. Explains relevant medical concepts in simple terms
3. Suggests appropriate next steps or recommendations
4. Includes important disclaimers about seeking professional medical care

Response:"""
                
                # Get response from Groq
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1000
                )
                response = chat_completion.choices[0].message.content
                enhanced_response = enhance_response(response, msg)
                
            except Exception as e:
                print(f"Error with Groq RAG: {e}")
                enhanced_response = "I'm sorry, I encountered an error processing your request. Please try again."
                
        elif qa and not DEMO_MODE:
            # Use traditional QA system (OpenAI or local LLM)
            result = qa.invoke({"query": msg})
            response = result.get("result", "I'm sorry, I couldn't generate a response.")
            enhanced_response = enhance_response(response, msg)
            
        else:
            # Demo mode or no QA system available
            demo_response = f"""Thank you for your question about: "{msg}"

I'm currently running in demo mode. In the full version, I would:
1. Search through medical literature for relevant information
2. Provide evidence-based medical guidance
3. Offer appropriate recommendations and next steps

**Important Medical Disclaimer**: This is for informational purposes only. Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment. For emergencies, call 911 immediately.

**Demo Note**: To access the full RAG capabilities, please configure your Pinecone API key in the .env file."""
            
            enhanced_response = enhance_response(demo_response, msg)
        
        # Store conversation in session
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        session['conversation_history'].append({
            'user': msg,
            'bot': enhanced_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 conversations to manage session size
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]
        
        session.modified = True
        
        print(f"Bot response: {enhanced_response[:100]}...")
        return enhanced_response
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history"""
    session['conversation_history'] = []
    session.modified = True
    return jsonify({"status": "success"})

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "pinecone_connected": docsearch is not None,
        "llm_type": "openai" if OPENAI_API_KEY else "local",
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

if __name__ == '__main__':
    print("Starting MediBot application...")
    print(f"Pinecone connected: {docsearch is not None}")
    print(f"LLM type: {'OpenAI' if OPENAI_API_KEY else 'Local Llama'}")
    app.run(host="0.0.0.0", port=8080, debug=True)

