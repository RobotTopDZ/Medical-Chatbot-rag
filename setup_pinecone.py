import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

# Load environment variables
load_dotenv()

def setup_pinecone_database():
    """Set up Pinecone vector database with medical documents"""
    
    # Get API key from environment
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key or api_key == 'your-pinecone-api-key-here':
        print("❌ Error: Please set a valid PINECONE_API_KEY in your .env file")
        return False
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=api_key)
        print("✅ Successfully connected to Pinecone")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False
    
    # Configuration
    index_name = "medical-chatbot"
    dimension = 384  # Dimension for all-MiniLM-L6-v2 model
    
    # Check if index exists, create if not
    try:
        existing_indexes = pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]
        
        if index_name not in index_names:
            print(f"📝 Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("✅ Index created successfully")
        else:
            print(f"✅ Index '{index_name}' already exists")
    except Exception as e:
        print(f"❌ Error with index operations: {e}")
        return False
    
    # Load medical documents
    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        print(f"❌ Error: Data directory not found at {data_path}")
        return False
    
    print("📚 Loading medical documents...")
    try:
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return False
    
    # Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(text_chunks)} text chunks")
    
    # Initialize embeddings
    print("🤖 Initializing embeddings model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✅ Embeddings model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Create vector store
    print("🔄 Creating vector store and uploading documents...")
    try:
        vectorstore = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=index_name
        )
        print("✅ Vector store created and documents uploaded successfully")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False
    
    # Test the vector store
    print("🧪 Testing vector store with sample query...")
    try:
        test_query = "What are the symptoms of diabetes?"
        results = vectorstore.similarity_search(test_query, k=3)
        print(f"✅ Test successful! Found {len(results)} relevant documents")
        print(f"Sample result: {results[0].page_content[:100]}...")
    except Exception as e:
        print(f"❌ Error testing vector store: {e}")
        return False
    
    print("\n🎉 Pinecone setup completed successfully!")
    print(f"📊 Index: {index_name}")
    print(f"📄 Documents: {len(documents)}")
    print(f"🔤 Text chunks: {len(text_chunks)}")
    print("🚀 Your medical chatbot is ready to use!")
    
    return True

if __name__ == "__main__":
    print("🏥 Setting up Medical Chatbot Pinecone Database")
    print("=" * 50)
    
    success = setup_pinecone_database()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("You can now run the webapp with full RAG capabilities.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)