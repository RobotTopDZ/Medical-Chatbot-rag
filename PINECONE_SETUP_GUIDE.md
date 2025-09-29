# 🔑 Pinecone API Key Setup Guide

## Step 1: Get Your Free Pinecone API Key

1. **Sign up for Pinecone** (Free Plan Available):
   - Go to: https://app.pinecone.io
   - Click "Sign Up" and create a free account
   - Choose the **Starter Plan** (Free tier with 100K vectors)

2. **Get Your API Key**:
   - After signing up, go to your Pinecone dashboard
   - Navigate to "API Keys" section
   - Copy your API key (it will look like: `pc-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

3. **Update Your .env File**:
   - Open your `.env` file in this project
   - Replace `your-pinecone-api-key-here` with your actual API key
   - Save the file

## Step 2: Run the Setup Script

Once you have your Pinecone API key configured:

```bash
python setup_pinecone.py
```

This script will:
- ✅ Connect to Pinecone using your API key
- 📝 Create a new index called "medical-chatbot"
- 📚 Load medical documents from the `data/` folder
- ✂️ Split documents into searchable chunks
- 🤖 Generate embeddings using HuggingFace
- 🔄 Upload everything to your Pinecone vector database
- 🧪 Test the setup with a sample query

## Step 3: Launch Full Mode

After successful setup:

```bash
python webapp.py
```

Your medical chatbot will now run with full RAG capabilities!

## Free Tier Limits

The Pinecone Starter Plan includes:
- 100K vectors (plenty for medical documents)
- 1 index
- Community support

Perfect for this medical chatbot project!

## Troubleshooting

If you encounter issues:
1. Verify your API key is correct in the `.env` file
2. Check your internet connection
3. Ensure the `data/medical_book.pdf` file exists
4. Run `pip install -r requirements.txt` to install dependencies

## Next Steps

Once setup is complete, your medical chatbot will have:
- 🔍 Semantic search through medical literature
- 🤖 Groq-powered AI responses
- 📚 Evidence-based medical information
- ⚡ Fast vector similarity search