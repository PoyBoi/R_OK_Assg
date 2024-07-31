import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
model_name = "decapoda-research/llama-7b-hf"  # Or another suitable model
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
chroma_db_path = "langchain_docs_db" # Specify where to store the database
langchain_docs_url = "https://python.langchain.com/en/latest/" # Starting URL 

# --- Functions ---

def scrape_langchain_docs(url):
  """Scrapes content from a LangChain documentation page."""
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')

  # Extract title and content (adjust selectors based on page structure)
  title = soup.find('h1', class_='page-title').text.strip()
  content_elements = soup.select('div.section > *') # Adjust selectors as needed
  content = " ".join([str(element) for element in content_elements]) 

  # Remove code blocks (optional)
  # content = re.sub(r'<pre>.*?</pre>', '', content, flags=re.DOTALL) 

  return {"title": title, "content": content}

def create_langchain_docs_db(url, embeddings_model, db_path):
  """Scrapes, embeds, and stores LangChain documentation in a ChromaDB."""
  # 1. Scrape the documentation
  page_data = scrape_langchain_docs(url) 

  # 2. Split into chunks
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=50,
      length_function=len,
  )
  docs = text_splitter.create_documents([page_data['content']])

  # 3. Create the database
  vectordb = Chroma.from_documents(
      docs, 
      embeddings_model, 
      persist_directory=db_path
  )
  vectordb.persist()
  print(f"Created ChromaDB at: {db_path}")
  return vectordb

def get_context_from_docs(query, vectordb):
    """Retrieves relevant context from the LangChain documentation database."""
    results = vectordb.similarity_search(query, k=3) 
    return [doc.page_content for doc in results]

def answer_question(question, vectordb, generator):
  """Answers user questions using the language model and RAG."""
  context = get_context_from_docs(question, vectordb) 
  prompt = f"""
  Answer the following question based on the provided context:

  Context:
  {context}

  Question: {question}
  Answer:
  """
  response = generator(prompt, max_length=200)[0]['generated_text']
  return response


# --- Main Execution ---

# 1. Load Language Model
generator = pipeline('text-generation', model=model_name)

# 2. Load Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# 3. Create the LangChain Documentation Database (run once)
# vectordb = create_langchain_docs_db(langchain_docs_url, embeddings, chroma_db_path)

# --- Load Existing ChromaDB ---
vectordb = Chroma(embedding_function=embeddings, persist_directory=chroma_db_path)

# 4. Ask Questions 
user_question = "How do I load a CSV file in LangChain?"
answer = answer_question(user_question, vectordb, generator)
print(answer) 