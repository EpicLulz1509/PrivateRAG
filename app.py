import os
import json
import fitz
import requests
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import chromadb
import ollama
from werkzeug.utils import secure_filename
from uuid import uuid4
from io import BytesIO
from bs4 import BeautifulSoup
from services.functions import readtextfiles, chunksplitter, getembedding
from services.web_text_extract import extract_text_from_website

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['WEB_TEXT_FOLDER'] = 'web_text_inputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

chromaclient = chromadb.HttpClient(host="localhost", port=8111)
# collection = chromaclient.get_or_create_collection(name="mycollectionrag")
collection = chromaclient.get_or_create_collection(
        name="mycollectionrag", metadata={"hnsw:space": "cosine"}
    )

def pdf_to_text(pdf_folder, txt_folder):
    # Make sure the destination folder exists
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # Walk through the PDF folder
    for subdir, dirs, files in os.walk(pdf_folder):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.lower().endswith(".pdf"):
                print(f"Converting PDF: {file}")
                pdf_document = fitz.open(filepath)
            else:
                print(f"Skipping non-PDF file: {file}")
                continue

            text_file_name = os.path.join(txt_folder, file.replace(".pdf", ".txt"))

            with open(text_file_name, "w", encoding="utf-8") as text_file:
                for page_number in range(len(pdf_document)):
                    page = pdf_document.load_page(page_number)
                    text = page.get_text()
                    text_file.write(text)

            pdf_document.close()
            print(f"Saved text to {text_file_name}")

            return text_file_name


def embed_text(text):
    return ollama.embed(model="nomic-embed-text", input=text)['embeddings']




def embed_text_pdf(textdocspath):
    text_data = readtextfiles(textdocspath)
    for filename, text in text_data.items():
        chunks = chunksplitter(text)
        embeds = getembedding(chunks)
        chunknumber = list(range(len(chunks)))
        # print(chunks, embeds, chunknumber)
        ids = [filename + str(index) for index in chunknumber]
        metadatas = [{"source": filename} for index in chunknumber]
        collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)
    return ollama.embed(model="nomic-embed-text", input=text)['embeddings']

def query_rag_model(query):
    # query_embed = embed_text(query)

    query_embed = ollama.embed(model="nomic-embed-text", input=query)['embeddings']

    relateddocs = '\n\n'.join(collection.query(query_embeddings=query_embed, n_results=10)['documents'][0])
    prompt = f"{query} - Answer that question using the following text as a resource: {relateddocs}"

    # related_docs = collection.query(query_embeddings=query_embed, n_results=10)['documents'][0]
    # context = '\n\n'.join(related_docs)
    # prompt = f"{query} - Answer that question using the following text as a resource: {context}"
    rag_response = ollama.generate(model="llama3.1", prompt=prompt, stream=False)['response']
    return rag_response

def add_document_to_collection(doc_text, doc_id):
    embeddings = embed_text(doc_text)
    collection.add(documents=[doc_text], ids=[doc_id])

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'conversations' not in session:
        session['conversations'] = []

    response = None
    if request.method == 'POST':
        if 'query' in request.form:
            query = request.form.get('query')
            rag = query_rag_model(query)
            response = {"question": query, "rag": rag}
            session['conversations'].append(response)
            session.modified = True
        elif 'url' in request.form:
            url = request.form.get('url')
            try:
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                text = soup.get_text()
                # add_document_to_collection(text, doc_id=str(uuid4()))
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

    search_term = request.args.get('search', '').lower()
    conversations = session['conversations']
    if search_term:
        conversations = [conv for conv in conversations if search_term in conv['question'].lower()]
    
    return render_template('index.html', response=response, conversations=conversations, search_term=search_term)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text_file_name = pdf_to_text("uploads", "text_inputs")

    embed_text_pdf("text_inputs")



@app.route('/web-upload', methods=['POST'])
def web_upload_file():
    url = request.form.get('url')
    if not url:
        return redirect(url_for('index'))
    try:
        title, text = extract_text_from_website(url)  # define this function elsewhere
        filename = secure_filename(title) + ".txt"
        filepath = os.path.join(app.config['WEB_TEXT_FOLDER'], filename)
        # os.makedirs(filepath, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        embed_text_pdf(app.config['WEB_TEXT_FOLDER'])

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    return redirect(url_for('index'))


@app.route('/view/<int:index>')
def view(index):
    conversations = session.get('conversations', [])
    if 0 <= index < len(conversations):
        return render_template('view.html', convo=conversations[index], index=index)
    return redirect(url_for('index'))

@app.route('/delete/<int:index>', methods=['POST'])
def delete(index):
    if 'conversations' in session:
        try:
            session['conversations'].pop(index)
            session.modified = True
        except IndexError:
            pass
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear():
    session['conversations'] = []
    session.modified = True
    return redirect(url_for('index'))

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    # Delete uploaded files
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    # Delete and recreate ChromaDB collection
    try:
        print("Deleting collection...")
        chromaclient.delete_collection("mycollectionrag")
        print("Deleted")
    except Exception as e:
        print(f"Failed to delete collection: {e}")

    global collection
    collection = chromaclient.get_or_create_collection(
        name="mycollectionrag", metadata={"hnsw:space": "cosine"}
    )

    return redirect(url_for('index'))

@app.route('/export')
def export():
    conversations = session.get('conversations', [])
    json_data = json.dumps(conversations, indent=2)
    buffer = BytesIO()
    buffer.write(json_data.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='conversations.json', mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
