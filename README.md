<h1 align="center">🧠 Auralis – The Intelligent Voice Assistant</h1>

<p align="center">
Your AI-powered librarian that <b>listens, understands, and responds</b> — intelligently and instantly.
</p>

<hr>

<h2>🚀 Overview</h2>
<p>
<b>Auralis</b> is a Flask-based <b>voice assistant</b> powered by <b>LangChain</b>, <b>Groq</b>, and <b>Deepgram</b>.  
It brings <b>Retrieval-Augmented Generation (RAG)</b> to life — answering only from the uploaded PDFs and refusing to hallucinate.  
Think of it as an <b>AI librarian</b> that actually knows where things are filed. 📚🎙️
</p>

<hr>

<h2>🧩 Key Features</h2>

<ul>
<li>🎤 <b>Real-Time Voice Interaction</b> — Speak naturally, get instant spoken replies.</li>
<li>📚 <b>Retrieval-Augmented Generation (RAG)</b> — Answers strictly from your uploaded PDFs, never guesses.</li>
<li>📂 <b>Dynamic PDF Uploads</b> — Upload new PDFs anytime — they’re automatically indexed with FAISS.</li>
<li>🧠 <b>Duplicate Detection</b> — Identical PDFs are detected using <code>SHA256</code> hash comparison, preventing re-indexing (try <code>new.pdf</code>!).</li>
<li>💬 <b>Contextual Memory</b> — Keeps short-term memory of the conversation for smoother interactions.</li>
<li>🗣️ <b>Barge-in Control</b> — Interrupt the assistant mid-sentence by speaking over it — because patience is overrated.</li>
<li>🧾 <b>PDF-based Knowledge</b> — Ask anything about the company’s product PDFs — from features to specifications, and Auralis responds instantly.</li>
<li>⚙️ <b>WebSocket + Flask-SocketIO</b> — Real-time streaming messages between frontend and backend.</li>
</ul>

<hr>

<h2>🧠 How It Works</h2>

<ol>
<li>📥 <b>Upload PDFs:</b> All uploaded files are stored in the <code>pdfs/</code> folder and indexed with <b>FAISS</b> using <b>HuggingFace Embeddings</b>.</li>
<li>🔍 <b>Query Processing:</b> When you ask a question, Auralis uses <b>LangChain</b> + <b>Groq LLaMA 3.1</b> to retrieve only relevant chunks.</li>
<li>🗣️ <b>Voice Pipeline:</b> 
    <ul>
      <li><b>Speech-to-Text (STT):</b> Deepgram converts your speech into text.</li>
      <li><b>LLM Reasoning:</b> The Groq LLM finds precise PDF-based answers via the RAG system.</li>
      <li><b>Text-to-Speech (TTS):</b> Deepgram’s <i>aura-asteria-en</i> voice responds naturally — with barge-in interruption support.</li>
    </ul>
</li>
<li>📊 <b>Duplicate Handling:</b> Even if the same file is renamed (like <code>new.pdf</code>), it’s identified and skipped — saving compute and time.</li>
<li>🧠 <b>Memory Context:</b> The system recalls previous user questions within a short context window for continuity.</li>
</ol>

<hr>

<h2>🧰 Tech Stack</h2>

<ul>
<li>🧠 <b>LangChain</b> – RAG pipeline and conversational logic</li>
<li>⚡ <b>Groq LLaMA 3.1</b> – High-speed inference engine</li>
<li>🎧 <b>Deepgram</b> – STT + TTS (Speech I/O)</li>
<li>📂 <b>FAISS</b> – Vector search for document retrieval</li>
<li>💬 <b>Flask + Socket.IO</b> – Backend & real-time communication</li>
<li>🧮 <b>HuggingFace Embeddings</b> – Text vectorization for semantic search</li>
</ul>

<hr>

<h2>⚙️ Installation</h2>

<pre>
# Clone the repository
git clone https://github.com/Niklaus2003/Auralis-The-Voice-Assistant.git
cd Auralis-The-Voice-Assistant

# Install dependencies
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ Usage</h2>

<pre>
# Run the Flask app
python app.py
</pre>

<p>
Then open <b>http://localhost:5001</b> in your browser.  
Upload PDFs, click the mic, and start talking.  
Auralis listens, thinks, and replies — both text and voice — in real time.
</p>

<hr>

<h2>🧩 File Highlights</h2>

<ul>
<li><b>app.py</b> – Main Flask server managing routes, uploads, and WebSocket events.</li>
<li><b>voice_backend.py</b> – Core logic: Deepgram STT/TTS, FAISS indexing, LangChain RAG, and barge-in handling.</li>
<li><b>pdfs/</b> – Folder containing uploaded product PDFs (including <code>new.pdf</code> for duplicate testing).</li>
<li><b>templates/</b> – Frontend HTML files for the web interface.</li>
<li><b>vectorstore/faiss_index/</b> – Stored vector database for RAG retrieval.</li>
</ul>

<hr>

<h2>🌟 Future Enhancements</h2>
<ul>
<li>🌐 Multilingual voice support</li>
<li>📱 Mobile-optimized interface</li>
<li>🧩 Long-term memory persistence</li>
<li>☁️ Cloud sync for documents and context</li>
</ul>

<hr>

<h2>📸 Example Interaction</h2>

<p>
Ask: <i>“What are the key features of product X from the uploaded PDFs?”</i><br>
Auralis searches the correct document, retrieves accurate info, and answers with voice in seconds.  
No hallucinations. No distractions. Just pure, PDF-backed knowledge.
</p>

<hr>

<h2>💡 Summary</h2>

<p>
Auralis combines <b>AI-driven voice interaction</b> with <b>document-grounded intelligence</b>.  
It’s not just a chatbot — it’s a conversational RAG system that listens smarter, speaks naturally, and remembers intelligently.  
Upload, ask, interrupt, repeat — Auralis never loses context.
</p>

<hr>

<p align="center">
⭐ If you like this project, give it a star and try uploading your own PDFs! ⭐
</p>
