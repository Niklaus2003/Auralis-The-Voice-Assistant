import os
import websocket
import threading
import queue
import tempfile
import requests
import soundfile as sf
import sounddevice as sd
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from collections import deque
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from rapidfuzz import fuzz, process
import re
import time
import json

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
assert GROQ_API_KEY and DEEPGRAM_API_KEY, "Missing API keys"

PDF_DIR = "pdfs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_RATE = 16000
CHANNELS = 1

stop_playback_event = threading.Event()
terminate_event = threading.Event()
listening_event = threading.Event()

ws_callback = None

db = None
docs = []
pdf_names = []
agent = None
pdf_hashes = set()
chat_history = []
last_product = None
recent_products = deque(maxlen=3)

def log_time(label):
    print(f"{label}: {datetime.now()}")

def set_ws_callback(cb):
    global ws_callback
    ws_callback = cb

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def push_message(sender, text):
    side = "right" if sender == "User" else "left"
    emoji = "👤" if sender == "User" else "🤖"
    message = f"{emoji} <b>{sender}:</b> {text}"
    ts = get_timestamp()
    if ws_callback:
        ws_callback({
            "sender": sender,
            "text": text,
            "message": message,
            "side": side,
            "timestamp": ts
        }, event='new_message')

def push_typing_interrupt():
    if ws_callback:
        ws_callback({}, event='interrupt_typing')

def _normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.replace(".pdf", "").lower())

def get_file_hash(file_stream):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hasher.update(chunk)
    file_stream.seek(0)
    return hasher.hexdigest()

def load_and_index_pdfs():
    global db, docs, pdf_names, pdf_hashes
    all_docs = []
    pdf_names.clear()
    pdf_hashes.clear()
    for file in os.listdir(PDF_DIR):
        filepath = os.path.join(PDF_DIR, file)
        if file.lower().endswith(".pdf"):
            with open(filepath, "rb") as f:
                file_hash = get_file_hash(f)
                if file_hash in pdf_hashes:
                    continue
                pdf_hashes.add(file_hash)
            loader = PyMuPDFLoader(filepath)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file
                all_docs.append(doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embed_model)
    docs.clear()
    docs.extend(all_docs)
    for doc in docs:
        norm = _normalize(doc.metadata.get("source", ""))
        if norm and norm not in pdf_names:
            pdf_names.append(norm)

def index_new_pdf(filepath):
    global db, docs, pdf_names, pdf_hashes
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_hash = get_file_hash(f)
        if file_hash in pdf_hashes:
            return False
        pdf_hashes.add(file_hash)
    loader = PyMuPDFLoader(filepath)
    new_docs = loader.load()
    for doc in new_docs:
        doc.metadata["source"] = filename
    docs.extend(new_docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = splitter.split_documents(new_docs)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db.add_documents(new_chunks)
    name = _normalize(filename)
    if name and name not in pdf_names:
        pdf_names.append(name)
    return True

def get_pdf_names():
    return [doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source", "")]

def get_best_product_name(query, threshold=50):
    norm_query = _normalize(query)
    if not pdf_names:
        return None
    for pdf in pdf_names:
        if norm_query in pdf or pdf in norm_query:
            return pdf
    best, score, _ = process.extractOne(norm_query, pdf_names, scorer=fuzz.token_set_ratio)
    if score >= threshold:
        return best
    return None

def resolve_pronouns(query):
    global last_product
    pronouns = r"\b(it|this product|the product|that product|its|his|her|their)\b"
    context_name = recent_products[0] if recent_products else last_product
    if context_name:
        replaced = re.sub(pronouns, context_name, query, flags=re.I)
        return replaced
    return query

def update_product_context(input_text):
    global last_product
    match = get_best_product_name(input_text)
    if match:
        last_product = match
        recent_products.appendleft(match)

def deepgram_stt_vad(timeout=45):
    ws_url = (
        "wss://api.deepgram.com/v1/listen"
        "?encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}"
        f"&channels={CHANNELS}"
        "&interim_results=true"
        "&utterance_end_ms=1000"
        "&endpointing=100"
        "&vad_events=false"
        "&model=nova"
    )
    all_segments = []
    result_queue = queue.Queue()

    def on_message(ws, message):
        if terminate_event.is_set():  # << Stop as soon as possible
            try:
                ws.close()
            except Exception:
                pass
            return
        try:
            data = json.loads(message)
        except Exception as e:
            print(f"[JSON decode error]: {e}")
            return

        if isinstance(data, dict):
            if data.get("type") == "Results" and "channel" in data:
                channel = data["channel"]
                if isinstance(channel, dict):
                    alt = channel.get("alternatives")
                    if isinstance(alt, list) and len(alt) > 0:
                        transcript = alt[0].get("transcript", "")
                        if transcript and data.get("is_final"):
                            print(f"[Segment] {transcript}")
                            all_segments.append(transcript)

            if data.get("type") == "UtteranceEnd":
                print("[UtteranceEnd: User paused, finalizing transcript!]")
                ws.close()

            if data.get("type") == "vad" and data.get("event") == "stop":
                print("[Deepgram VAD detected silence: finalizing]")
                ws.close()
        else:
            print(f"[Deepgram message was a list]: {data}")

    def on_error(ws, error):
        print("[WebSocket error]:", error)

    def on_close(ws, *a, **k):
        full = " ".join(all_segments).replace("  ", " ").strip()
        print("[WebSocket closed]")
        print("===== FULL TRANSCRIPT =====\n", full)
        result_queue.put(full)

    def on_open(ws):
        def run():
            print("Mic opened. Speak your sentence or paragraph, pause (~1s) to finish.")
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
                    while ws.keep_running:
                        if terminate_event.is_set():
                            ws.close()
                            break
                        data, _ = stream.read(1024)
                        ws.send(data.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                print("[Mic thread error]:", e)
        threading.Thread(target=run, daemon=True).start()

    ws = websocket.WebSocketApp(
        ws_url,
        header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()

    # Wait, but bail if stopped!
    try:
        result = result_queue.get(block=True, timeout=timeout)
        if terminate_event.is_set():
            return ""
        return result
    except queue.Empty:
        print("[STT Timeout]")
        return ""

def listen_and_get_speech(timeout=30):
    log_time("Listen start")
    result_queue = queue.Queue()
    def listener():
        transcript = deepgram_stt_vad(timeout=timeout)
        if transcript:
            result_queue.put(transcript)
    t = threading.Thread(target=listener, daemon=True)
    t.start()
    t.join(timeout + 5)
    try: sd.stop()
    except Exception: pass
    if terminate_event.is_set():
        return None
    try:
        transcript = result_queue.get_nowait()
        log_time("Transcript received")
        return transcript
    except queue.Empty:
        log_time("STT timeout")
        return None

def speak_streaming(text_chunks):
    stop_playback_event.clear()
    sd.stop()
    for chunk in text_chunks:
        if stop_playback_event.is_set() or terminate_event.is_set():
            sd.stop()
            break
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json={"text": chunk}, timeout=10)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(response.content)
                f.flush()
                data, sr = sf.read(f.name)
                sd.play(data, sr)
                while sd.get_stream().active:
                    if stop_playback_event.is_set() or terminate_event.is_set():
                        sd.stop()
                        break
                    sd.sleep(10)
        except Exception as e:
            push_message("Agent", f"TTS Error: {e}")
        if terminate_event.is_set():
            break

def split_text_for_streaming(text, max_len=200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks

def create_universal_search_tool():
    def search_fn(query):
        results = db.similarity_search(query, k=3)
        if not results:
            return ""
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'Document')}] {doc.page_content.strip()[:900]}"
            for doc in results
        )
    return Tool(
        "PDFUniversalSearch",
        search_fn,
        "Use this tool to answer any question about the uploaded PDFs. Always search for relevant content."
    )

def initialize_agent_and_tools():
    global agent
    load_and_index_pdfs()
    tools = [create_universal_search_tool()]
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.2, streaming=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=2)
    agent = initialize_agent(
        tools, llm, agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory, verbose=False,
        agent_kwargs={
            "system_message": (
                "You are a helpful assistant and must answer ONLY from the uploaded PDFs. "
                "Never use external knowledge. If the answer isn't in the PDFs, just say 'Sorry, I couldn't find that information in the uploaded documents.'"
            )
        },
        handle_parsing_errors=True
    )
    return agent, pdf_names

def speak_and_barge(answer_text):
    log_time("TTS start")
    next_query_queue = queue.Queue()
    tts_thread = threading.Thread(target=speak_streaming, args=(split_text_for_streaming(answer_text),))
    def barge_listener():
        transcript = deepgram_stt_vad(timeout=30)
        if transcript:
            next_query_queue.put(transcript)
            stop_playback_event.set()
            try: sd.stop()
            except Exception: pass
    barge_thread = threading.Thread(target=barge_listener)
    tts_thread.daemon = True
    barge_thread.daemon = True
    tts_thread.start()
    barge_thread.start()
    next_query = None
    while tts_thread.is_alive():
        if not next_query_queue.empty() or terminate_event.is_set():
            if not next_query_queue.empty():
                next_query = next_query_queue.get_nowait()
            stop_playback_event.set()
            push_typing_interrupt()
            try: sd.stop()
            except Exception: pass
            break
        time.sleep(0.01)
    tts_thread.join(timeout=3)
    barge_thread.join(timeout=3)
    try: sd.stop()
    except Exception: pass
    log_time("TTS end")
    if terminate_event.is_set():
        return None
    return next_query

def start_voice_mode():
    global last_product

    terminate_event.clear()
    stop_playback_event.clear()
    listening_event.clear()

    push_message("Agent", "Hello! I'm ready. You can talk to me anytime.")
    speak_streaming(["Hello! I'm ready. You can talk to me anytime."])

    query = None
    while not terminate_event.is_set():
        stop_playback_event.clear()
        listening_event.clear()
        try: sd.stop()
        except Exception: pass

        # Only do a new listen if we do NOT have a barge-in transcript:
        if not query:
            query = listen_and_get_speech(timeout=30)
            if terminate_event.is_set():
                break
            if not query:
                continue

        push_message("User", query)
        query_ctx = resolve_pronouns(query)
        update_product_context(query_ctx)
        log_time("Start retrieval")
        if terminate_event.is_set():
            break
        try:
            retrieval_start = time.time()
            retrieved_results = db.similarity_search(query_ctx, k=3)
            retrieval_elapsed = time.time() - retrieval_start
            print(f"VectorDB retrieval took: {retrieval_elapsed:.2f}s")
        except Exception as e:
            print("Retrieval error:", e)
        log_time("Agent invoke start")
        if terminate_event.is_set():
            break
        try:
            llm_start = time.time()
            agent_response = agent.invoke(query_ctx)
            llm_elapsed = time.time() - llm_start
            print(f"LLM/agent.invoke took: {llm_elapsed:.2f}s")
            answer = agent_response.get("output") if isinstance(agent_response, dict) else agent_response
        except Exception as exc:
            answer = "That question is too long or caused an error. Please try something shorter."
            print("Agent exception:", exc)
        log_time("Agent replied")
        if terminate_event.is_set():
            break
        push_message("Agent", answer)
        stop_playback_event.clear()

        # Here's the fix: let the *next_query* from speak_and_barge be routed for next turn
        next_query = speak_and_barge(answer)
        if terminate_event.is_set():
            break
        query = next_query  # <--- THIS IS THE FIX. "query" will be handled right away if non-empty!
        # Loop: if query is not set, a new listening session will begin.

def stop_conversation():
    terminate_event.set()
    stop_playback_event.set()
    try: sd.stop()
    except Exception: pass
