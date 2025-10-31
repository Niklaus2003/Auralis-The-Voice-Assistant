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
from langchain_groq import ChatGroq
from rapidfuzz import fuzz, process
import re
import time
import string
import json


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
assert GROQ_API_KEY and DEEPGRAM_API_KEY, "Missing API keys"


PDF_DIR = "pdfs"
VECTORSTORE_DIR = "vectorstore"
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


def is_query_valid(text):
    if not text or len(text.strip()) < 3:
        return False
    words = [w.strip(string.punctuation) for w in text.strip().split()]
    if len(words) < 2:
        return False
    if not any(c.isalpha() for c in text):
        return False
    stopwords = {"the", "a", "an", "is", "was", "it", "this", "that", "and", "or", "but", "of"}
    num_stopwords = sum(1 for w in words if w.lower() in stopwords)
    if len(words) > 0 and (num_stopwords / len(words)) > 0.7:
        return False
    # Basic vowel check per word to avoid gibberish
    vowels = set("aeiou")
    if all(not any(ch in vowels for ch in word.lower()) for word in words):
        return False
    return True


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

    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)

    vectorstore_path = os.path.join(VECTORSTORE_DIR, "faiss_index")

    
    if os.path.exists(vectorstore_path):
        try:
            embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            db = FAISS.load_local(vectorstore_path, embed_model,allow_dangerous_deserialization=True)
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
            docs.clear()
            docs.extend(all_docs)
            for doc in docs:
                norm = _normalize(doc.metadata.get("source", ""))
                if norm and norm not in pdf_names:
                    pdf_names.append(norm)
            print("[INFO] Loaded existing vectorstore from disk.")
            return
        except Exception as e:
            print(f"[WARNING] Could not load vectorstore from disk: {e}. Re-indexing PDFs.")

    
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)  # Reduced from 1000/200 to 600/100
    chunks = splitter.split_documents(all_docs)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embed_model)
    docs.clear()
    docs.extend(all_docs)
    for doc in docs:
        norm = _normalize(doc.metadata.get("source", ""))
        if norm and norm not in pdf_names:
            pdf_names.append(norm)

    try:
        db.save_local(vectorstore_path)
        print(f"[INFO] Vectorstore saved to {vectorstore_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save vectorstore: {e}")


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
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)  # Reduced from 1000/200 to 600/100
    new_chunks = splitter.split_documents(new_docs)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if db is None:
        db = FAISS.from_documents(new_chunks, embed_model)
    else:
        db.add_documents(new_chunks)
    name = _normalize(filename)
    if name and name not in pdf_names:
        pdf_names.append(name)

    
    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)
    vectorstore_path = os.path.join(VECTORSTORE_DIR, "faiss_index")
    try:
        db.save_local(vectorstore_path)
        print(f"[INFO] Updated vectorstore saved to {vectorstore_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save updated vectorstore: {e}")

    return True


def get_pdf_names():
    return [doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source", "")]


def get_best_product_name(query, threshold=30):  # Lowered threshold from 50 to 30
    norm_query = _normalize(query)
    if not pdf_names:
        return None
    
    # First check for exact substring matches
    for pdf in pdf_names:
        if norm_query in pdf or pdf in norm_query:
            return pdf
    
    # Also check original PDF names from docs
    original_names = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source:
            normalized = _normalize(source)
            if normalized not in original_names:
                original_names.append(normalized)
    
    for name in original_names:
        if norm_query in name or name in norm_query:
            return name
    
    # Fuzzy matching with lower threshold
    all_names = list(set(pdf_names + original_names))
    if all_names:
        best, score, _ = process.extractOne(norm_query, all_names, scorer=fuzz.token_set_ratio)
        if score >= threshold:  # Now uses the lowered threshold of 30
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


def deepgram_stt_vad(timeout=45, for_barge_in=False):
    ws_url = (
        "wss://api.deepgram.com/v1/listen"
        "?encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}"
        f"&channels={CHANNELS}"
        "&interim_results=true"
        "&utterance_end_ms=1500"
        "&endpointing=200"
        "&vad_events=false"
        "&model=nova-2"
    )
    all_segments = []
    result_queue = queue.Queue()
    ws_closed = threading.Event()
    ws_instance = None

    def on_message(ws, message):
        if terminate_event.is_set():
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
                            
                            # For barge-in, return immediately on any speech
                            if for_barge_in and transcript.strip():
                                full_text = " ".join(all_segments).replace("  ", " ").strip()
                                print(f"[Barge-in transcript]: {full_text}")
                                if not result_queue.qsize():  # Prevent duplicates
                                    result_queue.put(full_text)
                                    ws_closed.set()
                                    threading.Thread(target=lambda: ws.close(), daemon=True).start()
                                return

            # Only close on utterance end if NOT in barge-in mode
            if data.get("type") == "UtteranceEnd" and not for_barge_in:
                print("[UtteranceEnd: User paused, finalizing transcript!]")
                ws_closed.set()
                threading.Thread(target=lambda: ws.close(), daemon=True).start()

    def on_error(ws, error):
        print(f"[WebSocket error]: {error}")

    def on_close(ws, close_status_code=None, close_msg=None):
        print("[WebSocket closed]")
        if not result_queue.qsize():  # Only add if queue is empty
            full = " ".join(all_segments).replace("  ", " ").strip()
            if full:
                print(f"===== FULL TRANSCRIPT =====\n{full}")
                result_queue.put(full)
        ws_closed.set()

    def on_open(ws):
        print("Listening for barge-in..." if for_barge_in else "Mic opened. Speak your sentence or paragraph, pause to finish.")
        
        def audio_stream():
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
                    while not ws_closed.is_set() and not terminate_event.is_set():
                        if for_barge_in and stop_playback_event.is_set():
                            break
                        try:
                            data, _ = stream.read(1024)
                            if not ws_closed.is_set():
                                ws.send(data.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                        except Exception as e:
                            if not ws_closed.is_set():
                                print(f"[Audio send error]: {e}")
                            break
            except Exception as e:
                print(f"[Audio stream error]: {e}")
        
        threading.Thread(target=audio_stream, daemon=True).start()

    try:
        ws_instance = websocket.WebSocketApp(
            ws_url,
            header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close
        )
        
        def run_websocket():
            ws_instance.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()

        # Wait for result with timeout
        try:
            result = result_queue.get(block=True, timeout=timeout)
            print(f"[STT returning]: '{result}'")
            return result if not terminate_event.is_set() else ""
        except queue.Empty:
            print("[STT Timeout - no speech detected]")
            return ""
        finally:
            ws_closed.set()
            if ws_instance:
                try:
                    ws_instance.close()
                except:
                    pass

    except Exception as e:
        print(f"[WebSocket creation error]: {e}")
        return ""


def listen_and_get_speech(timeout=30):
    log_time("Listen start")
    transcript = deepgram_stt_vad(timeout=timeout, for_barge_in=False)
    if transcript and not terminate_event.is_set():
        log_time("Transcript received")
        return transcript
    log_time("Listen timeout or terminated")
    return None


def speak_streaming(text_chunks):
    stop_playback_event.clear()
    try:
        sd.stop()
    except:
        pass
    
    for chunk in text_chunks:
        if stop_playback_event.is_set() or terminate_event.is_set():
            break
            
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
        
        try:
            response = requests.post(url, headers=headers, json={"text": chunk}, timeout=10)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(response.content)
                f.flush()
                
                try:
                    data, sr = sf.read(f.name)
                    sd.play(data, sr)
                    
                    while sd.get_stream() and sd.get_stream().active:
                        if stop_playback_event.is_set() or terminate_event.is_set():
                            sd.stop()
                            break
                        sd.sleep(10)
                except Exception as e:
                    print(f"[Audio playback error]: {e}")
                finally:
                    try:
                        os.unlink(f.name)
                    except:
                        pass
                        
        except Exception as e:
            print(f"[TTS Error]: {e}")
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
        # First, try direct PDF name matching before vector search
        query_lower = query.lower().strip()
        
        # Check if query directly matches any PDF name
        for pdf_name in pdf_names:
            if pdf_name in query_lower or query_lower in pdf_name:
                # If direct match found, search specifically for that PDF
                pdf_specific_results = db.similarity_search(query, k=3, filter={"source": pdf_name + ".pdf"})
                if pdf_specific_results:
                    return "\n\n".join(
                        f"[{doc.metadata.get('source', 'Document')}] {doc.page_content.strip()[:400]}"
                        for doc in pdf_specific_results
                    )
        
        # Also check if query matches any original PDF filenames
        for doc in docs:
            source_name = doc.metadata.get("source", "").lower().replace(".pdf", "")
            if source_name in query_lower or query_lower in source_name:
                # Search for content from this specific PDF
                matching_docs = [d for d in docs if d.metadata.get("source", "").lower() == doc.metadata.get("source", "").lower()]
                if matching_docs:
                    # Use the first few matching documents
                    limited_docs = matching_docs[:2]
                    return "\n\n".join(
                        f"[{doc.metadata.get('source', 'Document')}] {doc.page_content.strip()[:400]}"
                        for doc in limited_docs
                    )
        
        # If no direct match, fall back to regular similarity search
        results = db.similarity_search(query, k=2)
        if not results:
            return "NO_RELEVANT_CONTENT_FOUND"
        
        # Check relevance threshold for general queries
        results_with_scores = db.similarity_search_with_score(query, k=2)
        if not results_with_scores:
            return "NO_RELEVANT_CONTENT_FOUND"
        
        # Get the best similarity score (lower score means more similar)
        best_score = results_with_scores[0][1]
        
        # Use a relaxed threshold for general content
        RELEVANCE_THRESHOLD = 1.2
        
        if best_score > RELEVANCE_THRESHOLD:
            return "NO_RELEVANT_CONTENT_FOUND"
        
        # Return the relevant content
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'Document')}] {doc.page_content.strip()[:400]}"
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
    llm = ChatGroq(api_key=GROQ_API_KEY, model="moonshotai/kimi-k2-instruct-0905", temperature=0.2, streaming=True)
    agent = initialize_agent(
        tools, llm, agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={
            "system_message": (
                "You are a helpful assistant and must answer ONLY from the uploaded PDFs. "
                "Never use external knowledge. If the search tool returns 'NO_RELEVANT_CONTENT_FOUND', you must respond with exactly: 'Sorry, I can only answer questions about the content in the uploaded documents. Please ask me something related to the PDFs.' "
                "If the answer isn't clearly in the PDFs, just say 'Sorry, I couldn't find that information in the uploaded documents.' "
                "Always use the exact product names as given in the uploaded documents when referring to products. "
                "Do NOT use user-provided variations, nicknames, or shorthand terms for product names. "
                "You must ALWAYS use the PDFUniversalSearch tool first for every query before responding."
            )
        },
        handle_parsing_errors=True
    )
    return agent, pdf_names


def speak_and_barge(answer_text):
    log_time("TTS start")
    next_query_queue = queue.Queue()
    tts_complete = threading.Event()
    
    def tts_worker():
        speak_streaming(split_text_for_streaming(answer_text))
        tts_complete.set()
    
    def barge_listener():
        print("[Starting barge listener]")
        transcript = deepgram_stt_vad(timeout=120, for_barge_in=True)
        print(f"[Barge listener result]: '{transcript}'")
        
        if transcript and transcript.strip():
            print(f"[Valid barge-in detected]: {transcript}")
            next_query_queue.put(transcript)
            stop_playback_event.set()
            try:
                sd.stop()
            except:
                pass
        else:
            print("[No valid barge-in detected]")
    
    # Start threads
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    barge_thread = threading.Thread(target=barge_listener, daemon=True)
    
    tts_thread.start()
    barge_thread.start()
    
    next_query = None
    start_time = time.time()
    
    # Monitor for barge-in or TTS completion
    while not terminate_event.is_set():
        try:
            next_query = next_query_queue.get_nowait()
            print(f"[Barge-in received]: {next_query}")
            stop_playback_event.set()
            push_typing_interrupt()
            try:
                sd.stop()
            except:
                pass
            break
        except queue.Empty:
            if tts_complete.is_set():
                print("[TTS completed, checking for late barge-in]")
                try:
                    next_query = next_query_queue.get(timeout=2.0)
                    print(f"[Late barge-in]: {next_query}")
                except queue.Empty:
                    print("[No late barge-in]")
                break
            
            # Timeout safety
            if time.time() - start_time > 150:
                print("[Speak and barge timeout]")
                break
                
            time.sleep(0.1)
    
    # Cleanup
    stop_playback_event.set()
    tts_thread.join(timeout=2)
    barge_thread.join(timeout=2)
    
    try:
        sd.stop()
    except:
        pass
    
    log_time("TTS end")
    
    if next_query:
        print(f"[Returning query]: {next_query}")
    else:
        print("[No query to return]")
    
    return next_query if not terminate_event.is_set() else None


def start_voice_mode():
    global last_product

    print("[Voice mode starting]")
    terminate_event.clear()
    stop_playback_event.clear()
    listening_event.clear()

    # Initial greeting
    push_message("Agent", "Hello! I'm ready. You can talk to me anytime.")
    speak_streaming(["Hello! I'm ready. You can talk to me anytime."])

    query = None
    iteration = 0
    
    while not terminate_event.is_set():
        iteration += 1
        print(f"\n[=== Conversation iteration {iteration} ===]")
        
        stop_playback_event.clear()
        listening_event.clear()
        
        try:
            sd.stop()
        except:
            pass

        # Get user input (either from barge-in or fresh listening)
        if not query:
            print("[Listening for new input]")
            query = listen_and_get_speech(timeout=30)
            if terminate_event.is_set():
                break
            if not query:
                print("[No input received, continuing to listen]")
                continue
        else:
            print(f"[Processing barge-in query]: {query}")

        # Process the query
        push_message("User", query)
        query_ctx = resolve_pronouns(query)

        if not is_query_valid(query_ctx):
            msg = "Sorry, I couldn't understand that. Could you please ask your question again?"
            push_message("Agent", msg)
            speak_streaming([msg])
            query = None
            continue

        update_product_context(query_ctx)
        
        print(f"[Processing query]: {query_ctx}")
        log_time("Start retrieval")
        
        if terminate_event.is_set():
            break

        try:
            # Vector search
            retrieval_start = time.time()
            retrieved_results = db.similarity_search(query_ctx, k=3)
            retrieval_elapsed = time.time() - retrieval_start
            print(f"VectorDB retrieval took: {retrieval_elapsed:.2f}s")

            # Agent processing
            log_time("Agent invoke start")
            llm_start = time.time()
            agent_response = agent.invoke(query_ctx)
            llm_elapsed = time.time() - llm_start
            print(f"LLM/agent.invoke took: {llm_elapsed:.2f}s")
            
            answer = agent_response.get("output") if isinstance(agent_response, dict) else str(agent_response)
            
        except Exception as exc:
            answer = "I encountered an error processing your question. Please try asking something else."
            print(f"Agent exception: {exc}")

        log_time("Agent replied")
        
        if terminate_event.is_set():
            break

        push_message("Agent", answer)
        update_product_context(answer)
        stop_playback_event.clear()

        # Speak response and listen for barge-in
        print("[Starting speak and barge]")
        next_query = speak_and_barge(answer)
        
        if terminate_event.is_set():
            break

        # Set up for next iteration
        query = next_query
        print(f"[Next iteration query]: {query if query else 'None - will listen'}")

    print("[Voice mode terminated]")


def stop_conversation():
    print("[Stopping conversation]")
    terminate_event.set()
    stop_playback_event.set()
    try:
        sd.stop()
    except:
        pass