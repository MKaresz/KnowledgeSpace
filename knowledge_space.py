import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
import faiss
import numpy as np
import ollama
import json
import hashlib
from collections import Counter
import gradio as gr

# IMPORTANT REQUIREMENT: Ollama agent with "ollama pull nomic-embed-text" for text embedding

# TODO: Add a NOTES folder and a button to dump the actual output text into a timestamped note.
# TODO: Get model: gemma4:e4b, deepseek-coder-v2:16b
# TODO: critic with another LLM?


# GLOBALS
DATA_DIR = "data"
IDX_DIR = "idx"
# TODO: find a simple better solution
# index to keep track of the last used ID in vectorization
INDEX_ID = 0


# suitable for only single user
app_config = {
    "answer_type": "compact",
    "model": "phi4-mini:latest",
    "embedding_dim": 768,  # nomic-embed-text
    "top_k": 5,
    "only_source": True,
    "temperature": 0.1,
    "answer_length": 400
}


def get_embedding(text):
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']


def get_subdirectory_paths(root_dir):
    subdirectory_paths = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            subdirectory_paths.append(item_path)
            # Recursively add subdirectories
            subdirectory_paths.extend(get_subdirectory_paths(item_path))

    return subdirectory_paths


def get_last_folder(path):
    # Get the directory part of the path
    dir_path = os.path.dirname(path)
    # Extract the last folder name from the directory path
    last_folder = os.path.basename(dir_path)

    return last_folder


def build_chunks(path: str):
    # Read only MD files
    docs = SimpleDirectoryReader(path, required_exts=[".md"]).load_data()

    if not docs:
        raise ValueError("No markdown documents found")

    print(f"Number of documents in folder: {len(docs)} in {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50
    )

    chunks = []
    for doc in docs:
        book_name = doc.metadata.get("file_name")
        for chunk in splitter.split_text(doc.text):
            chunks.append({
                "text": chunk,
                "book_id": book_name
            })

    print(f"Split into {len(chunks)} from folder: {path}")
    return chunks


def create_embeddings(path: str, chunks: list, file_name: str):
    print("Creating embeddings...")
    embeddings = [get_embedding(chunk["text"]) for chunk in chunks]
    embeddings_np = np.array(embeddings).astype('float32')
    n_vectors = embeddings_np.shape[0]

    create_k_centroids(embeddings_np, file_name)

    print(f"Number of vectors: {n_vectors} to embedd.")
    if n_vectors < 1000: # only for test small data
        index = faiss.IndexFlatL2(int(app_config["embedding_dim"]))
        index.add(embeddings_np)
    else:
        nlist = int(np.sqrt(n_vectors))
        quantizer = faiss.IndexFlatL2(int(app_config["embedding_dim"]))
        index = faiss.IndexIVFFlat(quantizer, int(app_config["embedding_dim"]), nlist)
        index.train(embeddings_np)
        index.add(embeddings_np)

    # Save
    print(f"FAISS index has {index.ntotal} vectors, saving now to disk in {path}.")

    # write out faiss vector index with chunk id
    faiss_filename_name = file_name + ".index"
    faiss_path = str(Path.joinpath(Path(path), faiss_filename_name))
    faiss.write_index(index, faiss_path)

    # write out np chunks with id
    np_filename_name = file_name
    np_path = str(Path.joinpath(Path(path), np_filename_name))
    np.save(np_path, chunks, allow_pickle=True)


def create_k_centroids(vectors, name: str):
    global INDEX_ID
    # Make sure top_k is valid.
    top_k = min(int(app_config["top_k"]), vectors.size)

    # Normalize all vectors for cosine-like similarity.
    norm_vectors = vectors.copy()
    faiss.normalize_L2(norm_vectors)

    # Use k-means clustering to find centroids.
    kmeans = KMeans(n_clusters=top_k, random_state=42)
    kmeans.fit(norm_vectors)
    cluster_centers = kmeans.cluster_centers_

    # Build a final FAISS index with the cluster centroids.
    topic_index = faiss.IndexIDMap2(faiss.IndexFlatIP(int(app_config["embedding_dim"])))

    # Create unique IDs for each centroid
    encoded_name = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
    id_list = np.array([encoded_name + i for i in range(len(cluster_centers))], dtype=np.int64)
    # book numb of indices
    INDEX_ID += len(cluster_centers)

    # Assign IDs to the topic index
    topic_index.add_with_ids(cluster_centers, id_list)

    # Store centroids with name for get back most relevant area of topic
    store_metadata_and_vectors(name, cluster_centers)

    # generate faiss index to topic np dic array
    build_faiss_topics_index()


def store_metadata_and_vectors(name, cluster_centers):
    # Store metadata in np
    print("Updating topics file...")

    # zipped archive of named NumPy multiple arrays .npy is just one array
    # conceptually a dict[str → np.ndarray]
    file_path = Path(IDX_DIR).joinpath("topics.npz")

    # Load existing data (if any)
    data = {}
    if file_path.exists():
        data = dict(np.load(file_path, allow_pickle=False))

    # Update / insert
    data[name] = cluster_centers

    # Save back
    np.savez(file_path, **data)


def build_faiss_topics_index():
    file_path = Path(IDX_DIR).joinpath("topics.npz")
    if file_path.exists():
        vectors = []
        id_to_topic = {}

        data = dict(np.load(file_path))

        current_id = 0
        for topic, vecs in data.items():
            for vec in vecs:
                vectors.append(vec)
                id_to_topic[current_id] = topic
                current_id += 1

        vectors = np.vstack(vectors).astype("float32")
        #dim = vectors.shape[1]
        index = faiss.IndexFlatL2(int(app_config["embedding_dim"]))
        index.add(vectors)

        # Save the Faiss index to a file (optional)
        index_path = Path(IDX_DIR).joinpath("topics_index.idx")
        faiss.write_index(index, str(index_path))

        # save metadata
        metadata_path = Path(IDX_DIR).joinpath("id_to_topic.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(id_to_topic, f, indent=2)

    else:
        print("No topics file was found!")


def load_search_topics():
    file_path = Path(IDX_DIR).joinpath("topics.npz")
    index_path = Path(IDX_DIR).joinpath("topics_index.idx")
    metadata_path = Path(IDX_DIR).joinpath("id_to_topic.json")

    if file_path.exists() and index_path.exists() and metadata_path.exists():
        # original centroid vectors with topics
        data = dict(np.load(file_path))
        # optimized vector database
        index = faiss.read_index(str(index_path))
        # save metadata for vector to topic
        with open(metadata_path, "r", encoding="utf-8") as f:
            id_to_topic = json.load(f)
        # keys come back as strings → fix, we need as integer values
        id_to_topic = {int(k): v for k, v in id_to_topic.items()}
        return index, data, id_to_topic

    return None

def get_relevant_topics(index, id_to_topic, query: str, k: int):
    question_emb = np.array([get_embedding(query)], dtype="float32")
    distances, indices = index.search(question_emb, k)

    relevant_topics = []
    for idx in indices[0]:
        topic = id_to_topic.get(idx)
        if topic is not None:
            relevant_topics.append(topic)

    # Remove duplicates while preserving order
    relevant_topics = list(dict.fromkeys(relevant_topics))

    #print("relevant topics:", relevant_topics)
    return relevant_topics


def query_topics(index, chunks, query):
    # Embed the question
    question_emb = np.array([get_embedding(query)]).astype('float32')

    # Search FAISS for top-k similar chunks
    distances, indices = index.search(question_emb, int(app_config["top_k"]))
    retrieved_chunks = [chunks[i]["text"] for i in indices[0]]
    context = " ".join(retrieved_chunks)
    sources = Counter([chunks[i]["book_id"] for i in indices[0]])
    sources_str = ', '.join(f"{element}:{count}" for element, count in sources.items())
    return context, sources_str

def query_llm(context, query):
    use_only_context = ""
    if bool(app_config["only_source"]):
        use_only_context = "Based you answer based only on the given context."

    # Build prompt with context
    prompt = f"""You are a precise assistant using answer type {app_config["answer_type"]}.

    Context: {context}

    Question: {query}
    
    {use_only_context}
    """

    # Generate with Ollama LLM
    response = ollama.generate(
        model=app_config["model"],
        prompt=prompt,
        options={"temperature": float(app_config["temperature"]), "num_predict": int(app_config["answer_length"])}
    )
    return response['response']


def query_engine(query):
    # don't use database sources is a ease of use decision
    # but still we want to separate it as it's not coming from our database
    if not app_config['only_source']:
        return str(query_llm("", query) + "\n\nUsing not verifiable sources!")

    index, data, id_to_topic = load_search_topics()
    relevant_topics = get_relevant_topics(index, id_to_topic, query, int(app_config["top_k"]))
    context = []
    sources = []
    for topic in relevant_topics:
        index_name = str(topic + ".index")
        chunks_name = str(topic + ".npy")
        index_path = str(Path(IDX_DIR).joinpath(index_name))
        chunk_path = str(Path(IDX_DIR).joinpath(chunks_name))
        # TODO CACHE IN ALL CHUNKS FOR SPEED
        # load relevant chunks
        index = faiss.read_index(index_path)
        chunks = np.load(chunk_path, allow_pickle=True)
        topic_context, topic_sources = query_topics(index, chunks, query)
        context.append(topic_context)
        sources.append(topic_sources)

    output_sources = set(sources)
    output_answer = query_llm(context, query)
    return str(output_answer + "\n\nSources:" + str(output_sources))


def create_search_idx():
    # check if data folder exists
    data_path = Path(DATA_DIR)
    if data_path.exists():
        print("Data folder named 'data' was found.")
    else:
        raise ValueError(f"Data folder, named: 'data' was not found!")

    # check if idx folder exists
    idx_path = Path(IDX_DIR)
    try:
        os.makedirs(idx_path)
    except FileExistsError:
        print("Scanning existing idx folder:")

    # scan folder for index in pairs, if there is no index we re-generate it
    subdirs = get_subdirectory_paths(DATA_DIR)
    for subdir in subdirs:
        # check index for folder name
        idx_file_name = Path(subdir).name
        idx_file_with_extension = idx_file_name + ".index"
        idx_file_path = (Path(IDX_DIR).joinpath(idx_file_with_extension))

        if idx_file_path.exists():
            print("Skip generating index file. ( For re-generation please delete .index and .npy files!")
        else:
            print("Generating missing index file...")

            # get chunks from subdirectory MD files
            chunks = build_chunks(subdir)
            # create FAISS embeddings from chunks
            create_embeddings(IDX_DIR, chunks, idx_file_name)


# only test?
def process_inputs(answer_type, only_source, top_k, temperature, answer_length, model, text_inputs):
    # setup app config for global reach
    app_config["answer_type"] = str(answer_type)
    app_config["only_source"] = bool(only_source)
    app_config["top_k"] = int(top_k)
    app_config["temperature"] = float(temperature)
    app_config["model"] = str(model)
    app_config["answer_length"] = int(answer_length)

    return query_engine(text_inputs)

if __name__ == "__main__":
    # UI layout
    with gr.Blocks() as demo:
        gr.Markdown("# MK Knowledge Space")
        gr.Markdown("Use LLM on local books, paper and code.")
        with gr.Row():
            with gr.Column(scale=1, min_width=20):
                answer_type = gr.Dropdown(
                    [
                        "compact",
                        "teaching",
                        "bullet-point",
                    ], label="Answer type", info="Style of answers."
                )
                only_source = gr.Checkbox(value=True, label="Use only source")
                top_k = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="Search depth (1-10)")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, label="Imagination temperature")
                answer_length = gr.Slider(minimum=100, maximum=1000, step=100, value=400, label="Answer length")
                model = gr.Dropdown(
                    [
                        "gemma4:latest",
                        "gemma3:270m",
                        "smollm2:360m",
                        "smollm2:135m",
                        "qwen3.6:latest",
                        "codellama:7b-instruct",
                        "qwen3.5:0.8b",
                        "deepseek-coder:6.7b",
                        "qwen2.5-coder:14b",
                        "phi4-mini:latest",
                        "Llama3.2:1b",
                        "reaperdoesntrun/Qwen3-0.6B-Distilled:latest",
                        "deepseek-r1:7b",
                        "llama3.1:latest",
                        "qwen2.5-coder:7b"
                    ], label="Model", info="Choose LLM Model"
                )
                # Train database
                clear_button = gr.Button("Train Database")
                clear_button.click(
                    fn=create_search_idx,
                    inputs=None,
                    outputs=None
                )

            with gr.Column(scale=3):
                output_text = gr.Textbox(label="Answer", lines=25)
                text_inputs = gr.Textbox(label="Question", lines=5)
                text_inputs.submit(
                    fn=process_inputs,
                    inputs=[answer_type, only_source, top_k, temperature, answer_length, model, text_inputs],
                    outputs=output_text
                )
                with gr.Row():
                    # Clears the input field
                    clear_button = gr.Button("Clear")
                    clear_button.click(
                        fn=lambda: "",
                        inputs=None,
                        outputs=text_inputs
                    )

                    # Create a submit button to process inputs
                    submit_button = gr.Button("Submit")
                    submit_button.click(
                        fn=process_inputs,
                        inputs=[answer_type, only_source, top_k, temperature, answer_length, model, text_inputs],
                        outputs=output_text
                    )

    # Launch the app
    demo.launch()