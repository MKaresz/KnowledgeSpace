The project contains two scripts: convert_lib.py and knowledge_space.py

The project is a simple system written in Python that implements a Retrieval-Augmented Generation (RAG) pipeline using vector embeddings.

---
The knowledge_space.py acts as a specialized chatbot/document retrieval engine by grounding its answers in a proprietary knowledge base. It was done to help writing thesis work to provide an easy to use framework to talk to your books and papers. Because many books and articles were researched, the script is using a two step retrival, a a 3-5 index vector were stored about every book as a topic vector, and based on the questions only the relevant topics are searched. The most similar books are loaded and searched through for a specific answer for chunk by chunk.

---
covnert_lib.py performs the following sequential steps:
- Directory Traversal: It recursively scans the root directory specified at the start (the KnowledgeSpace folder). It identifies every subfolder within this structure.
- File Detection: Within each subfolder, it detects files ending with .pdf or .epub.

Conversion:
- PDF Handling: It uses libraries to read the PDF content and intelligently convert the text and basic formatting into Markdown.
- EPUB Handling: It processes the structure of the EPUB book file, extracting chapters and content, and converting them into Markdown format.
- Output Generation: For every input file processed, it creates a corresponding .md file in the same folder, containing the extracted Markdown content.