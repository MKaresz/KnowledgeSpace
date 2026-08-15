'''
This file should collect and mass convert every pdf and epub source files to LLM readable Markdown files
'''

import os
from pathlib import Path
import pdfplumber
import ebooklib
from ebooklib import epub
#from bs4 import BeautifulSoup
import html2text


# TODO: add folder path as an input parameter not hardcoded


def get_subdirectory_paths(root_dir):
    subdirectory_paths = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            subdirectory_paths.append(item_path)
            # Recursively add subdirectories
            subdirectory_paths.extend(get_subdirectory_paths(item_path))

    return subdirectory_paths


def convert_pdf_to_md(pdf_path, md_path):
    with pdfplumber.open(pdf_path) as pdf:
        markdown_lines = []

        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                markdown_lines.append(text)

            # Extract tables (if any) and convert to Markdown tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    # Simple conversion to Markdown table format
                    header = table
                    rows = table[1:]
                    md_table = []

                    # Format header
                    md_table.append("| " + " | ".join(str(h) for h in header) + " |")
                    md_table.append("| " + "--- |" * len(header))

                    # Format rows
                    for row in rows:
                        md_table.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")

                    markdown_lines.append("\n".join(md_table))

        # Write to file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(markdown_lines))


def scan_folder_for_pdf(current_dir):
    """
    Scans the current local folder for .pdf files and converts them to markdown one by one.
    """
    # Configuration

    pdf_pattern = "*.pdf"

    # Find all PDF files in the current directory (non-recursive)
    # Use recursive=True if you want to search subfolders too
    pdf_files = list(current_dir.glob(pdf_pattern))

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process.\n")

    for pdf_file in pdf_files:
        # Define output path: same name, .md extension
        md_filename = pdf_file.stem + ".md"
        md_filepath = pdf_file.with_name(md_filename)
        #md_filepath = output_dir.joinpath("..", pdf_file.with_name(md_filename))

        # Call the conversion function
        print("Converting: " + str(md_filepath))
        convert_pdf_to_md(str(pdf_file), str(md_filepath))

    print("\n🎉 Batch conversion of PDF was complete.")


def epub_to_markdown(epub_path, output_path):
    book = epub.read_epub(epub_path)
    h = html2text.HTML2Text()
    h.body_width = 0  # Disable line wrapping
    h.ignore_links = False
    h.ignore_images = False

    markdown_content = ""

    # Iterate through items in the EPUB
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if isinstance(item, epub.EpubHtml):
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Convert to Markdown
            md_text = h.handle(str(soup))
            markdown_content += md_text + "\n\n---\n\n"

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"Converted {epub_path} to {output_path}")


def scan_folder_for_epub(current_dir):
    """
    Scans the current local folder for .epub files and converts them to markdown one by one.
    """
    # Configuration

    epub_pattern = "*.epub"

    # Find all PDF files in the current directory (non-recursive)
    # Use recursive=True if you want to search subfolders too
    epub_files = list(current_dir.glob(epub_pattern))

    if not epub_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(epub_files)} EPUB file(s) to process.\n")

    for pdf_file in epub_files:
        # Define output path: same name, .md extension
        md_filename = pdf_file.stem + ".md"
        md_filepath = pdf_file.with_name(md_filename)

        # Call the conversion function
        print("Converting: " + str(md_filepath))
        epub_to_markdown(str(pdf_file), str(md_filepath))

    print("\n🎉 Batch conversion of EPUB was complete.")



if __name__ == "__main__":
    subdirs = get_subdirectory_paths("E:\\KnowledgeSpace\\data")
    for subdir in subdirs:
        print(subdir)
        scan_folder_for_pdf(Path(subdir))
        scan_folder_for_epub(Path(subdir))





    # convert only one
    #convert_pdf_to_md("./in_pdf_epub/Deep Learning.pdf", "./in_pdf_epub/Deep Learning.md")



