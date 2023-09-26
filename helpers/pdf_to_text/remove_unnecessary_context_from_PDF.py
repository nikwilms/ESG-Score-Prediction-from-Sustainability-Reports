import os
import csv
import PyPDF2
import re
from collections import Counter


def pdf_to_text(pdf_path):
    """
    Convert a PDF file to a list of lists containing lines of text from each page.

    Args:
    - pdf_path (str): The path to the PDF file.

    Returns:
    - list: A list of lists containing lines of text from each page. Returns None if an exception occurs.
    """
    try:
        with open(pdf_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            page_lines_list = []
            for page in pdf.pages:
                page_text = page.extract_text()
                lines = [
                    re.sub(r"[\d]+|page", "", line.strip().lower())
                    for line in page_text.split("\n")
                    if line.strip()
                ]
                page_lines_list.append(lines)
            return page_lines_list
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def find_repeating_elements(all_line_list):
    """
    Find repeating elements across all pages of a PDF.

    Args:
    - all_line_list (list): A list of lists containing lines of text from each page.

    Returns:
    - set: A set containing elements that appear more than once.
    """
    flat_list = [elem for sublist in all_line_list for elem in sublist]
    counts = Counter(flat_list)
    repeating_elements = {key for key, val in counts.items() if val > 1}
    return repeating_elements


def remove_elements_from_list(all_line_list, elements_to_remove):
    """
    Remove specified elements from all pages of a PDF.

    Args:
    - all_line_list (list): A list of lists containing lines of text from each page.
    - elements_to_remove (set): A set containing elements to be removed.

    Returns:
    - list: A list of lists after removing the specified elements.
    """
    return [
        [item for item in sublist if item not in elements_to_remove]
        for sublist in all_line_list
    ]


def pdf_to_text_pipeline(pdf_path):
    """
    Pipeline to convert a PDF to a single text string after cleaning.

    Args:
    - pdf_path (str): The path to the PDF file.

    Returns:
    - str: A string containing all text after cleaning.
    """
    page_lines_list = pdf_to_text(pdf_path)
    repeating_elements = find_repeating_elements(page_lines_list)
    cleaned_all_line_list = remove_elements_from_list(
        page_lines_list, repeating_elements
    )
    text = " ".join([" ".join(sublist) for sublist in cleaned_all_line_list])
    return text


def process_pdfs_in_directory(directory, output_csv):
    """
    Process all PDF files in a directory and save the content to a CSV file.

    Args:
    - directory (str): The path to the directory containing PDF files.
    - output_csv (str): The path to the output CSV file.

    Returns:
    - None: Writes to the CSV file.
    """
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "ticker", "year", "content"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for file in os.listdir(directory):
            if file.endswith(".pdf"):
                print(f"Processing {file}...")
                filepath = os.path.join(directory, file)
                content = pdf_to_text_pipeline(filepath)
                if content:
                    _, ticker, year = file.rstrip(".pdf").split("_")
                    writer.writerow(
                        {
                            "filename": file,
                            "ticker": ticker,
                            "year": year,
                            "content": content,
                        }
                    )
