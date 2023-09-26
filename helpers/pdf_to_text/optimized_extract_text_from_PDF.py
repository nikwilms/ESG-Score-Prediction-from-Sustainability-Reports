import os
import csv
import PyPDF2
import re

"""
    This file is optional to use. If the other file does not work well, we can use this one.
    Created on: 25.09.2023
"""
def pdf_to_text(pdf_path):
    """Convert PDF file to plain text using PyPDF2."""
    page_lines_list = []

    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        
        for page in pdf.pages:
            # Extract text from the page
            page_text = page.extract_text().strip()
            # Removing numbers and 'page' text from the page text
            page_text = re.sub(r'\d+|page', '', page_text, flags=re.IGNORECASE)
            # Split the text into lines
            page_lines_list.append([line.strip().lower() for line in page_text.split('\n') if line])

    return page_lines_list


def make_a_list_of_lists(page_lines_list):
    """Convert page lines to a list of lists."""
    return [[line.strip() for line in page] for page in page_lines_list]


def finding_header_and_footer(all_line_list):
    """Find header and footer."""
    page_starts = set(all_line_list[0])
    page_ends = set(all_line_list[-1])
    return list(page_starts & page_ends), list(page_ends - page_starts)

def cleaning_lists(all_line_list):
    """Clean empty strings from the list of lists."""
    return [[line for line in sublist if line] for sublist in all_line_list]


def remove_header_and_footer(cleaned_all_line_list, unique_header, unique_footer):
    """Remove header and footer."""
    return [[line for line in sublist if line not in unique_header + unique_footer] for sublist in cleaned_all_line_list]


def create_unique_list(cleaned_all_line_list):
    """Create a unique list for each page."""
    return [list(set(sublist)) for sublist in cleaned_all_line_list]

def detect_same_items(unique_all_line_list):
    """Detect same items in pages."""
    return list(set.intersection(*map(set, unique_all_line_list)))


def remove_same_items(cleaned_all_line_list, same_items):
    """Remove the same items from each page."""
    return [[line for line in sublist if line not in same_items] for sublist in cleaned_all_line_list]


def list_of_lists_to_text(cleaned_all_line_list):
    """Convert list of lists to text for text preprocessing."""
    return ' '.join(' '.join(sublist) for sublist in cleaned_all_line_list)


def pdf_to_text_pipeline(pdf_path):
    """Define a pipeline for pdf to text steps."""
    page_lines_list = pdf_to_text(pdf_path)
    all_line_list = make_a_list_of_lists(page_lines_list)
    cleaned_all_line_list = cleaning_lists(all_line_list)
    unique_header, unique_footer = finding_header_and_footer(cleaned_all_line_list)
    cleaned_all_line_list = remove_header_and_footer(cleaned_all_line_list, unique_header, unique_footer)
    unique_all_line_list = create_unique_list(cleaned_all_line_list)
    same_items = detect_same_items(unique_all_line_list)
    cleaned_all_line_list = remove_same_items(cleaned_all_line_list, same_items)
    return list_of_lists_to_text(cleaned_all_line_list)

# Optimized processing function
def process_pdfs_in_directory(directory, output_csv):
    """Process all PDFs in a directory and save the content to a CSV file."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'ticker', 'year', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in os.listdir(directory):
            if file.endswith('.pdf'):
                print(f"Processing {file}...")
                filepath = os.path.join(directory, file)
                content = pdf_to_text_pipeline(filepath)
                if content:  # Only write if content exists
                    _, ticker, year = file.rstrip('.pdf').split('_')
                    writer.writerow({'filename': file, 'ticker': ticker, 'year': year, 'content': content})
