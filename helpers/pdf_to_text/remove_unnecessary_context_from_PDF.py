import os
import csv
import PyPDF2
import re

def pdf_to_text(pdf_path):
    """Convert PDF file to plain text using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            # Crate a empty list to store the lines of each page
            page_lines_list = []
            # Iterate through each page in the PDF
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
        
                # Extract text from the page
                page_text = page.extract_text()
        
                # Split the text into lines
                lines = page_text.split('\n')
        
                # Extract the whole lines to the txt file
                # Create a empty list to store the lines in the every single page
                lines_list = []
                for line in lines:
                    if line != '':
                        # Strip the lines
                        line = line.strip()
                        # Removing numbers
                        line = re.sub(r'\d+', '', line)
                        # Lowercase the lines
                        line = line.lower()
                        # Remove just 'page' text
                        line = re.sub(r'page', '', line)
                        # Append the lines to the list
                        lines_list.append(line)

        
                page_lines_list.append(lines_list)
            # text = ''.join(page.extract_text() for page in pdf.pages)
            return page_lines_list
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Creating a function to make a list of lists for each page
def make_a_list_of_lists(page_lines_list):
    """ Description of function
    This function takes a list of lists and returns stripped a list of lists for each page

    Args:
        page_lines_list (list): Takes a list of lists for each page

    Returns:
        list : Returns a stripped list of lists for each page
    """
    all_line_list = []
    for page_lines in page_lines_list:
        line_list = []
        for lines in page_lines:
            # Strip the all lines
            lines = lines.strip()
            line_list.append(lines)

        all_line_list.append(line_list)
    return all_line_list

def finding_header_and_footer(all_line_list):
    """ Description of function
    This function help to finding header and footer in the each pdf page

    Args:
        all_line_list (list): Takes a list of lists for each page

    Returns:
        lists : Returns a list of finding header and footer
    """
    header = []
    for i in range(len(all_line_list) - 1):
        # Control the first element of the each pdf page and the first element of next pdf page
        if all_line_list[i][0] == all_line_list[i+1][0]:
            # If they are equal, append the first element of the each pdf page to the header list
            header.append(all_line_list[i][0])
    # Create a unique list of header
    unique_header = list(set(header))

    footer = []
    for i in range(len(all_line_list) - 1):
        # Control the last element of the each pdf page and the last element of next pdf page
        if all_line_list[i][-1] == all_line_list[i+1][-1]:
            # If they are equal, append the last element of the each pdf page to the footer list
            footer.append(all_line_list[i][-1])
    # Create a unique list of footer
    unique_footer = list(set(footer))
    
    return unique_header, unique_footer


def cleaning_lists(all_line_list):
    ''' 
    This function removes empty strings from the list of lists
    Args:
        all_line_list (list): Takes a list of lists for each page
    Returns:
        list : Returns a cleaned list of lists for each page
    '''
    cleaned_all_line_list = []
    for sublist in all_line_list:
        # Remove empty strings from the list
        sublist = list(filter(lambda x: x != '', sublist))
        cleaned_all_line_list.append(sublist)
    return cleaned_all_line_list


def remove_header_and_footer(cleaned_all_line_list, unique_header, unique_footer):
    '''
    This function removes header and footer from the list of lists
    Args:
        cleaned_all_line_list (list): Takes a cleaned list of lists for each page
        unique_header (list): Takes a unique list of header
        unique_footer (list): Takes a unique list of footer
    Returns:
        list : Returns a cleaned list of lists for each page
    '''
    for sublist in cleaned_all_line_list:
        for item in unique_header:
            # Control the headers in the lists
            if item in sublist:
                sublist.remove(item)
        for item in unique_footer:
            # Control the footers in the lists
            if item in sublist:
                sublist.remove(item)
    return cleaned_all_line_list


def create_unique_list(cleaned_all_line_list):
    '''
    This function creates a unique list for each page (like a specific dictionary for each page)
    This list helps to detect the same items in the different pages
    Args:
        cleaned_all_line_list (list): Takes a cleaned list of lists for each page
    Returns:
        list : Returns a unique list of lists for each page
    '''
    unique_all_line_list = []
    for sublist in cleaned_all_line_list:
        unique_elements = list(set(sublist))
        unique_all_line_list.append(unique_elements)
    
    return unique_all_line_list

def detect_same_items(unique_all_line_list):
    '''
    This function detects if there are the same items in a pdf page and returns a list of the same items
    Args:
        unique_all_line_list (list): Takes a unique list of lists for each page
    Returns:
        list : Returns a list of the same items
    '''
    same_items = []
    for i in range(len(unique_all_line_list) - 1):
        for item in unique_all_line_list[i]:
            if item in unique_all_line_list[i+1]:
                same_items.append(item)

    same_items = list(set(same_items))

    return same_items

def remove_same_items(cleaned_all_line_list, same_items):
    ''' 
    This function removes the same items from each pdf page
    Args:
        cleaned_all_line_list (list): Takes a cleaned list of lists for each page
        same_items (list): Takes a list of the same items
    Returns:
        list : Returns a cleaned list of lists for each page
    '''
    for sublist in cleaned_all_line_list:
        for item in same_items:
            if item in sublist:
                sublist.remove(item)
    
    return cleaned_all_line_list

# Creating a function to convert list of lists to a text
def list_of_lists_to_text(cleaned_all_line_list):
    '''
    This function converts list of lists to a text for text preprocessing
    Args:
        cleaned_all_line_list (list): Takes a cleaned list of lists for each page
    Returns:
        string : Returns a text'''
    text = ''
    for sublist in cleaned_all_line_list:
        for item in sublist:
            text += item + ' '
    return text

# Defining Pipeline for pdf to text
def pdf_to_text_pipeline(pdf_path):
    '''
    This function defines a pipeline for pdf to text steps
    Args:
        pdf_path (string): Takes a pdf path
    Returns:
        string : Returns a text
    '''
    page_lines_list = pdf_to_text(pdf_path)
    all_line_list = make_a_list_of_lists(page_lines_list)
    cleaned_all_line_list = cleaning_lists(all_line_list)
    unique_header, unique_footer = finding_header_and_footer(cleaned_all_line_list)
    cleaned_all_line_list = remove_header_and_footer(cleaned_all_line_list, unique_header, unique_footer)
    unique_all_line_list = create_unique_list(cleaned_all_line_list)
    same_items = detect_same_items(unique_all_line_list)
    cleaned_all_line_list = remove_same_items(cleaned_all_line_list, same_items)
    text = list_of_lists_to_text(cleaned_all_line_list)
    return text



def process_pdfs_in_directory(directory, output_csv):
    """Process all PDFs in a directory and save the content to a CSV file."""

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile: #change 'w' to 'a' to append
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

process_pdfs_in_directory('/Users/neuefische/Downloads/test_ESG/text_only_pdfs', '../../data/extracted_text_sustainability_reports.csv')