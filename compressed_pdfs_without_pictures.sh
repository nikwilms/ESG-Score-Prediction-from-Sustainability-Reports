#!/bin/bash

# Get a list of all the PDF files in the current directory
pdf_files=$(ls *.pdf)

# Create a new directory to store the text-only PDF files
text_only_pdf_dir="text_only_pdfs"
mkdir -p $text_only_pdf_dir

# Loop through all the PDF files and remove vector graphics (which includes backgrounds)
for pdf_file in $pdf_files; do

  # Get the output filename for the text-only PDF file
  text_only_pdf=$text_only_pdf_dir/$pdf_file

  # Remove vector graphics from the PDF file
  gs -o $text_only_pdf -sDEVICE=pdfwrite -dFILTERVECTOR -dFILTERIMAGE $pdf_file

done

# Print a message to let the user know that the PDFs have been processed
echo "All non-text elements have been removed, and the results are saved to the 'text_only_pdfs' directory."
