import subprocess

def convert_md_to_pdf(md_file_path, pdf_file_path):
    # call pandoc using subprocess module
    subprocess.run(['pandoc', md_file_path, '-o', pdf_file_path])

# usage example
convert_md_to_pdf('stock_market/README.md', 'stock_market/Report.pdf')
