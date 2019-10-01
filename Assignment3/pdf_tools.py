# author        : Esteban
# course        : CS-691 Data Mining
# name          : pdf_tools.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import PyPDF2


def extractInformation(pdf_path):
    content = ''
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfFileReader(f)
        information = pdf.getDocumentInfo()
        for i in range(pdf.getNumPages()):
            content += pdf.getPage(i).extractText()

    return information.title, information.author, content
