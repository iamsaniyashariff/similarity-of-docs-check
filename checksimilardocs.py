import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def extract_text_from_pdf(pdf_path):
   
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def calculate_similarity(doc1, doc2):
  
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def select_files():
  
    Tk().withdraw()  
    pdf_path1 = askopenfilename(title="Select the first PDF file", filetypes=[("PDF files", "*.pdf")])
    pdf_path2 = askopenfilename(title="Select the second PDF file", filetypes=[("PDF files", "*.pdf")])
    return pdf_path1, pdf_path2

def main():

    pdf_path1, pdf_path2 = select_files()

    if not pdf_path1 or not pdf_path2:
        print("Both PDF files must be selected.")
        return


    text1 = extract_text_from_pdf(pdf_path1)
    text2 = extract_text_from_pdf(pdf_path2)


    similarity_score = calculate_similarity(text1, text2)


    print(f"Cosine Similarity: {similarity_score}")

if __name__ == "__main__":
    main()
