import os
import re
import chardet
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import json

stop_words = set(stopwords.words('english'))

def save_data(data,file_path):
    with open(file_path,'w') as file:
        json.dump(data,file)

def load_data(file_path):
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

def csr_matrix_to_dict(matrix):
    return {
        "data": matrix.data.tolist(),
        "indices": matrix.indices.tolist(),
        "indptr": matrix.indptr.tolist(),
        "shape": matrix.shape
    }

def save_csr(matrix,file_path):
    matrix_dict = csr_matrix_to_dict(matrix)
    with open(file_path,'w') as file:
        json.dump(matrix_dict,file)

def load_csr(file_path):
    with open(file_path, 'r') as file:
        matrix_dict = json.load(file)
    
    data = matrix_dict["data"]
    indices = matrix_dict["indices"]
    indptr = matrix_dict["indptr"]
    shape = tuple(matrix_dict["shape"])

    return csr_matrix((data, indices, indptr), shape=shape)

def find_popular_ones(folder_path):

    keywords = load_data("keywords_test.json")
    
    document_frequency = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        word_frequency = {keyword: 0 for keyword in keywords}

        with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result['encoding']
            
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                cleaned_text = re.sub(r'[";:,_!?.{}()\[\]-]', '', text)
                words = cleaned_text.split()

                for word in words:
                    word = word.lower()
                    word = word.replace('_', '')
                    if word in keywords:
                        word_frequency[word] += 1

            sorted_word_frequency = sorted(word_frequency.items(),key=lambda x: x[1], reverse=True)


            document_frequency += sorted_word_frequency[:5]

        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}. Skipping file: {file_path}")
            continue

    save_data(document_frequency,"word_freq_test.json")



def create_keywords_set(folder_path):
    keywords_set = set()
    

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if file_path.endswith('.txt'):
            
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result['encoding']
            
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    words = re.findall(r'\b\w+\b', text.lower())

                    for word in words:
                        word = word.replace('_', '')
                        if word not in stop_words and word not in keywords_set:
                            keywords_set.add(word)

            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError: {e}. Skipping file: {file_path}")
                continue

    save_data(list(keywords_set),"keywords_test.json")


def create_bag_of_words(file_path,keywords):
    word_vector = {keyword: 0 for keyword in keywords}

    with open(file_path, 'rb') as file:
        raw_data = file.read()
        encoding_result = chardet.detect(raw_data)
        encoding = encoding_result['encoding']


    try:
        with open(file_path,'r',encoding=encoding) as file:
            text = file.read()

            words = re.findall(r'\b\w+\b', text.lower())

            for word in words:
                word = word.replace('_', '')
                if word in keywords:
                    word_vector[word] += 1

    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}. Skipping file: {file_path}")

    return list(word_vector.values())

def create_term_by_document_matrix(folder_path,with_idf = True):
    keywords = load_data("keywords_test.json")
    all_vectors = []

    number_of_documents = 0
    document_frequency = {keyword: 0 for keyword in keywords}

    files = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path,filename)
        files.append(filename)
        d_i = create_bag_of_words(file_path,keywords)
        all_vectors.append(d_i)

        
        if number_of_documents%10 == 0:
            print(f"Current file: {file_path}, iteration: {number_of_documents}")

        number_of_documents += 1
        if with_idf:
            for i,value in enumerate(d_i):
                if value > 0:
                    keyword = keywords[i]
                    document_frequency[keyword] += 1

    if with_idf:            
        idf = {}

        for keyword,freq in document_frequency.items():
            idf[keyword] = np.log(number_of_documents/freq)

        for i in range(len(all_vectors)):
            for j in range(len(all_vectors[i])):
                if all_vectors[i][j] > 0:
                    all_vectors[i][j] *= idf[keywords[j]]

    save_data(files,"files_test.json")
    save_csr(csr_matrix(all_vectors),"csr_matrix_test_idf.json")
    return all_vectors

def svd_on_matrix(k):
    A = load_csr("csr_matrix_test_idf.json")
    U,S,VT = svds(A,k)

    S = np.diag(S[:k])

    svd_matrix = np.dot(U,np.dot(S,VT))  
    save_csr(csr_matrix(svd_matrix),"csr_matrix_svd_test.json")


folder_path = "books_test"

create_keywords_set(folder_path)
vectors = create_term_by_document_matrix(folder_path,False)
vectors = create_term_by_document_matrix(folder_path)
svd_on_matrix(4)










