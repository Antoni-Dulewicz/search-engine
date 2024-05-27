import numpy as np
from create_keywords import load_data,load_csr


def create_q_vector(input_text,keywords):
    q = {keyword: 0 for keyword in keywords}
    
    words = input_text.lower().split()
    for word in words:
        if word in keywords:
            q[word] += 1

    return list(q.values())

def similarity_function(A,q,k):

    q = np.array(q)

    cos_list = []

    for j in range(A.shape[0]):
        d_j = A.getrow(j).toarray()[0]

        result = 0
        for x in range(len(q)):
            if q[x] != 0 and d_j[x] !=0:
                result += q[x]*d_j[x]
        
        cos_list.append((result/(np.linalg.norm(q)*np.linalg.norm(d_j)),j))

    return sorted(cos_list,reverse=True)[:k]



A = load_csr("csr_matrix_test_idf.json")
#A = load_csr("csr_matrix_svd_test.json")
keywords = load_data("keywords_test.json")
files = load_data("files_test.json")


input_text = input("Enter your search query: ")


q = create_q_vector(input_text,keywords)

coss = similarity_function(A,q,5)

for i in range(len(coss)):
    print("File name: " + str(files[coss[i][1]])+ ", similarity value: " +str(coss[i][0]))
