import numpy as np
import tqdm.notebook as tqdm

def write_sentences_to_file(s, output_file):
    f = open(output_file.split(".")[0] + ".txt", "w")
    for sentence in s:
        f.write(sentence + "\n")
    f.close()

def create_matrices(path, punctuation=True):
    filename = path.split("/")
    filename = filename[-1]
    res, s = create_adjacency_matrix(path, punctuation=punctuation)

    if punctuation:
        addendum = "_punct_"
    else:
        addendum = "_nopunct_"
    np.save(filename.split(".")[0]+"{}.npy".format(addendum), res)
    write_sentences_to_file(s, filename.split(".")[0]+"{}.txt".format(addendum))

def get_matrix(samples):
    r = []
    s = []
    for i, sample in enumerate(samples):
        n = len(sample)
        #print("\r{}/{}".format(i+1,n),end="")
        adjacency = np.zeros((n,n))
        sentence = " ".join([x[1] for x in sample])
        s.append(sentence)
        real_id = {search_id: nid for nid, (search_id, *_) in enumerate(sample)}
        #print(real_id)
        found = True
        for j, (id, word, parent_id) in enumerate(sample):
      
            if parent_id!= 0: # Root word
                # get real id for parent
                try:
                    nid = real_id[parent_id]
                except:
                    found = False
                    break
                adjacency[j][nid] = 1.0
        if found:
            r.append(adjacency)
    return r, s

def create_adjacency_matrix(filename, punctuation = True):
    f = open(filename,"r")
    data = f.readlines()
    f.close()
    samples = []
    sample = []
    for line in tqdm.tqdm(data):
        if line[0] == "#":
            continue
        line = line.rstrip()
        if line != "":
            line = line.split("\t")
            if "-" in line[0]:
                continue
            if "." in line[0]:
                continue
            if line[3] == 'PUNCT' and not punctuation:
                continue
            sample.append([int(line[0]),line[1], int(line[6])]) # Get word and parent id
        else:
            samples.append(sample)
            sample = []
        m, s = get_matrix(samples)
    return np.array(m, dtype=object), s
