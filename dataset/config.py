

def get_dicts(dict_path):
    data = open(dict_path).readlines()
    word2index_dicts = {}
    index2word_dicts = {}
    for i,d in enumerate(data):
        word2index_dicts[d.replace("\n","")]=i
        index2word_dicts[i]=d.replace("\n","")

    return word2index_dicts,index2word_dicts
