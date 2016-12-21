import tools
class NaiveBayes(object):
    """
    Naive bayes
    """
    def createVocabList(self, data_set):
        vocab_set = set()
        for doc in data_set:
            vocab_set = vocab_set | set(doc)
        return list(vocab_set)

    def setOfWord2Vec(self, vocab_list, input_set):
        return_vec = [0] * len(vocab_list)
        for word in input_set:
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1
        return return_vec



def test():
    list_posts, list_classes = tools.loadDataSet()
    vocab = NaiveBayes().createVocabList(list_posts)
