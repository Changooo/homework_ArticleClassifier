"""Skeleton Code for PA1"""

# -*- coding: utf-8 -*-

#   *** Do not import any library except already imported libraries ***
from util import *
from mlp import MLP
#   *** Do not import any library except already imported libraries ***

class Preprocessing(AI_util):
    #   *** Do not modify the code below ***
    def Calculate_TF_IDF_Normalization(self, data: List[Tuple[str, List[str], int]], data_type: str = 'train')  -> List[Tuple[str, List[float], int]]:
        total_TF = list()
        normalized_tf_idf = list()
        document_freq = {i: 0 for i in range(len(self.word2idx))}
    #   *** Do not modify the code above ***
        # 1. **Train data** Document Frequency (df) Calculate And Sort with df value 
        ##### EDIT HERE #####
        if data_type == 'train':
            for _, paragraph, _ in data:
                for word, idx in self.word2idx.items():
                    if word in paragraph:
                        document_freq[idx] += 1
            sorted_word2idx_list = sorted(list(self.word2idx.items()), key=(lambda tup: document_freq[tup[1]]))
            #from here tfidf vector's order of words changed from alphabet to Df asc 
        ##### END #####

        
        # 2. **Train data** Erase self.word2idx Dictionary value not in top 15000
        ##### EDIT HERE #####       
            self.word2idx = {word: new_idx for new_idx, (word, idx) in enumerate(sorted_word2idx_list[0:15000])}
            s = sorted(list(self.word2idx.items()), key=(lambda tup: tup[0]))
            self.word2idx = {word: idx for (word, idx) in s}
        ##### END #####


        # 3. Term Frequency (tf) And Document Frequency (df) Calculate
        #   *** Do not modify the code below ***
        
        document_freq = {i: 0 for i in self.word2idx.values()}
        for _, paragraph, _ in tqdm(data, desc='tf calculate'):
            term_freq = list()
            for token, idx in self.word2idx.items():
                tf = paragraph.count(token)
                term_freq.append(tf)
                if tf != 0:
                    document_freq[idx] += 1
            total_TF.append(term_freq)
        #   *** Do not modify the code above ***

        # 4. **Train data** Inverse Document Frequencey (IDF) Calculate
        ##### EDIT HERE #####
        if data_type=='train':
            self.idf = list()
            for word_idx, df in document_freq.items():
                self.idf.append( math.log((len(data)/df),2) )
        ##### END #####
        
        # 5. Normalized TF-IDF Calculate
        ##### EDIT HERE #####
        for idx_of_para, tf_of_para in enumerate(total_TF):
            id_of_para = data[idx_of_para][0]
            labelidx_of_para = data[idx_of_para][2]
            raw_tf_idf = [tf*(self.idf[idx]) for idx, tf in enumerate(tf_of_para)]
            normalizing_factor = 0
            for tf_idf in raw_tf_idf:
                normalizing_factor += tf_idf**2
            normalizing_factor = math.sqrt(normalizing_factor)
            normalized_tf_idf.append((id_of_para, [0 if normalizing_factor==0 else round(tf_idf/normalizing_factor, 2) for tf_idf in raw_tf_idf], labelidx_of_para))
        ##### END #####

        return normalized_tf_idf[:]


#   *** Do not modify the code below ***
class TrainModels():
    def __init__(self, data, label2idx):
        train, test = data
        self.label2idx = label2idx
        self.train_input = list()
        self.train_label = list()
        for data in train:
            self.train_input.append(data[-2])
            self.train_label.append(data[-1])
        self.test_input = list()
        self.test_label = list()
        for data in test:
            self.test_input.append(data[-2])
            self.test_label.append(data[-1])   

    def calculate_score(self, preds):
        label_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        correct = 0

        for label, pred in zip(self.test_label, preds):
            if pred == label:
                correct += 1
                label_correct[label] += 1

        tmp_result = {
            'accuracy': ((correct / len(self.test_label)) * 100, len(self.test_label)),
        }
        for k, v in self.label2idx.items():
            precision = (label_correct[v] / preds.count(v)) if preds.count(v) != 0 else 0.0
            recall = (label_correct[v] / self.test_label.count(v)) if self.test_label.count(v) != 0 else 0.0
            f1 = (2 * (precision * recall) / (recall + precision)) if recall + precision != 0 else 0.0
            tmp_result[k] = (precision * 100, recall * 100, f1 * 100, self.test_label.count(v))

        micro_avg_pre = (sum(list(label_correct.values())) / len(self.test_label))
        micro_avg_rec = (sum(list(label_correct.values())) / len(self.test_label))
        micro_avg_f1 = 2 * (micro_avg_pre * micro_avg_rec) / (micro_avg_rec + micro_avg_pre)
        tmp_result['micro avg'] = (micro_avg_pre * 100, micro_avg_rec * 100, micro_avg_f1 * 100, len(self.test_label))
        return tmp_result

    def get_euclidean_dist(self, vec_a, vec_b):
        dist = math.sqrt(sum(pow(a-b, 2) for a, b in zip(vec_a, vec_b)))

        return dist

    def knn_classifier(self):
        """
            This function is for evaluating (testing) KNN Model.
            (Inference may take some time…)
        """
        print("\nPredicting with KNN…")
        K = 10
        knn_euclidean_preds = []
        for test_vec in self.test_input:
            knn_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            test_dist = []
            sort_dist = []
            for train_vec in self.train_input:
                test_dist.append(self.get_euclidean_dist(test_vec, train_vec))
            sort_dist = sorted(test_dist)
            sort_dist = sort_dist[:K]
            for dist in sort_dist:
                knn_predict[self.train_label[test_dist.index(dist)]] += 1
            knn_euclidean_preds.append(knn_predict.index(max(knn_predict)))
        knn_result = self.calculate_score(knn_euclidean_preds)
        print("Evaluating with KNN END!")
        
        return knn_result

    def svm_classifier(self):
        """
            This function is for training and evaluating (testing) SVM Model.
        """
        print("\nPredicting with SVM...")
        classifier = LinearSVC(C=1.0, max_iter=2000)
        classifier.fit(self.train_input, self.train_label)

        svm_preds = classifier.predict(self.test_input).tolist()
        svm_result = self.calculate_score(svm_preds)
        print("Training and evaluating with SVM END!") 

        return svm_result
    
    def mlp_classifier(self):
        """
            This function is for training and evaluating (testing) MLP Model.
            (Training may take some time...)
        """
        print("\nPredicting with MLP...")
        mlp = MLP(
            input_size=len(self.train_input[0]), 
            hidden_size=200,
            output_size=len(self.label2idx),
            learning_rate=0.05
        )

        d = list(zip(self.train_input, self.train_label))
        for epoch in trange(20, desc='Epoch...'):
            epoch_loss = .0
            random.shuffle(d)
            for train_vec, label in tqdm(d, desc='Train iter'):
                logits = mlp.forward(train_vec)
                loss = mlp.loss(label, logits)
                mlp.backward()
                mlp.step()

                epoch_loss += loss

            print('Epoch : {} | Loss : {:.4f}'.format(epoch+1, epoch_loss / len(d)))

        mlp_preds = list()
        for test_vec, label in zip(self.test_input, self.test_label):
            logits = mlp.forward(test_vec)
            max_label = np.argmax(logits)
            mlp_preds.append(max_label)
        mlp_result = self.calculate_score(mlp_preds)
        print("Training and evaluating with MLP END!")

        return mlp_result
    #   *** Do not modify the code above ***


def main(data, label2idx, std_name, std_id):
    # *** Do not modify the code below ***
    result = dict()

    result['knn'] = None
    result['svm'] = None
    result['mlp'] = None
    #   *** Do not modify the code above ***

    """
    1. Train the machine learning models (e.g., KNN, SVM, MLP) using the normalized TF-IDF vectors
        - Training and evaluation code is already implemented
        - Use the appropriate fuctions to train the models

    2. Store the evaluation results
    """
    ##### EDIT HERE #####
    test1 = TrainModels(data, label2idx)
    result['knn'] = test1.knn_classifier()
    result['svm'] = test1.svm_classifier()
    result['mlp'] = test1.mlp_classifier()
    ##### END #####

    save_test_result(result, std_name=std_name, std_id=std_id)



if __name__ == "__main__":
    #   *** Do not modify the code below ***
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--paragraph_id",
                        default=888,
                        type=int)
    args = parser.parse_args()
    #   *** Do not modify the code above ***


    ### IMPORTANT ###
    ### Please write your own name and student ID.
    NAME = "Leechangoo"
    ID = "2019312601"
    
    #   *** Do not modify the code below ***
    Preprocessing = Preprocessing()
    train_data = Preprocessing.load_data(data_path='Data/train.json', data_type='train')
    Preprocessing.train_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=train_data)
    test_data = Preprocessing.load_data(data_path='Data/test.json', data_type='test')
    Preprocessing.test_tfidf = Preprocessing.Calculate_TF_IDF_Normalization(data=test_data, data_type='test')

    Preprocessing.save_preprocess_result(std_name=NAME, std_id =ID, paragraph_id=args.paragraph_id)

    data = (Preprocessing.train_tfidf, Preprocessing.test_tfidf)
    main(data, Preprocessing.label2idx, std_name=NAME, std_id =ID)
    #   *** Do not modify the code above ***