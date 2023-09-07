# -*- coding: utf-8 -*-

#   *** Do not import any library except already imported libraries ***
from util import *
from mlp import MLP
from autoencoder import AutoEncoder
#   *** Do not import any library except already imported libraries ***

class Preprocessing(AI_util):
    '''
        words in the document -> 1
        words not in the document -> 0
    '''
    def Calculate_Binary(self, data: List[Tuple[str, List[str], int]])  -> List[Tuple[str, List[float], int]]:
        ############################################## EDIT HERE ###########################################
        one_hot_data = list()
        for id, para, labelidx in data:
            one_hot_para = [0] * len(self.word2idx)
            for word, index in self.word2idx.items():
                if(word in para):
                    one_hot_para[index] = 1
            one_hot_data.append((id, one_hot_para, labelidx))
        
        return one_hot_data
        ################################################# END ##############################################


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
        K = 5
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
            learning_rate=0.006
        )

        d = list(zip(self.train_input, self.train_label))
        for epoch in trange(30, desc='Epoch...'):
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


class Process_AutoEncoder():
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

        ############################################## EDIT HERE ###########################################
        self.ae = AutoEncoder(
            input_size=len(self.train_input[0]), 
            hidden_size=1000,
            latent_size=1000,
            output_size=len(self.train_input[0]),
            learning_rate=0.006
        )
        ################################################# END ##############################################

        self.test = test
        self.train = train

    def autoencoder_train(self):
        d = list(zip(self.train_input, self.train_label))
        for epoch in trange(20, desc='Epoch...'):
            epoch_loss = .0
            random.shuffle(d)
            for train_vec, label in tqdm(d, desc='Train iter'):
                logits = self.ae.forward(train_vec)
                self.ae.backward()
                self.ae.step()

            print('Epoch : {}'.format(epoch+1))

    def get_compressed_representations(self):
        train_latent_vectors = list()
        test_latent_vectors = list()

        d = list(zip(self.train_input, self.train_label))
        for idx, (train_vec, label) in tqdm(enumerate(d), desc='Train iter'):
            logits = self.ae.forward(train_vec)
            latent = self.ae.latent.squeeze().tolist()
            train_latent_vectors.append((self.train[idx][0], latent, self.train[idx][-1]))

        d = list(zip(self.test_input, self.test_label))
        for idx, (test_vec, label) in tqdm(enumerate(d), desc='Test iter'):
            logits = self.ae.forward(test_vec)
            latent = self.ae.latent.squeeze().tolist()
            test_latent_vectors.append((self.test[idx][0], latent, self.test[idx][-1]))

        return train_latent_vectors[:], test_latent_vectors[:]


def main(data, label2idx, std_name, std_id):
    """
    1. Train the machine learning models (e.g., KNN, SVM, MLP) using the normalized TF-IDF vectors
        - Training and evaluation code is already implemented
        - Use the appropriate fuctions to train the models

    2. Store the evaluation results
    """
    result = dict()

    result['knn'] = None
    result['svm'] = None
    result['mlp'] = None
    
    train_models = TrainModels(data, label2idx)

    knn_result = train_models.knn_classifier()
    result['knn'] = knn_result

    svm_result = train_models.svm_classifier()
    result['svm'] = svm_result

    mlp_result = train_models.mlp_classifier()
    result['mlp'] = mlp_result

    save_test_result(result, std_name=std_name, std_id=std_id)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--paragraph_id",
                        default=834,
                        type=int)
    args = parser.parse_args()

    ############################################## EDIT HERE ###########################################
    '''
    *** IMPORTANT ***
    Please write your own name and student ID.
    '''
    NAME = "Leechangoo"
    ID = "2019312601"
    ################################################# END ##############################################

    Preprocessing = Preprocessing()
    train_data = Preprocessing.load_data(data_path='Data/train.json', data_type='train')
    Preprocessing.train_binary = Preprocessing.Calculate_Binary(data=train_data)
    test_data = Preprocessing.load_data(data_path='Data/test.json', data_type='test')
    Preprocessing.test_binary = Preprocessing.Calculate_Binary(data=test_data)
    data = (Preprocessing.train_binary, Preprocessing.test_binary)

    autoencoder = Process_AutoEncoder(data, Preprocessing.label2idx)
    '''
    Train autoencoder and get the latent vectors (compressed representions)
    '''
    ############################################## EDIT HERE ###########################################
    autoencoder.autoencoder_train()
    data = autoencoder.get_compressed_representations()
    ################################################# END ##############################################

    main(data, Preprocessing.label2idx, std_name=NAME, std_id =ID)