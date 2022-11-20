# encoding=utf8
import numpy as np
import paddle
import paddle.nn.functional as F
import random

import utils
from sentimentClassifier import SentimentClassifier


# 切词 + 全部转为小写
def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")

        data_set.append((sentence, sentence_label))

    return data_set


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                        else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
    if not drop_last and len(sentence_batch) > 0:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


def train(model):
    # 开启模型训练模式
    model.train()

    # 建立训练数据生成器，每次迭代生成一个batch，每个batch包含训练文本和文本对应的情感标签
    train_loader = build_batch(word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len)

    for step, (sentences, labels) in enumerate(train_loader):
        # 获取数据，并将张量转换为Tensor类型
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(labels)

        # 前向计算，将数据feed进模型，并得到预测的情感标签和损失
        logits = model(sentences)

        # 计算损失
        loss = F.cross_entropy(input=logits, label=labels, soft_label=False)
        loss = paddle.mean(loss)

        # 后向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清除梯度
        optimizer.clear_grad()

        if step % 100 == 0:
            # 记录当前步骤的loss变化情况
            losses.append(loss.numpy()[0])
            steps.append(step)
            # 打印当前loss数值
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))


def evaluate(model):
    # 开启模型测试模式，在该模式下，网络不会进行梯度更新
    model.eval()

    # 定义以上几个统计指标
    tp, tn, fp, fn = 0, 0, 0, 0

    # 构造测试数据生成器
    test_loader = build_batch(word2id_dict, test_corpus, batch_size, 1, max_seq_len)

    for sentences, labels in test_loader:
        # 将张量转换为Tensor类型
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(labels)

        # 获取模型对当前batch的输出结果
        logits = model(sentences)

        # 使用softmax进行归一化
        probs = F.softmax(logits)

        # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        probs = probs.numpy()
        for i in range(len(probs)):
            # 当样本是的真实标签是正例
            if labels[i][0] == 1:
                # 模型预测是正例
                if probs[i][1] > probs[i][0]:
                    tp += 1
                # 模型预测是负例
                else:
                    fn += 1
            # 当样本的真实标签是负例
            else:
                # 模型预测是正例
                if probs[i][1] > probs[i][0]:
                    fp += 1
                # 模型预测是负例
                else:
                    tn += 1

    # 整体准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 输出最终评估的模型效果
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
    print("Accuracy: %.4f" % accuracy)


if __name__ == '__main__':
    train_corpus = utils.load_imdb(True)
    test_corpus = utils.load_imdb(False)

    train_corpus = data_preprocess(train_corpus)
    test_corpus = data_preprocess(test_corpus)

    word2id_freq, word2id_dict = build_dict(train_corpus)
    vocab_size = len(word2id_freq)

    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)

    for batch_id, batch in enumerate(
            build_batch(word2id_dict, train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
        print(batch)
        break

    paddle.seed(0)
    random.seed(0)
    np.random.seed(0)

    # 定义训练参数
    epoch_num = 5
    batch_size = 128

    learning_rate = 0.0001
    dropout_rate = 0.2
    num_layers = 1
    hidden_size = 256
    embedding_size = 256
    max_seq_len = 128
    vocab_size = len(word2id_freq)

    # 实例化模型
    sentiment_classifier = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)

    # 指定优化策略，更新模型参数
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                      parameters=sentiment_classifier.parameters())

    # 定义训练函数
    # 记录训练过程中的损失变化情况，可用于后续画图查看训练情况
    losses = []
    steps = []

    # 训练模型
    train(sentiment_classifier)

    # 保存模型，包含两部分：模型参数和优化器参数
    model_name = "sentiment_classifier"
    # 保存训练好的模型参数
    paddle.save(sentiment_classifier.state_dict(), "{}.pdparams".format(model_name))
    # 保存优化器参数，方便后续模型继续训练
    paddle.save(optimizer.state_dict(), "{}.pdopt".format(model_name))

    # 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
    saved_state = paddle.load("./sentiment_classifier.pdparams")
    sentiment_classifier = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)
    sentiment_classifier.load_dict(saved_state)

    # 评估模型
    evaluate(sentiment_classifier)
