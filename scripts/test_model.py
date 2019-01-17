import os
import torch
from fastNLP import Tester
from fastNLP import AccuracyMetric
from fastNLP.io.config_io import ConfigSection, ConfigLoader

from model import myESIM
import fastNLP.core.utils as util



args = ConfigSection()
ConfigLoader().load_config("../data/config.json", {"test": args})

# 加载训练集、测试集和词向量
print("\t* Loading train data...")
train_data = util.load_pickle(os.path.normpath(args["data_dir"]),args["train_file"])
print("\t* Loading test data...")
test_data = util.load_pickle(os.path.normpath(args["data_dir"]),args["test_file"])
print("\t* Loading word embeddings...")
embeddings = util.load_pickle(os.path.normpath(args["data_dir"]),args["embeddings_file"])
embeddings = torch.Tensor(embeddings)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = myESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 300,
                 embeddings=None,
                 dropout=0.5,
                 num_classes=3,
                 device=device).to(device)

print("\t* Testing on best model...")

checkpoint = args["model_file"]
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['model'])


# 测试模型在训练集上的准确率
print("\t* Testing model on the train dataset...")
tester_train = Tester(
    data=train_data,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=32,
)
tester_train.test()

# 测试模型在测试集上的准确率
print("\t* Testing model on the test dataset...")
tester = Tester(
    data=test_data,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=32,
)
tester.test()