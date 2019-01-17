import os
import torch
from fastNLP import Trainer
from fastNLP import Tester
from fastNLP import CrossEntropyLoss
from fastNLP import Adam
from fastNLP import AccuracyMetric
from fastNLP.io.config_io import ConfigSection, ConfigLoader
import fastNLP.core.utils as util
from model import myESIM




args = ConfigSection()
ConfigLoader().load_config("../data/config.json", {"train": args})

# 加载训练、验证数据集和词向量
print("\t* Loading train data...")
train_data = util.load_pickle(os.path.normpath(args["data_dir"]),args["train_file"])
print("\t* Loading dev data...")
dev_data = util.load_pickle(os.path.normpath(args["data_dir"]),args["dev_file"])
print("\t* Loading word embeddings...")
embeddings = util.load_pickle(os.path.normpath(args["data_dir"]),args["embeddings_file"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = myESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 300,
                 embeddings=embeddings,
                 dropout=0.5,
                 num_classes=3,
                 device=device).to(device)

trainer = Trainer(
    train_data=train_data,
    model=model,
    loss=CrossEntropyLoss(pred='pred', target='label'),
    metrics=AccuracyMetric(),
    n_epochs=10,
    batch_size=32,
    print_every=-1,
    validate_every=-1,
    dev_data=dev_data,
    use_cuda=True,
    optimizer=Adam(lr=0.0004, weight_decay=0),
    check_code_level=-1,
    metric_key='acc',
    use_tqdm=False
)

trainer.train()
# 训练结束后model为dev的最佳模型，保存
torch.save(model.state_dict(), '../data/checkpoints/best_model.pkl')

tester = Tester(
    data=test_data,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=32,
)
tester.test()