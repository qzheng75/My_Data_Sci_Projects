from helpers import *
from models import RNN
from trainer import LSTM_trainer
import torch.nn.functional as F


train_data, validation_data, test_data, text, label = get_datasets(raw_data_path='./AG_data',
                                                                          use_processed=True,
                                                                          describe_dataset=True)
train_dl, validation_dl, test_dl = get_dataloaders(train_data, validation_data, test_data, device='cuda')
dataloaders = {'train': train_dl, 'validation': validation_dl, 'test': test_dl}


INPUT_DIM = len(text.vocab)
PAD_IDX = text.vocab.stoi[text.pad_token]

model = RNN(INPUT_DIM, PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = LSTM_trainer(model=model,
                optimizer=optimizer,
                text=text,
                label=label,
                loss=F.cross_entropy,
                max_epochs=50,
                identifier='lstm_model_50_epochs',
                train_verbosity=True,
                batch_verbosity=100,
                batch_tqdm=True,
                save_dir='./results')
                
trainer.train(dataloaders)
                
sentence = """
          LG announces CEO Jo Seong-jin will be replaced by Brian Kwon Dec. 1, amid 2020 
          leadership shakeup and LG smartphone division's 18th straight quarterly loss
          """
prob, res = trainer.predict(sentence)
print(prob, res)

