import torch
import os
import logging
import time
import numpy as np
import spacy
from torchtext import data


logging.basicConfig(level=logging.INFO)


class LSTM_trainer():
    def __init__(self,
                model,
                optimizer,
                text,
                label,
                loss,
                max_epochs,
                identifier,
                train_verbosity,
                batch_verbosity,
                batch_tqdm,
                save_dir,
                nlp='en_core_web_sm'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.max_epochs = max_epochs
        self.train_verbosity = train_verbosity
        self.batch_verbosity = batch_verbosity
        self.batch_tqdm = batch_tqdm
        self.identifier = identifier
        self.save_dir = save_dir if save_dir else os.getcwd()
        self.save_path = f"{self.save_dir}/{self.identifier}.pt"
        self.text = text
        self.label = label
        self.nlp = nlp
        self.best_val_acc = 0

    def compute_metrics(self, data_loader):   
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                text, text_lengths = batch_data.content
                logits = self.model(text, text_lengths)
                cost = self.loss_fn(logits, batch_data.label.long())
                _, predicted_labels = torch.max(logits, 1)
                num_examples += batch_data.label.size(0)
                correct_pred += (predicted_labels.long() == batch_data.label.long()).sum()
            return cost, correct_pred.float()/num_examples * 100

    def train(self, dataloaders):
    
        assert isinstance(dataloaders, dict)
        train_dl, val_dl, test_dl = dataloaders['train'], dataloaders['validation'], dataloaders['test']

        if self.train_verbosity:
            logging.info(f"GPU is available: {torch.cuda.is_available()}")
            logging.info(f"\nStarting training on {self.device}")
            logging.info(f"running for {self.max_epochs} epochs on {type(self.model).__name__} model")

        self.model.to(self.device)

        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()

            train_lossses, val_losses, val_accs = [], [], []

            for batch_idx, batch_data in enumerate(train_dl):
                text, text_lengths = batch_data.content
                
                self.model.train()
                logits = self.model(text, text_lengths)
                cost = self.loss_fn(logits, batch_data.label.long())
                train_lossses.append(cost.item())
                
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                val_cost, val_acc = self.compute_metrics(val_dl)
                val_losses.append(val_cost.item())
                val_accs.append(val_acc.item())

                if batch_idx % self.batch_verbosity == 0:
                    logging.info(f"Batch {batch_idx} for epoch {epoch + 1}:")
                    logging.info('-' * 25)
                    logging.info(f"Test loss: {cost:.4f}, Val loss: {val_cost:.4f}, Val acc: {val_acc:.3f}%\n")

            acc_avg = np.mean(val_accs)
            if acc_avg > self.best_val_acc:
                self.best_val_acc = acc_avg
                torch.save(self.model.state_dict(), self.save_path)

            if self.train_verbosity:
                logging.info(f"Epoch {epoch + 1} / {self.max_epochs}")
                logging.info(f"Time: {time.time() - epoch_start_time:.3f}s")
                logging.info(f"Avg Test loss: {np.mean(train_lossses):.4f}, Val loss: {np.mean(val_losses):.4f}, Val acc: {acc_avg:.3f}%")
                logging.info('-' * 25 + '\n')

        self.model.load_state_dict(torch.load(self.save_path))
        test_loss, test_acc = self.compute_metrics(test_dl)
        logging.info(f"Test loss: {test_loss.item():.4f}, Test acc: {test_acc.item():.2f}%")
        

    def predict(self, news_content, min_len=4, existing_vocab_path=None, model_checkpoint_pt=None):
    
        map_dictionary = {
            0: "World",
            1: "Sports",
            2: "Business",
            3:"Sci/Tech",
        }
        
        if model_checkpoint_pt is not None:
            self.model.load_state_dict(torch.load(model_checkpoint_pt))
        
        if existing_vocab_path is not None:
            txt = data.Field(sequential=True,
                            use_vocab=True,
                            tokenize=lambda text: [token.text for token in nlp(text)],
                            include_lengths=True)
            text_vocab = torch.load(existing_vocab_path)
            txt.vocab = text_vocab
        else:
            txt = self.text
            
        self.model.eval()
        self.model.to(self.device)
        nlp = spacy.load(self.nlp)
        tokenized = [tok.text for tok in nlp.tokenizer(news_content)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [txt.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(dim=1)
        length_tensor = torch.LongTensor(length).to(self.device)
        preds = self.model(tensor, length_tensor)
        preds = torch.softmax(preds, dim=1)
        
        proba, class_label = preds.max(dim=1)
        return proba.item(), map_dictionary[class_label.item()]

