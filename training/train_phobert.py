import os, sys
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
import numpy as np
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from DataUtils.phobert_dataset import PhoBERTNERDataset, phobert_collate_fn
from DataUtils.NER_dataset import Vocab  # bạn giữ nguyên phần này để build tag vocab
from model.PhoBERT import PhoBERT_CRF

logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    losses = []

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} - Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Tạo mask cho CRF, True ở vị trí không phải padding (-100)
        crf_mask = labels != -100
        crf_mask[:, 0] = True  # token đầu tiên luôn True

        # Thay -100 thành 0 (thẻ O) để tránh lỗi indexing
        labels = labels.clone()
        labels[labels == -100] = 0

        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, mask=crf_mask)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return sum(losses) / len(losses)

def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            preds = model(input_ids=input_ids, attention_mask=attention_mask)

            for p, l, m in zip(preds, labels, attention_mask):
                valid_len = m.sum().item()
                y_pred.extend(p[:valid_len])
                y_true.extend(l[:valid_len].tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != -100
    return f1_score(y_true[mask], y_pred[mask], average="macro")

if __name__ == "__main__":
    data_dir = "/kaggle/input/ner-embedding-crf"
    output_dir = "/kaggle/working"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_vifinner.jsonl")
    dev_path = os.path.join(data_dir, "dev_vifinner.jsonl")
    test_path = os.path.join(data_dir, "test_vifinner.jsonl")

    logging.info(f"Device: {device}")

    vocab = Vocab(train_path)
    num_tags = vocab.num_tags

    train_dataset = PhoBERTNERDataset(train_path, vocab.tag2idx)
    dev_dataset = PhoBERTNERDataset(dev_path, vocab.tag2idx)
    test_dataset = PhoBERTNERDataset(test_path, vocab.tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=phobert_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=phobert_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=phobert_collate_fn)

    model = PhoBERT_CRF(num_tags=num_tags).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    best_f1 = 0
    patience = 0
    patience_limit = 5
    max_epochs = 30

    logging.info("Start training ...")
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        dev_f1 = evaluate(model, dev_loader)

        logging.info(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "phobert_crf_best.pt"))
            logging.info(f"New best model saved (F1={best_f1:.4f})")
        else:
            patience += 1
            logging.info(f"No improvement ({patience}/{patience_limit})")
            if patience >= patience_limit:
                logging.info("Early stopping.")
                break

    # Load best model và test
    model.load_state_dict(torch.load(os.path.join(output_dir, "phobert_crf_best.pt")))
    test_f1 = evaluate(model, test_loader)
    logging.info(f"Final TEST F1: {test_f1:.4f}")
