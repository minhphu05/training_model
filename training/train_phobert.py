import os, sys
from os import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm

from DataUtils.phobert_dataset import (
    PhoBERTNERDataset,
    phobert_collate_fn
)
from DataUtils.NER_dataset import Vocab
from model.PhoBERT import PhoBERT_CRF

# =====================
# SETUP
# =====================
logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# TRAIN
# =====================
def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    losses = []

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} - Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 1. TẠO MASK HỢP LỆ (Lọc -100: subwords, [CLS], [SEP], padding)
        crf_mask = (labels != -100) # True ở vị trí có nhãn thực

        # 2. XỬ LÝ LỖI CRF MASK: Buộc vị trí đầu tiên ([CLS]/<s>) phải là True.
        # Thư viện torchcrf yêu cầu token đầu tiên không được bị mask.
        crf_mask[:, 0] = True 
        
        # 3. THAY THẾ CÁC GIÁ TRỊ -100 BẰNG 0 (ID tag hợp lệ, thường là 'O')
        # Bắt buộc phải thay thế để tránh lỗi Index Out of Bounds trên CUDA/CRF.
        labels[labels == -100] = 0 

        optimizer.zero_grad()
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mask=crf_mask # Truyền mask đã được sửa
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


# =====================
# EVALUATE
# =====================
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for p, l, m in zip(preds, labels, attention_mask):
                valid_len = m.sum().item()
                y_pred.extend(p[:valid_len])
                y_true.extend(l[:valid_len].tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != -100
    f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
    return f1


# =====================
# MAIN (CHỖ BẠN HỎI)
# =====================
if __name__ == "__main__":

    # ---------- PATH ----------
    data_dir = "/kaggle/input/ner-embedding-crf"
    output_dir = "/kaggle/working/"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train_vifinner.jsonl")
    dev_path   = os.path.join(data_dir, "dev_vifinner.jsonl")
    test_path  = os.path.join(data_dir, "test_vifinner.jsonl")

    best_model_path = os.path.join(output_dir, "phobert_crf_best.pt")

    logging.info(f"Device: {device}")

    # ---------- VOCAB (CHỈ LẤY TAG) ----------
    logging.info("Building tag vocab ...")
    vocab = Vocab(train_path)
    num_tags = vocab.num_tags

    # ---------- DATASET ----------
    logging.info("Loading datasets ...")
    train_dataset = PhoBERTNERDataset(train_path, vocab.tag2idx)
    dev_dataset   = PhoBERTNERDataset(dev_path, vocab.tag2idx)
    test_dataset  = PhoBERTNERDataset(test_path, vocab.tag2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=phobert_collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=phobert_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=phobert_collate_fn
    )

    # ---------- MODEL ----------
    logging.info("Building PhoBERT-CRF model ...")
    model = PhoBERT_CRF(num_tags=num_tags).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    # ---------- TRAIN LOOP ----------
    best_f1 = 0.0
    patience = 0
    patience_limit = 5
    epoch = 0

    logging.info("Start training ...")

    while True:
        epoch += 1

        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch
        )
        f1 = evaluate(model, dev_loader)

        logging.info(
            f"Epoch {epoch} | Train loss: {train_loss:.4f} | Dev F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved (F1={best_f1:.4f})")
        else:
            patience += 1
            logging.info(f"No improvement ({patience}/{patience_limit})")

        if patience >= patience_limit or epoch >= 30:
            logging.info("Early stopping.")
            break

    # ---------- TEST ----------
    logging.info("Evaluating best model on TEST set ...")
    model.load_state_dict(torch.load(best_model_path))
    test_f1 = evaluate(model, test_loader)
    logging.info(f"Final TEST F1: {test_f1:.4f}")
