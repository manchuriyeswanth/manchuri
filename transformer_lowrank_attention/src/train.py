import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from tqdm import tqdm
from data import get_dataloaders
from model import LowRankTransformerModel
from utils import compute_basis_from_activations

def train_one_epoch(model, dataloader, optimizer, device, mlflow_run=None):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad token id 0
    for xb, yb in tqdm(dataloader, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)  # [b,s,vocab]
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def eval_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--r', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_learn_basis', action='store_true')
    parser.add_argument('--mlflow_uri', type=str, default=None)
    args = parser.parse_args()

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment('lowrank-transformer')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(seq_len=args.seq_len, batch_size=args.batch_size)
    model = LowRankTransformerModel(vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, r=args.r, max_len=args.seq_len, learn_basis=args.use_learn_basis)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # If not learning basis, compute a basis by sampling some activations from training data
    if not args.use_learn_basis:
        print("Computing PCA/SVD basis from initial activations (quick sample)...")
        model.eval()
        acts = []
        with torch.no_grad():
            for i, (xb, _) in enumerate(train_loader):
                xb = xb.to(device)
                # get pre-projection activations (token embedding + pos)
                tok = model.tok_emb(xb)
                pos = model.pos_emb(torch.arange(xb.size(1), device=xb.device).unsqueeze(0).expand(xb.size(0), xb.size(1)))
                h = tok + pos
                # project to W_q space: apply W_q weight (d_model x d_model)
                # We'll sample first layer W_q applied to h
                Wq = model.layers[0].self_attn.W_q.weight.data.t()  # shape (d_model, d_model)
                A = (h @ Wq).reshape(-1, args.d_model).cpu()
                acts.append(A)
                if i >= 2:
                    break
        acts = torch.cat(acts, dim=0)
        B = compute_basis_from_activations(acts, r=args.r // args.nhead if args.r >= args.nhead else 1)
        # Our model expects B of shape (d_model, r_head) for shared basis
        model.layers[0].self_attn.set_basis(B)
        # set same basis for all layers
        for layer in model.layers[1:]:
            layer.self_attn.set_basis(B)

    # MLflow run
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = eval_loss(model, val_loader, device)
            mlflow.log_metrics({'train_loss': tr_loss, 'val_loss': val_loss}, step=epoch)
            print(f"Epoch {epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        # save model
        ckpt_path = 'model_last.pt'
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_artifact(ckpt_path)
        print("Training finished. Model saved and logged to MLflow.")

if __name__ == '__main__':
    main()
