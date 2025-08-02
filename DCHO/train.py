import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.optim as optim
import pandas as pd

def train_H(
        encoderModel,
        pred_model,
        decoderModel,
        config,
        pre_model_path,
        model_save_dir,
        train_dataloader,
        test_dataloader,
):
    
    device=config['train']['device']

    # 初始化模型
    encoder = encoderModel(config['encoder']).to(device)
    decoder = decoderModel(config['decoder']).to(device)
    prediction = pred_model(config['pred_model']).to(device)

    checkpoint = torch.load(pre_model_path, map_location=device, weights_only=True)

    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False) # 只加载匹配的参数
    decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
    
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        list(prediction.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-10)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in tqdm(range(config['train']['epochs']), desc="Epochs", position=0):

        epoch_loss = 0.0

        epoch_loss_re1, epoch_loss_re2 = 0.0, 0.0

        epoch_loss_decoder_pred = 0.0
        epoch_loss_pred = 0.0

        start_time = time.time()

        for X, A, H in tqdm(train_dataloader, desc="Batches", position=1, leave=False):
            X, A, H = X.to(device), A.to(device), H.to(device)

            time_len = int(X.shape[1] / 2)
            Xt, At, Ht = X[:, : time_len], A[:, : time_len], H[:, : time_len]
            Xt_1, At_1, Ht_1 = X[:, time_len :], A[:, time_len :], H[:, time_len :]

            zt = encoder(Xt, At)          # Zt   (batch_size, N, F_latent)
            zt_1 = encoder(Xt_1, At_1)    # Zt+1 (batch_size, N, F_latent)

            re_Ht = decoder(zt)      # (B, T, M)
            re_Ht_1 = decoder(zt_1)  # (B, T, M)

            loss_re_t = criterion(Ht, re_Ht)
            loss_re_t_1 = criterion(Ht_1, re_Ht_1)

            zt_1_pred  = prediction(zt)
            loss_pred = criterion(zt_1, zt_1_pred)

            re_Ht_1_pred = decoder(zt_1_pred)
            loss_decoder_pred = criterion(Ht_1, re_Ht_1_pred)

            total_loss = loss_pred

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_loss_re1 += loss_re_t.item()   # reconstruct
            epoch_loss_re2 += loss_re_t_1.item() # reconstruct
            epoch_loss_pred += loss_pred.item()
            epoch_loss_decoder_pred += loss_decoder_pred.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        scheduler.step(avg_epoch_loss)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:

            duration = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{config['train']['epochs']}] - {duration:.2f}s\n"
                f"total: {avg_epoch_loss:.4f}\n"
                f"re1: {epoch_loss_re1 / len(train_dataloader):.4f}\n"
                f"re2: {epoch_loss_re2 / len(train_dataloader):.4f}\n"
                f"loss_pred: {epoch_loss_pred / len(train_dataloader):.4f}\n"
                f"decoder_pred: {epoch_loss_decoder_pred / len(train_dataloader):.4f}\n"
            )
            print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']:.10e}")

            metrics_summary = repeated_evaluate_H(encoder, decoder, prediction, test_dataloader, device, model_save_dir)
            print("Repeated Evaluation (10 runs):")
            for k, v in metrics_summary.items():
                print(f"{k}: {v:.4f}")
            
            save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'pred_state_dict': prediction.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, save_path)
            print(f"Model saved to {save_path}")


def compute_metrics(y_true, y_pred):

    y_true = y_true.float()
    y_pred = y_pred.float()

    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))

    nonzero_mask = y_true != 0
    if torch.any(nonzero_mask):
        mape = torch.mean(torch.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = torch.tensor(float('nan'))

    return {'MAE': mae.item(), 'RMSE': rmse.item(), 'MAPE': mape.item()}


def evaluate_H(encoder, decoder, prediction, test_dataloader, device):
    loss_mae = 0.0
    loss_rmse = 0.0
    loss_mape = 0.0

    total_metrics = {}

    for X, A, H in tqdm(test_dataloader, desc="Batches test", position=2, leave=False):
        time_len = int(X.shape[1] / 2)
        X, A, H = X.to(device), A.to(device), H.to(device)
        Xt, At, Ht = X[:, : time_len], A[:, : time_len], H[:, : time_len]
        Xt_1, At_1, Ht_1 = X[:, time_len :], A[:, time_len :], H[:, time_len :]

        zt = encoder(Xt, At)          # Zt   (batch_size, N, F_latent)

        zt_1_pred  = prediction(zt)

        pr_Ht_1 = decoder(zt_1_pred)

        loss_metrics = compute_metrics(y_true=Ht_1, y_pred=pr_Ht_1)

        loss_mae += loss_metrics["MAE"]
        loss_rmse += loss_metrics["RMSE"]
        loss_mape += loss_metrics["MAPE"]

    loss_mae = loss_mae / len(test_dataloader)
    loss_rmse = loss_rmse / len(test_dataloader)
    loss_mape = loss_mape / len(test_dataloader)
        
    total_metrics["MAE"] = loss_mae
    total_metrics["RMSE"] = loss_rmse
    total_metrics["MAPE"] = loss_mape

    return total_metrics



def repeated_evaluate_H(encoder, decoder, prediction, test_dataloader, device, csv_folder, repeat=10):
    maes, rmses, mapes = [], [], []
    results_per_run = []

    for i in range(repeat):
        metrics = evaluate_H(encoder, decoder, prediction, test_dataloader, device)
        maes.append(metrics["MAE"])
        rmses.append(metrics["RMSE"])
        mapes.append(metrics["MAPE"])
        results_per_run.append({
            "Run": i + 1,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"]
        })
    
    maes = torch.tensor(maes)
    rmses = torch.tensor(rmses)
    mapes = torch.tensor(mapes)

    result = {
        "MAE_mean": maes.mean().item(),
        "MAE_std": maes.std(unbiased=True).item(),
        "RMSE_mean": rmses.mean().item(),
        "RMSE_std": rmses.std(unbiased=True).item(),
        "MAPE_mean": mapes.mean().item(),
        "MAPE_std": mapes.std(unbiased=True).item()
    }

    results_per_run.append(result)

    csv_path = os.path.join(csv_folder, f"eval_H_results.csv")

    df = pd.DataFrame(results_per_run)
    df.to_csv(csv_path, index=False)
    print(f"[✔] Evaluation results saved to: {csv_path}")

    return result


