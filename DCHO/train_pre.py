import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.optim as optim

def pre_train(
        encoderModel,
        decoderModel,
        config,
        model_save_dir,
        train_dataloader,
):

    device=config['train']['device']

    encoder = encoderModel(config['encoder']).to(device)
    decoder = decoderModel(config['decoder']).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-10)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(config['train']['epochs']), desc="Epochs", position=0):

        epoch_loss_re = 0.0
        start_time = time.time()

        for X, A, H in tqdm(train_dataloader, desc="Batches train", position=1, leave=False):
            """
            X: (batch_size, 2*T, N, F_in)
            A: (batch_size, 2*T, N, N)
            H: (batch_size, 2*T, N, N, N)
            """
            X, A, H = X.to(device), A.to(device), H.to(device)


            zt = encoder(X, A)   # Zt   (batch_size, N, F_latent)
            re_Ht = decoder(zt)  # (B, T, M)

            loss_re = criterion(H, re_Ht)
 
            optimizer.zero_grad()
            loss_re.backward()
            optimizer.step()

            epoch_loss_re += loss_re.item()   # reconstruct


        avg_epoch_loss = epoch_loss_re / len(train_dataloader)
        scheduler.step(avg_epoch_loss)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            duration = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{config['train']['epochs']}] - {duration:.2f}s\n"
                f"total loss: {avg_epoch_loss:.4f}\n"
            )
            print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']:.10e}")

            save_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, save_path)
            print(f"Model saved to {save_path}")


