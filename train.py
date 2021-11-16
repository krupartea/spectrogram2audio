import torch
import dataset
from torch.utils.data import DataLoader
import VAE
from tqdm import tqdm
from hparams import *

model = VAE.model.to(DEVICE)
#model = torch.load(MODEL_PATH).to(DEVICE)

train_dataset=dataset.SpeechAndNoiseDataset('train', AUDIO_DIR, SAMPLE_RATE, N_SAMPLES, FRAME_OFFSET, DEVICE)
val_dataset=dataset.SpeechAndNoiseDataset('val', AUDIO_DIR, SAMPLE_RATE, N_SAMPLES, FRAME_OFFSET, DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


def my_loss(output, target, mu, log_variance):
    reconstruction_loss = torch.mean((output - target)**2)
    kl_loss = -0.5 * torch.sum(1 + log_variance - torch.square(mu) -\
        torch.exp(log_variance))
    return reconstruction_loss + 0.001*kl_loss


optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_fn = my_loss

total_progress = tqdm(total=NUM_EPOCHS, desc='Total progress')
epoch_losses = []





for epoch in range(NUM_EPOCHS):

    model.train()
    epoch_progress = tqdm(total=len(train_loader), desc=f'Epoch {epoch}', leave=False)
    for data, target in train_loader:
        optimizer.zero_grad()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output, mu, log_variance = model(target)
        loss = loss_fn(output, target, mu, log_variance)
        loss.backward()
        optimizer.step()
        epoch_progress.update()
    epoch_progress.close()
    
    with torch.no_grad():
        model.eval()
        val_progress = tqdm(total=len(val_loader), desc=f'Epoch {epoch}. Validating...', leave=False)
        val_losses = []
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output, mu, log_variance = model(target)
            loss = loss_fn(output, target, mu, log_variance)
            val_losses.append(loss.item())
            val_progress.update()
        val_progress.close()    
        
    mean_val_loss = torch.mean(torch.tensor(val_losses))
    epoch_losses.append(mean_val_loss)

    if mean_val_loss == min(epoch_losses):
        torch.save(model, MODEL_PATH)
        
    scheduler.step(mean_val_loss)
    
    print(f"Val loss {epoch}: {epoch_losses[-1]}")

    total_progress.update()

total_progress.close()


