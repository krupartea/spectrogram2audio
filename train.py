import torch
import dataset
from torch.utils.data import DataLoader
import model
from tqdm import tqdm
from hparams import *

model = model.model.to(DEVICE)
#model = torch.load(MODEL_PATH).to(DEVICE)

train_dataset=dataset.SpeechAndNoiseDataset('train', AUDIO_DIR, SAMPLE_RATE, N_SAMPLES, FRAME_OFFSET, DEVICE)
val_dataset=dataset.SpeechAndNoiseDataset('val', AUDIO_DIR, SAMPLE_RATE, N_SAMPLES, FRAME_OFFSET, DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_fn = torch.nn.MSELoss()

total_progress = tqdm(total=NUM_EPOCHS, desc='Total progress')
epoch_losses = []


for epoch in range(NUM_EPOCHS):

    model.train()
    epoch_progress = tqdm(total=len(train_loader), desc=f'Epoch {epoch}', leave=False)
    for data, target in train_loader:
        optimizer.zero_grad()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(target)
        loss = loss_fn(output, target)
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
            output = model(target)
            loss = loss_fn(output, target)
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


