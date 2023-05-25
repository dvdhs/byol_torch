from rich.progress import track

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    epoch_loss = 0
    for batch, _ in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = model(batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.update_teacher()
    return epoch_loss / len(dataloader)

def train(model, optimizer, dataloader, device, epochs=100):
    for epoch in track(range(epochs)):
        epoch_loss = train_one_epoch(model, optimizer, dataloader, device)
        print(f"Epoch {epoch} loss: {epoch_loss}")
    return model