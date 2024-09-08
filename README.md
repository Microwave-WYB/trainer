# Trainer

Are you still writing long, tedious training loop like this?

```python
model = MyModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 64
total_epochs = 10
patience = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs('./checkpoints', exist_ok=True)

start_epoch = load_checkpoint(model, optimizer, model_name)

best_accuracy = 0
early_stopping_counter = 0

for epoch in range(start_epoch, total_epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, targets)

        progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1),
                                    'acc': train_acc / (progress_bar.n + 1)})

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_acc += accuracy(outputs, targets)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(test_loader)
    val_acc /= len(test_loader)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save checkpoint
    save_checkpoint(model, optimizer, epoch+1, val_loss, val_acc, model_name)

    # Early stopping
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

print(f"Final {model_name} accuracy: {best_accuracy:.4f}")
```

Try trainer, achieving the same effect (even better) with simple syntax:

```python
model = MyModel()
batch_size = 64
total_epochs = 3

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=nn.NLLLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),  # type: ignore
    metrics=[Metric(compute=accuracy)],
    total_epochs=total_epochs,
    plugins=[
        ProgressBar(),
        EarlyStopping("loss", 3),
        Checkpoint(f"./checkpoints/{model_name}"),
    ],
)

try:
    load_latest_checkpoint_to_trainer(f"./checkpoints/{model_name}", trainer)
except FileNotFoundError:
    pass

for epoch_result in trainer:
    # Your custom logic between epochs here
    pass

print(f"Final {model_name} accuracy: {trainer.history[-1].val_metrics['accuracy']:.4f}")

```
