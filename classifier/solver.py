import torch
from utils import one_hot_encode
import time

def solver(model, optimizer, loss_fn, train_loader, test_loader, epochs=1, load_weights=True):

    # Call MP
    scaler = torch.cuda.amp.GradScaler()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Load the pre-trained weights
    if model.weights_path and load_weights:
        model.load_weights()

    model.to(device)

    # Set the solver parameters
    num_epochs = epochs
    total_time = 0

    # Training loop
    for epoch in range(num_epochs):
        # Set the classifier to training mode
        model.train()

        # Iterate over the training dataset in batches
        for batch_idx, (images, labels) in enumerate(train_loader):

            start_time = time.time()

            # One-hot encode the labels
            labels = one_hot_encode(labels)

            # Move batch to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()

            # Update the weights
            scaler.step(optimizer)
            scaler.update()

            end_time = time.time()
            total_time += end_time - start_time

            # Print training progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {total_time:.2f}s')
                total_time = 0

            # Save model parameters every 500 batches
            if (batch_idx + 1) % 500 == 0:
                model.save_weights()

    # Set the classifier to evaluation mode
    model.eval()

    # Evaluate the classifier on the test dataset
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

