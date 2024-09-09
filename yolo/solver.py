import torch
import time
from torch.cuda.amp import autocast, GradScaler

def solver(model, optimizer, loss_fn, train_loader, test_loader, epochs=1):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    model.to(device)

    # Set the solver parameters
    num_epochs = epochs
    total_time = 0

    # Training loop
    for epoch in range(num_epochs):

        # Iterate over the training dataset in batches
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Set the classifier to training mode
            model.train()

            start_time = time.time()

            # Move batch to GPU
            images = images.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs1, outputs2, outputs3 = model(images)

            # loss across all scales
            tloss1, bbloss1, hasobjloss1, noobjloss1, clsloss1 = loss_fn(outputs1, targets)
            tloss2, bbloss2, hasobjloss2, noobjloss2, clsloss2 = loss_fn(outputs2, targets)
            tloss3, bbloss3, hasobjloss3, noobjloss3, clsloss3 = loss_fn(outputs3, targets)
            tloss = tloss1 + tloss2 + tloss3
            bbloss = bbloss1 + bbloss2 + bbloss3
            hasobjloss = hasobjloss1 + hasobjloss2 + hasobjloss3
            noobjloss = noobjloss1 + noobjloss2 + noobjloss3
            clsloss = clsloss1 + clsloss2 + clsloss3

            # Backward pass
            tloss.backward()

            # Update the weights
            optimizer.step()

            end_time = time.time()
            total_time += end_time - start_time
            
            # Print training progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Total Loss: {tloss.item():.2f},  BB Loss: {bbloss.item():.2f}, Has object Loss: {hasobjloss.item():.2f}, No object Loss: {noobjloss.item():.2f}, Class Loss: {clsloss.item():.2f}, Time: {total_time:.2f}s')
                total_time = 0

            # Save model parameters every 1000 batches
            if (batch_idx + 1) % 1000 == 0:
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
