import torch
from utils import accuracy, loss_fn
from tqdm import tqdm


def train_epoch(net, train_iter, test_iter, loss_fn, optimizer, num_epochs, devices):
    net = net.to(devices)
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0

        for batch, (X, y) in tqdm(enumerate(train_iter)):

            #Send data to device
            X, y = X.to(devices), y.to(devices)

            #Forward pass
            y_pred = net(X)

            #Compute loss
            loss = loss_fn(y_pred, y)

            #Optimizer zero
            optimizer.zero_grad()

            #Backward pass
            loss.sum().backward()

            #Optimize step
            optimizer.step()

            train_loss_sum += loss.sum()
            train_acc_sum += accuracy(y_pred, y)
            n += y.numel()


        # Testing 
        net.eval()
        test_loss_sum, test_acc_sum, m = 0.0, 0.0, 0
        with torch.inference_mode():
            for batch, (X, y) in tqdm(enumerate(test_iter)):

                #Send data to device
                X, y = X.to(devices), y.to(devices)

                #Forward pass
                y_pred = net(X)

                
                test_acc_sum += accuracy(y_pred, y)
                m+= y.numel()

#         

        print(f"Epoch: {epoch} | loss {train_loss_sum / len(train_iter)}, train_acc {train_acc_sum / n} |test_acc {test_acc_sum / m}")