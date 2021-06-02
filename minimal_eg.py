import torch.multiprocessing as mp
from model import MyModel

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

# def train(rank, args, model, device, dataset, dataloader_kwargs):
#     torch.manual_seed(args.seed + rank)
#
#     train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
#
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#     for epoch in range(1, args.epochs + 1):
#         train_epoch(epoch, args, model, device, train_loader, optimizer)
#
if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()