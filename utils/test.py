import torch
from torch.autograd import Variable

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    net.eval()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input, adj_matrix, annotation, target = Variable(init_input), Variable(adj_matrix), Variable(annotation), Variable(target)
        output = net(init_input, annotation, adj_matrix)

        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
