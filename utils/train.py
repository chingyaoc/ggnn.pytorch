import torch
from torch.autograd import Variable

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        net.zero_grad()

        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        init_input, adj_matrix, annotation, target = Variable(init_input), Variable(adj_matrix), Variable(annotation), Variable(target)
        output = net(init_input, annotation, adj_matrix)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data[0]))
