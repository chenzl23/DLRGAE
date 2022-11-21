import torch
import copy
import torch.nn.functional as F
from utils import accuracy


def rec_loss(estimated_A, mask):
    estimated_A_list = estimated_A[mask]
    rec_loss = - torch.log(estimated_A_list).mean()
    return rec_loss

def sp_loss(model):
    loss_U = 0
    for i in range(model.U_number):
        loss_U += (torch.norm(model.U[i]) ** 2)
    loss_U = 0.5 * loss_U * model.U_number

    return loss_U

def train(model, optimizer, graph, args):
    best_valid_acc = 0.0
    patience = args.patience
    best_model = copy.deepcopy(model)
    best_epoch = 0

    for epoch in range(1, args.epoch_num + 1):
        loss, valid_acc, model, optimizer = train_fullbatch(model, optimizer, graph, args)
        if (valid_acc >= best_valid_acc):
                best_valid_acc = valid_acc
                best_model = copy.deepcopy(model)
                best_epoch = epoch
        if args.early_stop:
            if (valid_acc >= best_valid_acc):
                patience = args.patience
            else:
                patience -= 1
                if (patience < 0):
                    print("Early Stopped!")
                    break
        with torch.no_grad():
            model.eval()
            Z, Z_knn, estimated_A, estimated_A_knn = model(graph)
            predictions = F.log_softmax(args.alpha * Z + (1 - args.alpha) * Z_knn, dim=1)
            train_acc = accuracy(predictions[graph.train_mask], graph.y[graph.train_mask])
        if args.verbose == 1:
            print("Epoch: {0:d}".format(epoch), 
                "Training loss: {0:1.5f}".format(loss.cpu().detach().numpy()), 
                "Training accuracy: {0:1.5f}".format(train_acc),
                "Valid accuracy: {0:1.5f}".format(valid_acc)
                )
    test_model = best_model
    with torch.no_grad():
        test_model.eval()
        Z, Z_knn, estimated_A, estimated_A_knn = test_model(graph)
        predictions = F.log_softmax(args.alpha * Z + (1 - args.alpha) * Z_knn, dim=1)
        accuracy_value = accuracy(predictions[graph.test_mask], graph.y[graph.test_mask])
    print("Best epoch:", str(best_epoch))
    print("Test accuracy: {0:1.5f}".format(accuracy_value))


def train_fullbatch(model, optimizer, graph, args):
    model.train()
    optimizer.zero_grad()
    Z, Z_knn, estimated_A, estimated_A_knn = model(graph)
    predictions = F.log_softmax(args.alpha * Z + (1 - args.alpha) * Z_knn, dim=1)
    
    mask_tp = graph.ori_adj.to_dense() > 0
    loss_tp = rec_loss(estimated_A, mask_tp)
    mask_knn = graph.ori_adj_knn.to_dense() > 0
    loss_knn = rec_loss(estimated_A_knn, mask_knn)
    loss_sp = sp_loss(model)

    loss_ce = F.nll_loss(predictions[graph.train_mask], graph.y[graph.train_mask])
    loss_rec = args.alpha * loss_tp + (1 - args.alpha) * loss_knn
    loss_sp = args.gamma * loss_sp
    loss = loss_ce + loss_rec + loss_sp
    
    loss.backward()
    optimizer.step()

    # Evaluation Valid Set
    with torch.no_grad():
        model.eval()
        Z, Z_knn, estimated_A, estimated_A_knn = model(graph)
        predictions = F.log_softmax(args.alpha * Z + (1 - args.alpha) * Z_knn, dim=1)
        valid_acc = accuracy(predictions[graph.valid_mask], graph.y[graph.valid_mask])
    
    return loss, valid_acc, model, optimizer