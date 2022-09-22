import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


def train_bbox(model, train_loader, warm_start_epoch, epochs, conv_base_lr, lr_decay_rate, dense_lr, lr_decay_freq, name, num_classes, train_batch_size, optimizer='Adam', decay=True, MSE=True, lr_milestones=10):


    model = model.to(device)

    if optimizer=='SGD':
      optimizer = optim.SGD([
          {'params': model.features.parameters(), 'lr': conv_base_lr},
          {'params': model.classifier.parameters(), 'lr': dense_lr}],
          momentum=0.9
          )
    if optimizer=='Adam':
      optimizer = optim.Adam([
          {'params': model.features.parameters(), 'lr': conv_base_lr},
          {'params': model.classifier.parameters(), 'lr': dense_lr}]
          )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

#        val_loader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size,
#            shuffle=False, num_workers=num_workers)
    # for early stopping
#        count = 0
#        init_val_loss = float('inf')
    stats = {'epoch': [], "loss": []}
    train_losses = []
#        val_losses = []
    for epoch in range(warm_start_epoch, epochs):
        batch_losses = []
        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['annotations'].to(device).float()
            outputs = model(images)
            outputs = outputs.view(-1, num_classes, 1)

            optimizer.zero_grad()
            if MSE:
              #loss = emd_loss(labels, outputs)
              loss_func = torch.nn.MSELoss()
              loss = loss_func(outputs, labels)
            else:
              pinball_loss = AllQuantileLoss([0.05,0.95])
              loss = pinball_loss(outputs, labels)
            batch_losses.append(loss.item())
            #batch_losses.append(loss)

            loss.backward()

            optimizer.step()

            #print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, epochs, i + 1, len(train_loader.dataset) // train_batch_size + 1, loss.data[0]))
            #writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(train_loader.dataset) // train_batch_size + 1))

        avg_loss = sum(batch_losses) / (len(train_loader.dataset) // train_batch_size + 1)
        train_losses.append(avg_loss)
        print('Epoch %d mean training loss: %.4f' % (epoch + 1, avg_loss))
        print(conv_base_lr)
        stats['epoch'].append(epoch)
        stats['loss'].append(avg_loss)

        # exponetial learning rate decay
        if decay:
            if (epoch + 1) % lr_milestones == 0:
                conv_base_lr = conv_base_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                dense_lr = dense_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
                if optimizer=='SGD':
                  optimizer = optim.SGD([
                      {'params': model.features.parameters(), 'lr': conv_base_lr},
                      {'params': model.classifier.parameters(), 'lr': dense_lr}],
                      momentum=0.9
                      )
                if optimizer=='Adam':
                  optimizer = optim.Adam([
                      {'params': model.features.parameters(), 'lr': conv_base_lr},
                      {'params': model.classifier.parameters(), 'lr': dense_lr}]
                      )
                
    saved_final_state = dict(stats=stats,
                             model_state=model.state_dict(),
                             )
    torch.save(saved_final_state, name)

#            # do validation after each epoch
#            batch_val_losses = []
#            for data in val_loader:
#                images = data['image'].to(device)
#                labels = data['annotations'].to(device).float()
#                with torch.no_grad():
#                    outputs = model(images)
#                outputs = outputs.view(-1, 10, 1)
#                val_loss = emd_loss(labels, outputs)
#                batch_val_losses.append(val_loss.item())
#            avg_val_loss = sum(batch_val_losses) / (len(valset) // val_batch_size + 1)
#            val_losses.append(avg_val_loss)
#            print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
#            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)
#
#            # Use early stopping to monitor training
#            if avg_val_loss < init_val_loss:
#                init_val_loss = avg_val_loss
#                # save model weights if val loss decreases
#                print('Saving model...')
#                if not os.path.exists(ckpt_path):
#                    os.makedirs(ckpt_path)
#                torch.save(model.state_dict(), os.path.join(ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
#                print('Done.\n')
#                # reset count
#                count = 0
#            elif avg_val_loss >= init_val_loss:
#                count += 1
#                if count == early_stopping_patience:
#                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % early_stopping_patience)
#                    break

    print('Training completed.')

    '''
    # use tensorboard to log statistics instead
    if config.save_fig:
        # plot train and val loss
        epochs = range(1, epoch + 2)
        plt.plot(epochs, train_losses, 'b-', label='train loss')
        plt.plot(epochs, val_losses, 'g-', label='val loss')
        plt.title('EMD loss')
        plt.legend()
        plt.savefig('./loss.png')
    '''


def predict(model, test_loader, return_Y=False, num_classes=1):
    model = model.to(device)
    mean_preds = []
    y_true = []
    with torch.no_grad():
      model.eval()
      for data in test_loader:
          y_true.append(data['annotations'].cpu().numpy()[0])
          image = data['image'].to(device)
          output = model(image)
  #        output = output.view(num_classes, 1)
  #        predicted_mean = 0.0
  #        for i, elem in enumerate(output, 1):
  #            predicted_mean += i * elem
  #        mean_preds.append(predicted_mean) 
          mean_preds.append(output)
    
    if return_Y:
      return mean_preds, np.asarray(y_true)[:,0]
    else:
      return mean_preds
    

