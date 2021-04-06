import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention#, vgg_face_dag
from torchsummary import summary
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from pickle import dump
from pickle import load
import os
from torch.utils.tensorboard import SummaryWriter

#HyperParameter Optimization
from clearml import Task, Logger
task = Task.init(reuse_last_task_id = False, project_name='Facial Image Description Generation utilizing Face-To-Text dataset and transfer learning', task_name='Face2Text_GloVe_Attention_ResNet')
args = {'epochs': 1, 'batch_size': 24, 'encoder_lr': 1e-4, 'decoder_lr': 4e-4}
args = task.connect(args)  # enabling configuration override by clearml
print(args) # printing actual configuration (after override in remote mode)


# Data parameters
data_folder = './caption data'  # folder with data files saved by create_input_files.py
data_name = 'face2text_2_cap_per_img_2_min_word_freq'  # base name shared by data files
pretrained_embeddings_file = './glove.6B.50d.txt'
pretrained_embedding_matrix = 'embedding_matrix.pkl'

# Model parameters
emb_dim = 50  # dimension of word embeddings - look at GloVe
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 50  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 24
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
reload_pretrained_embed = True

tensorboard_writer = SummaryWriter('./tensorboard_logs')


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Load Pretrained Embeddings and compare to Wordmap if True, otherwise reload the pickle file
    if reload_pretrained_embed == True:  
      embeddings_index = dict()
      fid = open(pretrained_embeddings_file, encoding="utf8")
      for line in fid:
          values = line.split()
          word = values[0]
          coefs = np.asarray(values[1:], dtype='float32')
          embeddings_index[word] = coefs
      fid.close()

      pretrained_embeddings = torch.zeros((len(word_map) + 1, emb_dim))

      for word, idx in word_map.items():
          embed_vector = embeddings_index.get(word)
          if embed_vector is not None:
              # words not found in embedding index will be all-zeros.
              pretrained_embeddings[idx] = torch.from_numpy(embed_vector)
          else:
              pretrained_embeddings[idx] = torch.from_numpy(np.random.uniform(-1, 1, emb_dim))

    #   print(pretrained_embeddings[0:2, :])

    #   fid = open("embedding_matrix.pkl","wb")
    #   dump(pretrained_embeddings, fid)
    #   fid.close()

    # else:
    #   pretrained_embeddings = open(pretrained_embedding_matrix, "wb")
    #   print('Successfully Loaded Pretrained Embeddings Pickle')
    

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        # decoder.load_pretrained_embeddings(pretrained_embeddings)  # pretrained_embeddings should be of dimensions (len(word_map), emb_dim)
        # decoder.fine_tune_embeddings(True)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args['decoder_lr'])


        # encoder = vgg_face_dag() #VGG Face
        encoder = Encoder() #OG Encoder
        # encoder.cuda()
        # print(summary(encoder, (3, 224, 224)))
        # print('ENCODER SUMMARY')
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args['encoder_lr']) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr = args['encoder_lr'])

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) #OG figures
    # normalize = transforms.Normalize(mean= [129.186279296875, 104.76238250732422, 93.59396362304688], #VGG Face figures
    #                                  std= [1, 1, 1])
                     
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=args['batch_size'], shuffle=True, num_workers=workers, pin_memory=True)
    
    # print('validation_loader')
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=args['batch_size'], shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, args['epochs']):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 15
        if epochs_since_improvement == 15:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        # print('validation_loader_2')
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                epoch = epoch)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        tensorboard_writer.add_scalar('BLEU-4/epoch', recent_bleu4, epoch)

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

    PATH = './cifar_net.pth'
    tensorboard_writer.close()
    
    print('Task ID number is: {}'.format(task.id))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        # print(f'Alphas is {alphas}')
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data   #Replace this with the below two options
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data   #Replace this with the below two options
        
        # scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        # targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        
        # Calculate loss
        loss = criterion(scores, targets)

        # # Add doubly stochastic attention regularization this is on in the OG
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        
        if i % 5 == 0:
          Logger.current_logger().report_scalar(
            "train", "loss", iteration=(epoch * len(train_loader) + i), value=loss.item())
          Logger.current_logger().report_scalar(title='train', series='top_5_accuracy', value=top5, iteration=(epoch * len(train_loader) + i))
          tensorboard_writer.add_scalar('loss/epoch', loss, epoch)
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    # print('during_validation')

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data   #Replace this with the below two options
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data   #Replace this with the below two options
            
            # scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            # targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            

            # Calculate loss
            loss = criterion(scores, targets)

            # # Add doubly stochastic attention regularization This is on in the OG
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        Logger.current_logger().report_scalar(
          "val", "loss", iteration=(epoch * len(val_loader) + i), value=loss.item())
        Logger.current_logger().report_scalar(title='val', series='top_5_accuracy', value=top5, iteration=(epoch * len(val_loader) + i))
        Logger.current_logger().report_scalar(title='val', series='bleu4', value=bleu4, iteration=(epoch * len(val_loader) + i))

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4

if __name__ == '__main__':
    main()
