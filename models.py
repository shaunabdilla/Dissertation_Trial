import torch
from torch import nn
import torchvision
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)
        # print(list(resnet.children()))
         # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # print(out.shape)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# class Encoder(nn.Module):
# #VGG FACE with removed classification for use with attention
#     def __init__(self, encoded_image_size=14):
#         super(Encoder, self).__init__()
#         self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
#                      'std': [1, 1, 1],
#                      'imageSize': [224, 224, 3]}
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu1_1 = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu1_2 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu2_1 = nn.ReLU(inplace=True)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu2_2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_1 = nn.ReLU(inplace=True)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_2 = nn.ReLU(inplace=True)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_3 = nn.ReLU(inplace=True)
#         self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_1 = nn.ReLU(inplace=True)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_2 = nn.ReLU(inplace=True)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_3 = nn.ReLU(inplace=True)
#         self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_1 = nn.ReLU(inplace=True)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_2 = nn.ReLU(inplace=True)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_3 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         # self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
#         # self.relu6 = nn.ReLU(inplace=True)
#         # self.dropout6 = nn.Dropout(p=0.5)
#         # self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
#         # self.relu7 = nn.ReLU(inplace=True)
#         # self.dropout7 = nn.Dropout(p=0.5)
#         # self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

#         # self.fc = nn.Linear(in_features = 25088, out_features = 4096, bias=True)

#         self.fine_tune()

#     def forward(self, x0):
#         x1 = self.conv1_1(x0)
#         x2 = self.relu1_1(x1)
#         x3 = self.conv1_2(x2)
#         x4 = self.relu1_2(x3)
#         x5 = self.pool1(x4)
#         x6 = self.conv2_1(x5)
#         x7 = self.relu2_1(x6)
#         x8 = self.conv2_2(x7)
#         x9 = self.relu2_2(x8)
#         x10 = self.pool2(x9)
#         x11 = self.conv3_1(x10)
#         x12 = self.relu3_1(x11)
#         x13 = self.conv3_2(x12)
#         x14 = self.relu3_2(x13)
#         x15 = self.conv3_3(x14)
#         x16 = self.relu3_3(x15)
#         x17 = self.pool3(x16)
#         x18 = self.conv4_1(x17)
#         x19 = self.relu4_1(x18)
#         x20 = self.conv4_2(x19)
#         x21 = self.relu4_2(x20)
#         x22 = self.conv4_3(x21)
#         x23 = self.relu4_3(x22)
#         x24 = self.pool4(x23)
#         x25 = self.conv5_1(x24)
#         x26 = self.relu5_1(x25)
#         x27 = self.conv5_2(x26)
#         x28 = self.relu5_2(x27)
#         x29 = self.conv5_3(x28)
#         x30 = self.relu5_3(x29)
#         x31_preflatten = self.pool5(x30)
#         # x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
#         # x32 = self.fc6(x31)
#         # x33 = self.relu6(x32)
#         # x34 = self.dropout6(x33)
#         # x35 = self.fc7(x34)
#         # x36 = self.relu7(x35)
#         # x37 = self.dropout7(x36)
#         # x38 = self.fc8(x37)
#         out = self.adaptive_pool(x31_preflatten)  # (batch_size, 512, encoded_image_size, encoded_image_size)
#         out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)
#         # print(out.shape)
#         return out
#         # return x35

#     def fine_tune(self, fine_tune=False):
#       #Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

#       #:param fine_tune: Allow?
#       for p in self.parameters():
#           p.requires_grad = False
#       # Only fine tune last convolutional block
#       for c in list([self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5]):
#         # self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5
#           for p in c.parameters():
#               p.requires_grad = fine_tune
#       return

# class Encoder(nn.Module):
# #VGG FACE with removed classification for use with standard LSTM
#     def __init__(self, encoded_image_size=14):
#         super(Encoder, self).__init__()
#         self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
#                      'std': [1, 1, 1],
#                      'imageSize': [224, 224, 3]}
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu1_1 = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu1_2 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu2_1 = nn.ReLU(inplace=True)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu2_2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_1 = nn.ReLU(inplace=True)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_2 = nn.ReLU(inplace=True)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu3_3 = nn.ReLU(inplace=True)
#         self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_1 = nn.ReLU(inplace=True)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_2 = nn.ReLU(inplace=True)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu4_3 = nn.ReLU(inplace=True)
#         self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_1 = nn.ReLU(inplace=True)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_2 = nn.ReLU(inplace=True)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
#         self.relu5_3 = nn.ReLU(inplace=True)
#         self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
#         self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.dropout6 = nn.Dropout(p=0.5)
#         self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.dropout7 = nn.Dropout(p=0.5)
#         self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
#         self.fine_tune()

#     def forward(self, x0):
#         x1 = self.conv1_1(x0)
#         x2 = self.relu1_1(x1)
#         x3 = self.conv1_2(x2)
#         x4 = self.relu1_2(x3)
#         x5 = self.pool1(x4)
#         x6 = self.conv2_1(x5)
#         x7 = self.relu2_1(x6)
#         x8 = self.conv2_2(x7)
#         x9 = self.relu2_2(x8)
#         x10 = self.pool2(x9)
#         x11 = self.conv3_1(x10)
#         x12 = self.relu3_1(x11)
#         x13 = self.conv3_2(x12)
#         x14 = self.relu3_2(x13)
#         x15 = self.conv3_3(x14)
#         x16 = self.relu3_3(x15)
#         x17 = self.pool3(x16)
#         x18 = self.conv4_1(x17)
#         x19 = self.relu4_1(x18)
#         x20 = self.conv4_2(x19)
#         x21 = self.relu4_2(x20)
#         x22 = self.conv4_3(x21)
#         x23 = self.relu4_3(x22)
#         x24 = self.pool4(x23)
#         x25 = self.conv5_1(x24)
#         x26 = self.relu5_1(x25)
#         x27 = self.conv5_2(x26)
#         x28 = self.relu5_2(x27)
#         x29 = self.conv5_3(x28)
#         x30 = self.relu5_3(x29)
#         x31_preflatten = self.pool5(x30)
#         x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
#         # x32 = self.fc6(x31)
#         # x33 = self.relu6(x32)
#         # x34 = self.dropout6(x33)
#         # x35 = self.fc7(x34)
#         # x36 = self.relu7(x35)
#         # out = self.dropout7(x36)
#         # x38 = self.fc8(x37)
#         out = self.adaptive_pool(x31_preflatten)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
#         out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
#         # print(out.shape)
#         return out
#         # return x35

#     def fine_tune(self, fine_tune=True):
#       #Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

#       #:param fine_tune: Allow?
#       for p in self.parameters():
#           p.requires_grad = False
#       # If fine-tuning, only fine-tune convolutional block 5
#       for c in list([self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5]):
#         # self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5
#           for p in c.parameters():
#               p.requires_grad = fine_tune
#       return

# def vgg_face_dag(weights_path='vgg_face_dag.pth', **kwargs):
#     """
#     load imported model instance

#     Args:
#         weights_path (str): If set, loads model weights from the given path
#     """
#     model = Encoder()
#     # model.cuda()
#     # print(summary(model, (3, 224, 224)))
#     if weights_path:
#         state_dict = torch.load(weights_path)
#         model.load_state_dict(state_dict)
#     return model

# encoder = vgg_face_dag()

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        # print(f'{mean_encoder_out.shape} mean_encoder_out')
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # print(f'{h.shape} h')
        c = self.init_c(mean_encoder_out)
        # print(f'{c.shape} c')
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        # print(f'{batch_size} batch_size')
        encoder_dim = encoder_out.size(-1)
        # print(f'{encoder_dim} encoder_dim')
        # print(f'{encoder_out.shape} encoder_out')
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # print(f'{encoder_out.shape} encoder_out after flattening')
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        # print(f'{encoder_out.shape} encoder_out after sorting')
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t]) 
            # print(f'Max Alpha is {torch.max(alpha)}')
            # print(f'Att_Weighted_Enc is {attention_weighted_encoding}')

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha #This is the OG to be switched on

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind