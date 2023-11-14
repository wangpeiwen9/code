import torch
from torchsummary import summary

from onekey_core.models.vaes import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')


class AutoVAE(BaseVAE):

    def __init__(self, feature_len: int, latent_dim: int, hidden_dims: List = None,
                 kernel_size=3, stride=2, **kwargs) -> None:
        super(AutoVAE, self).__init__()
        self.feature_len = feature_len
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims
        # Build Encoder
        in_channels = 1
        padding = kernel_size // 2
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * feature_len // (stride ** len(hidden_dims)), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * feature_len // (stride ** len(hidden_dims)), latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * feature_len // (stride ** len(hidden_dims)))
        self.mid_feature_len = feature_len // (stride ** len(hidden_dims))
        hidden_dims.reverse()

        # Build Decoder
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=kernel_size, stride=stride,
                                       padding=padding, output_padding=stride-1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], hidden_dims[-1], kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=stride -1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dims[-1], out_channels=1, kernel_size=kernel_size, padding=padding),
            nn.Tanh())
        # print(self.feature_len, self.mid_feature_len)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # print('En', z.shape)
        result = self.decoder_input(z)
        # print('Pre De', result.shape)
        result = result.view(-1, self.hidden_dims[-1], self.mid_feature_len)
        result = self.decoder(result)
        # print('De', result.shape)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # print(input.shape)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # print(mu.shape, log_var.shape, z.shape)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z


if __name__ == '__main__':
    f_len = 27146
    model = FlatVAE(f_len, 32, kernel_size=3, stride=1)
    x = torch.randn(16, 1, f_len)

    result = model(x)
    loss = model.loss_function(*result, M_N=0.005)
    print(loss)
