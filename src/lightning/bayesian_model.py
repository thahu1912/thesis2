import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.networks.imageretrievalnet import init_network
from losses.bayesian_triplet_loss import BayesianTripletLoss
from miners.cache_miner import CacheMiner
from torch.optim.lr_scheduler import ExponentialLR

class BayesianModel(pl.LightningModule):
    def __init__(self, config, savepath=None, seed=42):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize network
        self.net = init_network(config)
        
        # Set dropout rate to 0.2 for MC dropout
        self.dropout_rate = 0.2
        self._set_dropout_rate()
        
        # Initialize loss
        self.criterion = BayesianTripletLoss(
            margin=config.margin,
            varPrior=config.varPrior,
            kl_scale_factor=config.kl_scale_factor,
            distribution=config.distribution
        )
        
        # Initialize miner
        self.miner = CacheMiner(
            cache_size=config.cache_size,
            update_freq=config.cache_update_freq,
            distance=config.distance
        )
        
    def _set_dropout_rate(self):
        """Set dropout rate to 0.2 for all dropout layers"""
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train()  # Keep dropout active during inference
        
    def forward(self, x):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        
        # Form triplets: 1 anchor, 1 positive, 5 negatives
        triplets = self.miner.mine(embeddings, labels)
        
        if triplets is None:
            return None
            
        loss = self.criterion(embeddings, labels)
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = ExponentialLR(
            optimizer,
            gamma=self.config.optimizer.lr_gamma
        )
        
        return [optimizer], [scheduler]
        
    def mc_predict(self, x, n_samples=30):
        """Perform Monte Carlo dropout sampling for uncertainty estimation
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            mean_embeddings: Mean embeddings across MC samples
            std_embeddings: Standard deviation of embeddings (uncertainty)
        """
        self.eval()  # Set to eval mode but keep dropout active
        with torch.no_grad():
            # Collect samples
            samples = []
            for _ in range(n_samples):
                embeddings = self(x)
                samples.append(embeddings)
            
            # Stack samples and compute statistics
            samples = torch.stack(samples)
            mean_embeddings = samples.mean(dim=0)
            std_embeddings = samples.std(dim=0)
            
        return mean_embeddings, std_embeddings 