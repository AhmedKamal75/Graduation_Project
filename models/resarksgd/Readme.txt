arkface:
scale=30.0, margin=0.5


model:
embeddign=512
input_size=224
n_class=4605


training:
optimizer = torch.optim.SGD
scheduler = torch.optim.lr_scheduler.MultiStepLR
if epoch_val_loss < best_loss:
# golden formula: lr=0.1, lr_decay_epochs=[1.3], lr_decay_factor=0.5    
config = {
	'batch_size': 64,
	'learning_rate': 0.1,
	'momentum': 0.9,
	'weight_decay': 1e-4,
	'epochs': 20,
	'lr_decay_epochs': [1, 3],
	'lr_decay_factor': 0.5,
	'embedding_size': 512,
	'input_size': 224
    }


