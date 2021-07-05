import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,UNet_se,UNet_RLV,UNet_RLM,UNet_cbam
import csv

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type == 'U_Net':
			self.unet = U_Net(img_ch = 3, output_ch = 1)
		elif self.model_type == 'UNet_se':
			self.unet = UNet_se(img_ch = 3, output_ch = 1)
		elif self.model_type == 'UNet_cbam':
			self.unet = UNet_cbam(img_ch = 3, output_ch = 1)
		elif self.model_type == 'UNet_RLV':
			self.unet = UNet_RLV(img_ch = 3, output_ch = 1)
		elif self.model_type == 'UNet_RLM':
			self.unet = UNet_RLM(img_ch = 3, output_ch = 1)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)


	def print_network(self, model, name):
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s.pkl' %(self.model_type))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				PC = 0. 	# Precision
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth

					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					PC += get_precision(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
					length += images.size(0)

				acc = acc/length
				PC = PC/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, PC: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,PC,JS,DC))
				appendWrite(train_reslut,
							'\n Epoch: {}, Loss: {:.5f}, Acc: {:.5f}, pc:{:.5f},jc: {:.5f},dc:{:.5f}'.format(
								epoch+1, epoch_loss, acc,PC,JS,DC))
			

				# Decay learning rate
				if (epoch+1) % 50 ==0:
					lr = lr * 0.1
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr

				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				PC = 0. 	# Precision
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				for i, (images, GT) in enumerate(self.valid_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))
					acc += get_accuracy(SR,GT)
					PC += get_precision(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					length += images.size(0)
					
				acc = acc/length
				PC = PC/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC

				print('[Validation] Acc: %.4f, PC: %.4f,  JS: %.4f, DC: %.4f'%(acc,PC,JS,DC))
				appendWrite(valid_reslut,
							'\n Epoch: {},  Acc: {:.5f}, pc:{:.5f},jc: {:.5f},dc:{:.5f}'.format(
								epoch + 1, acc, PC, JS, DC))

				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)
					
			#===================================== Test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			PC = 0. 	# Precision
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT) in enumerate(self.valid_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				PC += get_precision(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				length += images.size(0)
					
			acc = acc/length
			PC = PC/length
			JS = JS/length
			DC = DC/length
			unet_score = JS + DC
			print('[test] Acc: %.4f, PC: %.4f, JS: %.4f, DC: %.4f, unet_score: %.4f' % (
			acc, PC,  JS, DC,unet_score))

			f = open(os.path.join(self.result_path,'result_UNet_RLM.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,PC,JS,DC])
			f.close()
			

			
