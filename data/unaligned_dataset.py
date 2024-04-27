import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from data.sam import get_sam_mask
import torch
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        # self.dir_Amask = os.path.join(opt.dataroot, opt.phase + 'A_mask')  # create a path '/path/to/data/trainA'
        # self.dir_Bmask = os.path.join(opt.dataroot, opt.phase + 'B_mask')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_mask_paths = sorted(make_dataset(self.dir_Amask, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_mask_paths = sorted(make_dataset(self.dir_Bmask, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_mask = get_transform(self.opt,grayscale=1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # Amask_path = self.A_mask_paths[index % self.A_size] 
        # Amask_path ='.'+str(A_path).split('.')[1]+'_mask.'+str(A_path).split('.')[2]
        # Amask_path = Amask_path.replace("trainA", "trainA_mask")
        parts = A_path.split('/')
        filename_parts = parts[-1].split('_')
        number = filename_parts[0]  # This is '012' in your example
        if int(number) >9  :
            formatted_number = '0'+ number  # Ensure the number is two digits
        else:
            formatted_number = number

        Amask_path =   '/'.join(parts[:-1]).replace("trainA", "trainA_mask")+'/'  + formatted_number + '_' + '_'.join(filename_parts[1:])
        # print(A_path, Amask_path)       
        Amask_path ='.'+str(Amask_path).split('.')[1]+'_mask.jpg'

        # print(A_path, Amask_path)

        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        parts = B_path.split('/')
        filename_parts = parts[-1].split('_')
        number = filename_parts[0]  # This is '012' in your example
        if int(number) >9  :
            formatted_number = '0'+ number  # Ensure the number is two digits
        else:
            formatted_number = number

        Bmask_path =   '/'.join(parts[:-1]).replace("trainB", "trainB_mask")+'/'  + formatted_number + '_' + '_'.join(filename_parts[1:])
        # print(A_path, Amask_path)       
        Bmask_path ='.'+str(Bmask_path).split('.')[1]+'_mask.jpg'
        # print(B_path, Bmask_path)

        # Bmask_path = self.B_mask_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # A_mask = get_sam_mask(A_img)
        A_mask = Image.open(Amask_path).convert('L')
        B_mask = Image.open(Bmask_path).convert('L')
        # apply image transformation

        A_m = self.transform_mask(A_mask)
        B_m = self.transform_mask(B_mask)
        A = self.transform_A(A_img)
        # print(self.transform_mask,self.transform_A)
        # print(A.shape,A_m.shape)#(1,x,x)
        B = self.transform_A(B_img)
        A = torch.cat((A,A_m),dim=0)#default resize_and_crop
        B = torch.cat((B,B_m),dim=0)
        # print(A.shape)#(4,256,256)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """

        return max(self.A_size, self.B_size)
