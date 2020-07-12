# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import Sequence
import numpy as np
import h5py


class VOCClassificationDataGenerator(Sequence):
    def __init__(self, db, bs, split = "train", shuffle = True, preprocessors = None, lb = None):
        # call the parent class constructor
        super(VOCClassificationDataGenerator, self).__init__()

        # initialize the instance variables
        self.db = h5py.File(db, "r")
        self.num_imgs = self.db.shape[0]
        self.bs = bs
        self.shuffle = shuffle
        self.preprocessors = preprocessors

        # initialize the appropriate label binarizer
        if split == "train":
            self.lb = LabelBinarizer()
            self.lb.fit(self.db["lbls"])
        else:
            # if lb is not supplied, raise an error
            if lb is None:
                raise ValueError(f"[ERROR] lb cannot be None when split is {split}")

            self.lb = lb

        # initialize a variable to store the indices
        self.indices = list(range(self.num_imgs))

    def __get_batch(self, start, end):
        # grab the current batch indices and sort them
        cur_indices = sorted(self.indices[start: end])

        # grab the images and the labels from the dataset
        imgs = self.db["imgs"][cur_indices]
        lbls = self.db["lbls"][cur_indices]

        # check if the images are to be preprocessed
        if self.preprocessors is not None:
            # loop through the images and preprocess them
            proc_imgs = []
            for img in imgs:
                for p in self.preprocessors:
                    img = p.preprocess(img)
                proc_imgs.append(img)

            # update the images
            imgs = np.array(proc_imgs)

        # one-hot encode the labels
        lbls = self.lb.transform(lbls)

        # return the batch
        return imgs, lbls

    def __len__(self):
        """
            Number of batches in an epoch
        """
        return np.floor(self.num_imgs / self.bs)

    def __getitem__(self, index):
        """
            Return the current batch of data
        """
        # calculate the starting and ending indices
        start = index * self.bs
        end = (index + 1) * self.bs

        # grab the current batch
        imgs, lbls = self.__get_batch(start, end)

        # return the current batch
        return imgs, lbls

    def on_epoch_end(self):
        """
            Reset the indices and (optionally) shuffle them
        """
        self.indices = list(range(self.num_imgs))
        if self.shuffle is True:
            np.random.shuffle(self.indices)
