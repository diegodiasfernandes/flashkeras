from models.FlashSequential import FlashSequential
from models.transferlearning.TransferLearning import TransferLearning
from keras.layers import Dense, InputLayer

input_shape = (125, 125, 3)

flash = FlashSequential(input_shape)
transfer = TransferLearning(input_shape, include_top=True, freeze=2)

flash.add(transfer.transferVGG16())

flash.summary()