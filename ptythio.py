from transferlearning.TransferLearning import TransferLearning as ts

t = ts((224, 224, 3), False)

resnet = t.transferResnet50()

for layer in resnet.layers:
    print(type(layer))

resnet.summary()