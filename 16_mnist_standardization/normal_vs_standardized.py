import neural_network as nn
#import mnist as normal
import standardization_example as standardized


print("Standardized mnist")
nn.train(standardized.X_train, standardized.Y_train,
         standardized.X_validation, standardized.Y_validation, n_hidden_nodes = 200, iterations = 60, lr = 0.1)


