import neural_network_ex14 as nn
import data_handler as normal
import standardization_example as standardized

print("Original mnist")
nn.train(normal.X_train, normal.Y_train, normal.X_validation, normal.Y_validation,
         n_hidden_nodes = 200, epochs=2, batch_size = 60, lr = 0.1)

print("Standardized mnist")
nn.train(standardized.X_train, standardized.Y_train, standardized.X_validation, standardized.Y_validation,
         n_hidden_nodes = 200, epochs=2, batch_size = 60, lr = 0.1)


