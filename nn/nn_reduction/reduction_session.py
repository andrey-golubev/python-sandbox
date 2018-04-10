model2 = Network(5, 20)
import copy
data_copy = copy.deepcopy(data)
data_copy['conv1.weight']
conv1_weights_array = np.array(data_copy['conv1.weight'])
conv1_biases_array = np.array(data_copy['conv1.bias'])
data_copy['conv1.weight'] = torch.from_numpy(conv1_weights_array[:5])
data_copy['conv1.bias'] = torch.from_numpy(conv1_biases_array[:5])
model2 = Network(5, 20)
conv2_weights_array = np.array(data_copy['conv2.weight'])
conv2_weights_array = np.array([row[:5] for row in conv2_weights_array])
data_copy['conv2.weight'] = torch.from_numpy(conv2_weights_array)
model2.load_state_dict(data_copy)
model2 = add_flops_counting_methods(model2)
model2.start_flops_count()
test(model2)
model2.compute_average_flops_cost()