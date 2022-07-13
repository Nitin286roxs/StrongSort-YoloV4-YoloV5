import sys
from openvino.inference_engine import IECore

xml_file = sys.argv[1]
bin_file = sys.argv[2]
device = sys.argv[3]
#Create object of core class
ie = IECore()
#Read network
net = ie.read_network(model=xml_file, weights=bin_file)
#load network
exec_net = ie.load_network(network=net, device_name=device, num_requests=1)
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
net.input_info[input_blob].precision = "FP32"
n,c = net.input_info[input_blob].input_data.shape
print(input_blob, output_blob, (n,c))

