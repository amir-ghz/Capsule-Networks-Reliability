
# Evaluating the reliability of capsule networks
In this project, the resiliency of the Capsule Network is analyzed and quantified under fault injection. The motivation for this project is to find out whether Capsule Networks are fault-resilient like the traditional Convolutional Neural Networks.

- First, a simple Capsule Network (For the detail, see [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf), Sara Sabour, Nicholas Frosst, Geoffrey E Hinton, NIPS 2017) is implemented in `PyTorch`. These models are trained under hours of training for 500 epochs using a single NVIDIA RTX 3060 GPU; what's more, the model tuned parameters are saved for convenient reuse! For instance, you can find the trained model for the `MNIST` dataset in `model-parameters.pt`.

- In the next step, a fault injection procedure is designed to uniformly distribute single bit flips (i.e. where there is logic '0' in a 32-bit value of the weight, this bit will be flipped to '1' and vice versa). The bit manipulation of tensors is done by several custom functions which are derived from [pytorch-binary-converter](https://github.com/KarenUllrich/pytorch-binary-converter) with little modification in order to satisfy our project. Specifically, the fault injection procedure is located in `utils.py` where the util function takes as input, a 32-bit IEEE 754 standard conventional weight tensor and outputs the whole tensor in binary format, so the bit manipulation begins!

  A thorough implementation is below to make a sense of what I've done programmatically:
  
     



 ```python
import random

number_of_bit_flips = 1000000 # Number of bits to flip deliberately

exponential_fault_growth = 113000000 # increase the value of number bits to flip until it reaches this value

while exponential_fault_growth < 169869311: # this number represents the number of bits in the whole weight tensor

    capsule_net.load_state_dict(torch.load('./model-parameters.pt'))

    capsule_net.eval()

    state_dict = capsule_net.state_dict()

    temp = torch.cat((state_dict['primary_capsules.capsules.0.weight'],
                      state_dict['primary_capsules.capsules.1.weight'],
                      state_dict['primary_capsules.capsules.2.weight'],
                      state_dict['primary_capsules.capsules.3.weight'],
                      state_dict['primary_capsules.capsules.4.weight'],
                      state_dict['primary_capsules.capsules.5.weight'],
                      state_dict['primary_capsules.capsules.6.weight'], 
                      state_dict['primary_capsules.capsules.7.weight']), dim=0)

    primary_caps_tensor = float2bit(temp, num_e_bits=8, num_m_bits=23, bias=127.)

    
    for i in range (exponential_fault_growth):

        random_bit_number = random.randint(0, 169869311)  

        if( torch.reshape(primary_caps_tensor, (-1,))[random_bit_number] == float(0) ):

            torch.reshape(primary_caps_tensor, (-1,))[random_bit_number] = float(1)

        if( torch.reshape(primary_caps_tensor, (-1,))[random_bit_number] == float(1) ):

            torch.reshape(primary_caps_tensor, (-1,))[random_bit_number] = float(0)


    print("Number of Faults Injected: ", exponential_fault_growth, '\n')

    exponential_fault_growth = exponential_fault_growth + 1000000 # Whether by addition or *n to increase exponentially

    float_tensor = bit2float(primary_caps_tensor, num_e_bits=8, num_m_bits=23, bias=127.)

    final_tensor = torch.split(float_tensor, 32)

    state_dict['primary_capsules.capsules.0.weight'] = final_tensor[0]
    state_dict['primary_capsules.capsules.1.weight'] = final_tensor[1]
    state_dict['primary_capsules.capsules.2.weight'] = final_tensor[2]
    state_dict['primary_capsules.capsules.3.weight'] = final_tensor[3]
    state_dict['primary_capsules.capsules.4.weight'] = final_tensor[4]
    state_dict['primary_capsules.capsules.5.weight'] = final_tensor[5]
    state_dict['primary_capsules.capsules.6.weight'] = final_tensor[6]
    state_dict['primary_capsules.capsules.7.weight'] = final_tensor[7]

    capsule_net.load_state_dict(state_dict)
    capsule_net.eval()
    caps_output, images, reconstructions = test(capsule_net, test_loader)

}
```


  By the time faults are injected randomly, the faulty weight values are loaded to the model itself for inference. 
## Results

Here you can see the results followed by my explaination of the conclusion I came to based on fault injection in Capsule Netwrok trained on `FMNIST` dataset.
 
 ![alt text](https://raw.githubusercontent.com/amir-ghz/Capsule-Networks-Reliability/11a1ae2f4c830ac3a01b043a77a709601fdf2e20/result1.svg)
 
 ![alt text](https://raw.githubusercontent.com/amir-ghz/Capsule-Networks-Reliability/11a1ae2f4c830ac3a01b043a77a709601fdf2e20/results.svg)
 
Recent results for analyzing the reliability of convolutional neural networks have demonstrated that not all SDCs are critical in object-detection/-classification tasks. If the present SDCs in the network do not affect detection and classification, they are characterized as benign or tolerable. However, there are certain types of SDCs that can impact the accuracy and lead to misclassification or leave an object undetected. Namely, these SDCs are the critical ones that were not masked. In the case of tolerable SDCs, this phenomenon is intuitively explained to be true because of the max-pooling layer; the faulty values act as redundant information which will not flow through the rest of the CNN architecture. Therefore, these SCDs are masked and are not decisive; thus, they will not stimulate the detection or classification task. In this project, it is shown that random bit flips in 47,185,920 million weight values of the Capsule Network can not cause a tangible decrease in classification precision. Unlike the intuition that Capsule Networks are not fault-resilient (maybe!) because they do not have any max-pooling layer and are architecturally different in comparison with CNNs, we can see from the results that not only is this new neural network architecture resilient to fault injection, but it has also shown more resiliency in comparison with traditional CNNs.
 
