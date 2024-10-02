# DeepUWF
This is a repository of DeepUWF, a privacy-preserving deep learning framework for ultra-wide-field (UWF) fundus images. 
The framework is based on the decentralized federated learning paradigm, which allows multiple cohorts to collaboratively train a deep learning model without sharing their data.
It is implemented in Python using PyTorch, torchattacks, and paho-mqtt.

## Installation
### 1. Clone the repository
```bash
git clone git@github.com:Anduin1109/DeepUWF.git
```
### 2. Install the required python packages
```bash
pip install -r requirements.txt
```

## Docker
You can also run the code in a Docker container.
(To be finished later)

## Usage
### 1. Start the MQTT broker (take EMQX as an example)
#### Use public brokers
You can use public MQTT brokers for testing purposes. 
It is not recommended to use them in training or production environments.
```python
# broker address
broker_address = "broker.emqx.io"
broker_port = 1883
```

#### Broker running on the local machine
You can run the EMQX broker on your local machine (Take Ubuntu 20.04 as an example).
```bash
# download the EMQX broker
wget https://www.emqx.io/downloads/broker/v4.3.7/emqx-ubuntu20.04-v4.3.7_amd64.deb
# install the EMQX broker
sudo dpkg -i emqx-ubuntu20.04-v4.3.7_amd64.deb
# start the EMQX broker
sudo systemctl start emqx
```

#### Broker running with Docker
You can also directly run the EMQX broker with Docker.
```bash
docker run -d --name emqx -p 1883:1883 -p 8083:8083 -p 4369:4369 -p 8084:8084 -p 8883:8883 -p 18083:18083 emqx/emqx
```
### 2. Prepare the dataset
The dataset should be organized as follows:
```
dataset
├── cohort1
│   ├── train
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── test
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── cohort2
│   ├── ...
└── ...
```
### 3. Encrypt the dataset (Optional)
To encrypt the images, you can use the [**Recoverable privacy-preserving Image Classification (RIC)**](https://dl.acm.org/doi/full/10.1145/3653676) module.
We modified the original RIC module to make the ciphertexts available for model training. In this step, you need to change the working directory to [RIC](RIC) and follow the instructions in the [README.md](RIC/README.md) file.

### 4. Start the clients

### 5. Evaluation and Visualization
You can use the code in [visualize.ipynb](visualize.ipynb) to visualize both masked image modeling and class activation mapping.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.