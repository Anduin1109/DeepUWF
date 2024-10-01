# DeepUWF
This is a repository of DeepUWF, a privacy-preserving deep learning framework for ultra-wide-field (UWF) fundus images. 
The framework is based on the decentralized federated learning paradigm, which allows multiple cohorts to collaboratively train a deep learning model without sharing their data.
It is implemented in Python using PyTorch, torchattacks, and paho-mqtt.

## Installation
### 1. Clone the repository
```bash
git clone git@github.com:Anduin1109/DeepUWF.git
```
### 2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
### 1. Start the MQTT broker (take EMQX as an example, you can use public brokers to skip this step)
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
│   └── val
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── cohort2
│   ├── ...
└── ...
```
### 3. Start the clients
### 4. Visualization (optional)
### 5. Evaluation (optional)

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.