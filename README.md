# HALight
Official implementation of HAlight based on PyTorch .

## Environment Requirements

We recommend running the code on **Linux** systems.
 The experiments are conducted using **Python 3.8.20**.

All required Python dependencies are listed in the provided environment configuration files (`requirements.txt`). In addition, the following external libraries are required:

- **CityFlow**
- **HARL**

Please refer to the official repositories of [CityFlow](https://github.com/cityflow-project/CityFlow) and [HARL](https://github.com/PKU-MARL/HARL) for detailed installation instructions.


## Usage

The training-related hyperparameters can be configured in the `main.py` file of HALight.
 After setting the desired parameters, run the following commands to start training:

```
cd HALight
python main.py
```


## Datasets and Road Networks

The `Nets` directory contains road network files and traffic flow files in **CityFlow** and **SUMO** formats.

- **4×4** corresponds to the real-world **Gudang** roadnet.
- **3×3** corresponds to a **synthetic** roadnet.

Please note the following configuration requirements:

- When using the **Gudang** roadnet, the number of agents should be set to **16**.
- When using the **synthetic** roadnet, the number of agents should be set to **9**.

------

## Acknowledgements

Our implementation draws inspiration from and reuses components of the following open-source projects:

- **HARL**: https://github.com/PKU-MARL/HARL
- **RegionLight**: https://github.com/HankangGu/RegionLight
