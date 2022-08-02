# neurips_rebuttal

## Experiment 3

!pip install pykeops
!pip install geomloss
!pip install pot

python run_experiment3_sweep.py

|   | **dataset** | **loss method** | **run time (s)** | **final loss** |
|--:|------------:|----------------:|-----------------:|---------------:|
| 0 |       grid9 |        geomloss |        31.484792 |       0.119040 |
| 1 |       grid9 |        analytic |        84.825071 |       0.119040 |
| 2 |      grid25 |        geomloss |        24.986482 |       0.230762 |
| 3 |      grid25 |        analytic |        78.162563 |       0.230762 |
| 4 |     circle8 |        geomloss |        24.785490 |       0.078548 |
| 5 |     circle8 |        analytic |        74.292535 |       0.078548 |
| 6 |        moon |        geomloss |        23.827746 |       0.314113 |
| 7 |        moon |        analytic |        38.605538 |       0.314113 |

# grid9 geomloss (top) vs analytic (bottom)
![image](https://user-images.githubusercontent.com/61277072/182481384-fb0c5ff0-0032-4d3c-9f29-a6f00a6848cc.png)
![image](https://user-images.githubusercontent.com/61277072/182481359-e2938331-b17a-49f8-9398-dac7ce9f407e.png)

# grid25 geomloss (top) vs analytic (bottom)
![image](https://user-images.githubusercontent.com/61277072/182481326-2ebd10ef-9926-4cfa-8698-f2244dc47d5f.png)

![image](https://user-images.githubusercontent.com/61277072/182481289-bdc89c7d-9ced-4107-936a-58e5d4ef73a7.png)

# circle8 geomloss (top) vs analytic (bottom)
![image](https://user-images.githubusercontent.com/61277072/182480948-6961fd62-fffd-4049-a21d-0b8e6d3a0e72.png)
![image](https://user-images.githubusercontent.com/61277072/182481043-8fc64924-0079-4d10-a236-52757edd175e.png)

# circle8 geomloss (top) vs analytic (bottom)
![image](https://user-images.githubusercontent.com/61277072/182481159-1eb6ed42-9165-4f48-93e1-081615f2a434.png)
![image](https://user-images.githubusercontent.com/61277072/182481205-c16c4219-e802-43dc-a46a-0b833d653f4f.png)
