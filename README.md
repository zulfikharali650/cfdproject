# Computational Fluid Dynamics Project

# Project Overview

This project aims to develop and implement a hybrid machine-learning approach to predict particle trajectory and erosion distribution in an industrial-scale boiler header using GPT-2 and 3D CNN. The particle trajectories based on five initial parameters (i.e., particle size, main-inlet speed, main-inlet pressure, sub-inlet speed, and sub-inlet pressure) were predicted using GPT-2, followed by an erosion prediction based on the predicted trajectories from GPT-2. The GPT-2 hyperparameters were optimized for the best training efficiency and prediction performance. An initial parameter ranking system is implemented on the model's erosion predictions, suggesting the best initial parameters to minimize erosion. At the time of writing, this work is the first to implement GPT-2 in a hybrid method for erosion predictions in an industrial-scale boiler header.  

This project was a collaborative research with Steve D. Yang.

The data was generated using the Ansys Fluent software for the fluid dynamics erosion calculations on an OPT steam boiler header. The model used a cylindrical shaped header, where the 3D mesh contained 38,312 points. The system contained 196 particles and the simulation lasted for 50 times steps. 


