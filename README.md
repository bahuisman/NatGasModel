# Natural-Gas-Model
ABM market model based on the natural gas market written in Python. The model is created to run a simulation for roughly 35 years and represents a time frame of the year 2013 to 2050. Within the main directory the model is tailored to run a 5 year simulation while in the Long run section the model runs for the maximum time frame. The short run in the main directory is meant to keep the model runs modest in order to make development easier.

## .ipynb
The Notebook files contain the model and the visualization. The notebook files run Python code. Both in markdown blocks and within the code notes have been made to make reading the code easier and explain the thoughts behind the coding.

## .json
In order to communicate information of the model between the different files and to help the visualization of the model run results. The Json format (although based on JavaScript) is human readable in any text editor.

## .xlsx
The excel files are mainly used to import data and configure a model run. It contains parameterization and should follow a precise predefined format in order to be readable by the .ipynb files.

## .py
.py files are proof of concepts or not yet implemented functionality, made to function outside of the Jupyter Notebook environment.

Note:
This model is written as part of my graduation thesis and is currently being expanded. As part of the academic trajectory I record every major addition to the model in a separate Jupyter Notebook (.ipynb) file. The latest of these files is the most recent model.
