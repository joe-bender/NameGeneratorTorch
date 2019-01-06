# NameGeneratorTorch
Generate character names for the fantasy genre

### Description
Coming up with fantasy names can be hard because a good name should seem unusual but familiar at the same time. A neural network is a great tool for the job because it can learn the patterns of letters in names, like which letter should come after a certain series of other letters. Injecting a bit of randomness into this process can produce names that range from familiar to bizarre. Adjusting the amount of randomness is something the user can do to still feel like they have some input in this creative process, instead of letting the neural network do all the work.

### Usage
This app requires PyTorch and Numpy to be installed. Before running training.py, you'll need to download the zip file called 'National Data' from https://www.ssa.gov/oact/babynames/limits.html
This download contains a folder with many files with names like 'yob2017.txt'. Choose one of those, depending on which year's baby names you want to train from, and move it to the project's main directory. Make sure to put its name into the 'filename' setting inside of hyperparameters.py, where you can also adjust the other hyperparemeters before training. The model will be saved at the end of the number of epochs you specify. After training is finished, run inference.py and you will get a generated name. You can adjust the randomness of the name generation with the 'softmax_tuning' setting in hyperparameters.py if you don't like the results. Just keep running inference.py for more names. 
