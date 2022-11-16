
# REX-AI 0.0.2

### DISCLAIMER

Rex-AI is not a registered investment, legal or tax advisor or a broker/dealer. All investments or financial opinions expressed or predicted by the model are from personal research and experience of the owners of the site and are intended as educational material.

## What is Rex-AI

Rex-ai is a Forex Automated Trading System that, when fine-tuned correctly, could predict one hour ahead, control risk and execute the orders in a private forex service (read disclaimer)

The data cleaning and predictions are done using Pandas, Sklearn, TensorFlow (Keras) LSTM and RNN models and we intend to make it runable in a Google Cloud Service or Amazon Web Service virtual device to make it scalable.



## Installation and Run
To try Rex-AI, follow the instructions below

### clone
> git clone https://github.com/quantium-rock/rex-ai

### directory
> cd rex-ai/

### install
> python setup.py

### update database
This will update the database, taking into account the last hour, you only need to do this once.

> python db/bin/update_db.py



### run program
This will update the database, do the prediction, output the side and size signals and run the executor

> python first_run.py

...follow terminal instructions
