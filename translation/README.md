# Translation

**Steps to run the code:**
1. Install docker
2. Create a virtual environment with python 3.10.13
`conda create --name myenv python=3.10.13`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
`pip install -r requirements.txt`
5. Create a folder called `/data` under `translation/` and add review data from https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset
6. Create a folder called `/env` and add a file with the following:
    - postgres.env
    ```
    POSTGRES_DB=postgres
    POSTGRES_USER=admin
    POSTGRES_PASSWORD=root
    ```
6. Run the command `docker-compose up -d`
7. Run the command `python populate.py` to populate the vectordb with user reviews
8. Run the command `streamlit run app.py` to initiate our app and ask about the product
9. Or you can skip step 7. and 8. and run the notebook `translation.ipynb` 

## Folder Structure:
------------

    ├── translation
    │
    ├──────── base                                          <- Configuration class
    ├──────── classifier                                    <- Language Detector class
    ├──────── encoder                                       <- Encoder class
    ├──────── generator                                     <- Generator class
    ├──────── retriever                                     <- Retriever class
    ├──────── translator                                    <- Translator class
    ├──────── data                                          <- csv file
    ├──────── env                                           <- env files
    │
    │──── config.yaml                                       <- Config definition
    │──── lang_map.yaml                                     <- language mapping between XLM-RoBERTa and mBART
    │──── requirements.txt                                  <- package version for installing
    │
    │──── translation.ipynb                                 <- notebook
    │──── populate.py                                       <- python script to populate PGVector
    └──── app.py                                            <- streamlit application to chat with our LLM
--------
