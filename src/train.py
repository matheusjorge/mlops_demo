from data_collect import data_collect
from pre_processing import preprocessing
from modeling import modeling
from evaluation import evaluation
from utils import save_model

import logging
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
logging.basicConfig()

logger.info("Rodando coleta de dados")
data = data_collect()
logger.info("Coleta de dados executada")
logger.debug(f"A coleta tem {data.shape[0]} linhas e {data.shape[1]} colunas")

logger.info("Separando em treino e teste")
X_train, X_test, y_train, y_test = preprocessing(data)
logger.debug(f"Os dados de treino tem {X_train.shape[0]} coletas")
logger.debug(f"Os dados de teste tem {X_test.shape[0]} coletas")

logger.info("Selecionando e treinando modelo")
model = modeling(X_train, y_train)
logger.debug(f"O modelo selecionado é {model['model'].__class__.__name__}")

logger.info("Avaliando modelo")
accuracy = evaluation(model, X_test, y_test)
logger.info(f"O modelo teve {accuracy: .2%} de acurácia")

logger.info("Salvando modelo")
save_model(model, "models/model.pkl")
logger.info("Modelo salvo")