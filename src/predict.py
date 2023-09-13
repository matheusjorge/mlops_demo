from utils import load_model
from data_collect import data_collect

import logging
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
logging.basicConfig()

logger.info("Carregando modelo")
model = load_model("models/model.pkl")
logger.debug(f"O modelo carregado é {model['model'].__class__.__name__}")

logger.info("Fazendo previsões")
data = data_collect().drop(columns="target")
preds = model.predict(data)
logger.debug(f"{preds.shape[0]} coletas previstas")