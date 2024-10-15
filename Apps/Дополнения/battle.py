import requests
import json
def sendResults(user, groupId, taskId, model, history):
  loss = model.loss
  epochs = max(history.epoch)
  trainloss = min(history.history['loss'])
  valloss = min(history.history['val_loss'])
  weights = model.count_params()

  url = 'https://dnn.vb2005.ru/api/SendResults'
  modelInfo = {"validLoss" : valloss, "trainLoss" : trainloss, "lossFunc": loss, "epochs" : epochs, "weightsCount" : weights}
  userResult = {"username" : user, "groupId" : groupId, "taskId" : taskId, "modelInfo" : modelInfo}
  x = requests.post(url, json = userResult)
  if (x.status_code == 200):
    print('Результат записан')

