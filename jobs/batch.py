import requests
from requests_ntlm import HttpNtlmAuth


visjob = {
  "ClusterId": "philly-prod-cy4",
  "VcId": "input",
  "JobName": "visjob_0_50000",
  "UserName": "ehazar",
  "BuildId": 70839,
  "ToolType": "cust",
  "ConfigFile": "ehazar/philly/mmod/runvis.py",
  "Inputs": [{
      "Name": "dataDir",
      "Path": "/hdfs/input/jianfw/data/qd_data"
    }
  ],
  "Outputs": [],
  "IsDebug": False,
  "CustomDockerName": "custom-msrccs-caffe-devel",
  "RackId": "anyConnected",
  "MinGPUs": 8,
  "PrevModelPath": None,
  "ExtraParams": "-range 0 50000 -p ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_nobb/train.prototxt -weights ehazar/trained/tax/Tax1300SGV1_1_darknet19_448_B_noreorg_extraconv2_tree_init3491_IndexLossWeight0_bb_nobb/snapshot/model_iter_244004.caffemodel",
  "SubmitCode": "p",
  "IsMemCheck": False,
  "IsCrossRack": False,
  "Registry": None,
  "RegistryUsername": None,
  "RegistryPassword": None,
  "Repository": None,
  "Tag": "default",
  "OneProcessPerContainer": True,
  "DynamicContainerSize": False,
  "NumOfContainers": "1",
  "CustomMPIArgs": None
}


session = requests.Session()
session.auth = HttpNtlmAuth('REDMOND\ehazar', raw_input("Enter password: "))

start = 0
while start < 1852119:
    end = start + 50000
    visjob["JobName"] = "visjob_{}_{}".format(start, end)
    visjob["ExtraParams"] = "-range {} {} ".format(start, end) + "-p ehazar/tax/Tax1300SGV1_1/Tax1300SGV1_1_darknet19_1_bb_nobb/train.prototxt -weights ehazar/trained/tax/Tax1300SGV1_1_darknet19_448_B_noreorg_extraconv2_tree_init3491_IndexLossWeight0_bb_nobb/snapshot/model_iter_244004.caffemodel"
    res = session.post("http://phillyonap/api/v2/submit", json=visjob)
    start = end
