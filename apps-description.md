the python app will read data from the dataset (https://zenodo.org/records/10958775), filtering some of the features available for the 3 wind farms and will send the data through to the MQTT Broker. Based on some easy logic it will also produce on a different topic prediction of power produced and possible faults. HMI will connect to MQTT Broker and show real time data, based on specific topics.
Decide to switch to this dataset: https://www.kaggle.com/code/yohanesnuwara/iiot-wind-turbine-analytics as it is easier
Will use panda to manipulate dataset, reading in memory and writing (with sleep to simulate real data) to mosquitto.

Since we have labeled data we will use RandomForest instead of IsolationForest (with unbalanced data as explained in the python notebook). For reference read here: https://medium.com/sfu-cspmp/surviving-in-a-random-forest-with-imbalanced-datasets-b98b963d52eb
Raw deployment modelcar with all apps on microshift (https://developers.redhat.com/articles/2025/06/09/optimize-model-serving-edge-rawdeployment-mode)
https://docs.redhat.com/en/documentation/red_hat_build_of_microshift/4.20/html/using_ai_models/using-artificial-intelligence-with-microshift#microshift-rhoai-supported-models_microshift-rh-openshift-ai 

gemini chat: https://gemini.google.com/app/b0e22f87e46ba4ad

To detect anomalies decided to use isolation forest technique

Microshift: use version 4.20
Install OpenShift AI RPM: https://docs.redhat.com/en/documentation/red_hat_build_of_microshift/4.20/html/using_ai_models/using-artificial-intelligence-with-microshift#microshift-rhoai-install_microshift-rh-openshift-ai
$ sudo dnf install microshift-ai-model-serving
$  sudo dnf install microshift-ai-model-serving-release-info
$ oc create ns windguard
since OpenVINO doesn't support the following ONNX operations: ai.onnx.ml.TreeEnsembleRegressor, ai.onnx.ml.LabelEncoder I will have to build a custom inference server
    create the Containerfile embedding onnx in a OpenVINO model server and build it
    $ podman build -t quay.io/luferrar/windguard:model -f ContainerfileModel
    Create a ServingRuntime custom resource (CR) based on installed manifests and release information.
    Copy and modify the original ServingRuntime cr to local directory
    $ cp /usr/lib/microshift/manifests.d/050-microshift-ai-model-serving-runtimes/ovms-kserve.yaml ./ovms-kserve.yaml
    Create servingRuntime object
    $ oc create -n windguard -f ovms-kserve.yaml
    Create an InferenceService custom resource (CR) to instruct KServe how to create a deployment for serving your AI model. KServe uses the ServingRuntime based on the modelFormat value specified in the InferenceService CR. 
    $ oc create -n windguard -f inference-service.yaml

create a custom predictor for the isolation forest algorithm: https://github.com/hbelmiro/kserve-onnx-predictor-demo
to install on microshift follow this: https://github.com/hbelmiro/kserve-onnx-predictor-demo/tree/main?tab=readme-ov-file#deploying-to-kind-with-kserve

