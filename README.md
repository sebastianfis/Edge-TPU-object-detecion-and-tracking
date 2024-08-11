# Edge TPU detection and tracking web server

This repo contains an adaption of the object detection code from the google coral camera examples 
([original repo](https://github.com/google-coral/examples-camera.git)).
The following changes have been made:
* included code to stream the decection result via open-cv and flask
* included pre-compiled spaghettinet detection models (pre-trained on the COCO 2017 dataset) ready for usage on the edgetpu. 
If you're interested in how to get the models compiled, I highly recommend this guide: 
[Export "SpaghettiNet" to TF-Lite, Edge TPU Models using tf1](https://gist.github.com/NobuoTsukamoto/eade17835e57a02f5414aae907293707).
That did the trick for me.
* included code for persistent object tracking based on this example: [example-object-tracker](https://github.com/google-coral/example-object-tracker.git)
* removed all code using other frameworks (e.g. gstreamer)

Code has been tested on the coral dev board mini, but should run on any coral devicesuch as (no guarantees, though!): 
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board).

## Installation

1. First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2. Clone this Git repo onto your computer:

    ```
    mkdir Edge-TPU-object-detecion-and-tracking && cd Edge-TPU-object-detecion-and-tracking

    git clone https://github.com/sebastianfis/Edge-TPU-object-detecion-and-tracking.git
    ```

3. Install requirements:

    ```
    cd opencv

    bash install_requirements.sh
    ```

## Running the code

To run the detection server without object tracking, open a console on your device and run:
```
python3 opencv/detect_server.py --model all_models/spaghettinet_l_optimized_nms.tflite \
  --labels all_models/coco_labels.txt
```

To run the detection server with object tracking, open a console on your device and run:
```
python3 opencv/detect_server.py --model all_models/spaghettinet_l_optimized_nms.tflite \
  --labels all_models/coco_labels.txt --tracking sort
```

The code allows for the following optional flags:
* `--model <tflite model name>`: Name of the model to be used. Checkout the `/all_models/` directory for 
possible choices. Default is `spaghettinet_l_edgetpu.tflite`. Be sure to use the models labeled _edgetpu, as those are
compiled for the accelerator -  otherwise the model will run on the CPU and
be much slower.
* `--labels`: Name of the label file to be used. Models contained in the `/all_models/` directory 
are trained on 90 classes COCO data. Allows for localization of the class descriptors. Default 
is `coco_labels.txt`.
* `--top_k`: Number of objects with the highest score to display. Default value is 5.
* `--camera_idx`: Index of which video source to use. Default value is 1.
* `--threshold`: Threshold value for detection score. Only Scores above this vlaue will be considered valid 
detections. Default value is 0.5.
* `--tracker`: Name of the Object Tracker To be used. Default is `None`. Possible choices are 
`None` or `sort`.
* `--ip`: IP address of the streaming server. Default is `'0.0.0.0'`, meaning the stream will be set up at localhost.
* `--port`: Prot number on which to stream. Valid entries are between 1024 to 65535. It is recommended to use a port 
that is otherwise free (check your device port configuration). Default is 4664

While the detection server is running, you can access the stream through your web browser from any machine in the 
same network, by accessing `<IP adress>:<port number>`, as defined above. So e.g. if `detect_server.py` is running 
on the same machine, you can access the stream through the address `127.0.0.1:4664`.

On the Coral Dev Board Mini the code runs at ~ 0.09 s/frame, so framerates of ~10 fps are possible. 