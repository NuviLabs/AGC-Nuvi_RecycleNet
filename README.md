# AGC - Nuvi RecycleNet
Effective trash detection model for AI Grand Challenge 2020 track 3 round 1.  
Nuvilabs Solution.
The classes to be detected are:

 - combustion 
 - paper 
 - steel 
 - glass 
 - plastic 
 - plasticbag 
 - styrofoam 
 - food

## Usage
Download the pretrained model

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LnCw4L8RnM3qM96czxTTsOHn3xAKEpya' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LnCw4L8RnM3qM96czxTTsOHn3xAKEpya" -O ./model/model_checkpoint.pth && rm -rf /tmp/cookies.txt


Run inference

    python main.py --img_path ./sample_img.jpg
Will return a dictionary with the following format:
{'Annotations': [{'Label': 'steel', 'Bbox': [564, 383, 695, 443], 'Confidence': 0.9856872}]}
The bbox format follows: [xmin, ymin, xmax, ymax]
