#!/bin/bash
fileid="1pdkOtX1fvfQucxvaFLkWi7uxR6fq7xSW"
filename="checkpoint.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}