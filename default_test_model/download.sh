#!/bin/bash
fileid="1ZAu0x9M6A_xAr-h_fCLjbZqSlB9WftNe"
filename="checkpoint.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}