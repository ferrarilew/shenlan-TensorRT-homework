export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model.plan | tee log.txt
# fp16
# python builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model.plan -f | tee log.txt
