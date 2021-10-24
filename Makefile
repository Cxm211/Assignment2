make build:
	nvcc -o goi_cuda sb/sb.cu exporter.cu goi_cuda.cu main.cu util.cu -rdc=true -lcudadevrt
