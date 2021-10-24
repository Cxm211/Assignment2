example:
	nvcc -o goi_cuda.out sb/sb.cu exporter.cu cuda_thread.cu main.cu util.cu -rdc=true -lcudadevrt
