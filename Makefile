build:
	nvcc -o export.out sb/sb.cu exporter.cu cuda_thread.cu main.cu util.cu -rdc=true -lcudadevrt

clean:
	rm -f *.out *.gch