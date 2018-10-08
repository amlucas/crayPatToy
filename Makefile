libCPToy:
	mkdir -p build;
	(cd build; cmake ..; make -j)

.PHONY: libCPToy
