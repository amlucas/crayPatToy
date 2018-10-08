libCPToy:
	mkdir -p build;
	(cd build; make ..; make -j)

.PHONY: libCPToy
