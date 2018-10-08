class CPToy
{
public:
    CPToy(int size);
    ~CPToy();

    void cuda_test(float value);
    void mpi_test(float value);
    
private:
    int size;
    float *data;
};
