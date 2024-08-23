// Referencias:
// https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf

#include <stdio.h>
#define N 512

__global__ void helloCUDA()
{
    printf("Hello, from CUDA!\n");
}

__global__ void add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();

    /// 
    /// Memoria
    ///

    int *a, *b, *c;           // cpu anfitrión (host)
    int *d_a, *d_b, *d_c;     // tarjeta dispositivo(device)
    int size = N * sizeof(int);

    // Solicitar espacio en la tarjeta
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Asignar espacio en el anfitrión
    // Llenar los vectores con valores aleatorios
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b ,N);
    // Espacio para el resultado
    c = (int *)malloc(size);


    ///
    /// Transferencia
    ///

    // Enviar datos a la tarjeta
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Ejecutar el kernel en la tarjeta con N bloques
    add<<<N,1>>>(d_a, d_b, d_c);

    // Copiar el resultado de regreso al anfitrión
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


    ///
    /// Limpieza
    ///
    free(a); free(b); free(c);                    // cpu
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);  // tarjeta

    return 0;
}

