import torch

# Siempre van (filas, columnas)

mytensor = torch.zeros(5)

mytensor

mytensor2 = torch.ones(5,5)

mytensor2

my_tensor3 = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

my_tensor3
my_tensor3[:]
my_tensor3[1:]
my_tensor3[1, :] # Solo el segundo elemento
my_tensor3[0, :] # Solo el primer elemento

my_tensor3[-1, 0] # Ultima matriz y primer numero


short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
short_points

short_points2 = torch.ones(10, 2).to(dtype=torch.short)
short_points2

double_points = torch.zeros(10, 2).to(torch.double)
double_points

# Metodos vs Funciones de los Tensores

a = torch.ones(3, 2) # Funcion
a
a_t_f = torch.transpose(a, 0, 1) # Funcion
a_t_f

a_t = a.transpose(0, 1) # Metodo
a_t

a_t2 = a.transpose(0, 0) # Metodo
a_t2


# Ejercicio práctico tensors

a1 = torch.tensor([[2], [3], [4], [1]], dtype=torch.short)
a1

a2 = torch.tensor([[5], [-2], [3], [7]], dtype=torch.short)
a2

a_result = a1 + a2
a_result


# Pruebas Broadcasting


# Aquu lo hace por filas
a3 = torch.tensor([[2,2], [5,5]], dtype=torch.short)
a3

a4 = torch.tensor([[5], [4]], dtype=torch.short)
a4

a3 + a4


# Aqui lo hace por columnas
a5 = torch.tensor([[2,2], [5,5]], dtype=torch.short)
a5

a6 = torch.tensor([[5, 4]], dtype=torch.short)
a6

a5 + a6

# De todas formas, no hay que liarse mucho con esto que es mala praxis




# Aggregations of tensors

# Sumas agregadas

a = torch.tensor([[1, 2]], dtype=torch.int64)
a
a.sum()

a = torch.tensor([[1], [2]], dtype=torch.int64)
a
a.sum()


a = torch.tensor([[1,3], [2,7]], dtype=torch.int64)
a
a.sum()

# Comparaciones

b = torch.tensor([0, 4], dtype=torch.int64)
a < b

# Multiplicaciones

a = torch.tensor([[1, 2], [3, 2]])
b = torch.tensor([[3, 4], [3, 4]])
torch.matmul(a, b)

# Broadcasting en Multiplicaciones

a = torch.tensor([[3, 4], [3, 4]])
b = torch.tensor([1, 2])

torch.matmul(a, b)


# GPU vs CPU

torch.cuda.device_count()
cuda = torch.device('cuda') # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')


x = torch.tensor([1., 2.], device=cuda0)
print(f'x: {x}')
# x.device is device(type='cuda', index=0)



with torch.cuda.device(0):
    # allocates a tensor on GPU 0
    a = torch.tensor([1, 2], device=cuda)
    print(f'a:{a}')



points_gpu = torch.tensor([[4.0,1.0], [5.0,3.0], [2.0,1.0]], device=cuda)
points_gpu = 2 * points_gpu
print(points_gpu)