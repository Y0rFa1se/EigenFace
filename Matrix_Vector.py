from math import sqrt

class Matrix:
    def __init__(self, array_2d):
        self.mat = array_2d
        
    def __str__(self):
        ret_string = ""
        for row in self.mat:
            for item in row:
                ret_string += f"{item} "
                
            ret_string += "\n"
            
        return ret_string
    
    def __len__(self):
        return len(self.mat)
    
    def __getitem__(self, row):
        return Vector(self.mat[row])
    
    def __setitem__(self, row, vec):
        self.mat[row] = list(vec)
    
    def __add__(self, matrix):
        mat = self.mat
        for y in range(len(mat)):
            for x in range(len(mat[0])):
                mat[y][x] += matrix[y][x]
                
        return Matrix(mat)
    
    def __sub__(self, matrix):
        mat = self.mat
        for y in range(len(mat)):
            for x in range(len(mat[0])):
                mat[y][x] -= matrix[y][x]
                
        return Matrix(mat)
    
    def __mul__(self, scalar):
        mat = self.mat[:]
        for y in range(len(mat)):
            for x in range(len(mat[0])):
                mat[y][x] *= scalar
                
        return Matrix(mat)
    
    def size(self):
        return (len(self.mat), len(self.mat[0]))
    
    def column(self, index):
        vec = []
        for i in range(len(self.mat)):
            vec.append(self.mat[i][index])
            
        return Vector(vec)
    
    def append(self, vec):
        mat = self.mat[:]
        mat.append(list(vec))
        
        return Matrix(mat)
    
    def col_append(self, vec):
        mat = self.mat[:]
        for i in range(len(mat)):
            mat[i].append(list(vec[i]))
            
        return Matrix(mat)
        
    def concate(self, matrix):
        mat = self.mat[:]
        for row in matrix:
            mat.append(list(row))
            
        return Matrix(mat)
    
    def col_concate(self, matrix):
        mat = Matrix(self.mat[:])
        for i in range(len(mat)):
            mat[i] = mat[i].concate(matrix[i])
                
        return mat
        
    def mat_mul(self, mat):
        ret = []
        for i in range(len(self.mat)):
            row_buffer = []
            for j in range(len(mat[0])):
                buffer = 0
                for k in range(len(self.mat[0])):
                    buffer += self.mat[i][k] * mat[k][j]
                
                row_buffer.append(buffer)
            ret.append(row_buffer)
            
        return Matrix(ret)
    
    def left_mat_mul(self, mat):
        return mat.mat_mul(self.mat)
    
    def vector_mul(self, vec):
        ret = []
        for i in range(len(self.mat)):
            buffer = 0
            for j in range(len(self.mat[0])):
                buffer += self.mat[i][j] * vec[j]
                
            ret.append(buffer)
            
        return Vector(ret)
                
    def transpose(self):
        mat = []
        for x in range(len(self.mat[0])):
            buffer = []
            for y in range(len(self.mat)):
                buffer.append(self.mat[y][x])
                
            mat.append(buffer)
                
        return Matrix(mat)
    
    def identity(self):
        mat = []
        for i in range(len(self.mat)):
            buffer = []
            for j in range(len(self.mat)):
                buffer.append(int(i == j))
                
            mat.append(buffer)
            
        return Matrix(mat)
    
    def permutation(self, a, b):
        mat = self.mat[:]
        mat[a], mat[b] = mat[b], mat[a]
        
        return Matrix(mat)
    
    def pm(self, a, b):
        id = self.identity()
        id = id.permutation(a, b)
        
        return id
    
    def row_multiply(self, index, multiplier):
        mat = Matrix(self.mat[:])
        mat[index] *= multiplier
        
        return mat
    
    def rmm(self, index, multiplier):
        rmm = self.identity()
        rmm[index][index] = multiplier
            
        return rmm
    
    def row_mul_add(self, a, b, multiplier):
        mat = Matrix(self.mat[:])
        mat[b] += mat[a] * multiplier
        
        return mat
    
    def rmam(self, a, b, multiplier):
        rmam = self.identity()
        rmam[b][a] = multiplier
            
        return rmam
    
    def rref(self):
        mat = Matrix(self.mat[:])
        row_cnt = 0
        pivot_idx = 0
        while (pivot_idx < len(mat[0])):
            for i in range(row_cnt, len(mat)):
                if mat[i][pivot_idx] != 0:
                    for j in range(len(mat)):
                        if i == j:
                            continue
                        
                        multiplier = (mat[j][pivot_idx] / mat[i][pivot_idx]) * (-1)
                        mat = mat.row_mul_add(i, j, multiplier)
                        
                    mat = mat.row_multiply(i, 1 / mat[i][pivot_idx])
                    mat = mat.permutation(i, row_cnt)
                    
                    row_cnt += 1
                    
                    break
                
            pivot_idx += 1
            
        return mat
    
    def rrefm(self):
        mat = Matrix(self.mat[:])
        rrefm = self.identity()
        row_cnt = 0
        pivot_idx = 0
        while (pivot_idx < len(mat[0])):
            for i in range(row_cnt, len(mat)):
                if mat[i][pivot_idx] != 0:
                    for j in range(len(mat)):
                        if i == j:
                            continue
                        
                        multiplier = (mat[j][pivot_idx] / mat[i][pivot_idx]) * (-1)
                        rrefm = rrefm.left_mat_mul(self.rmam(i, j, multiplier))
                        mat = mat.row_mul_add(i, j, multiplier)
                        
                    rrefm = rrefm.left_mat_mul(self.rmm(i, 1 / mat[i][pivot_idx]))
                    mat = mat.row_multiply(i, 1 / mat[i][pivot_idx])
                    rrefm = rrefm.left_mat_mul(self.pm(i, row_cnt))
                    mat = mat.permutation(i, row_cnt)
                    
                    row_cnt += 1
                    
                    break
                
            pivot_idx += 1
            
        return rrefm
    
class Vector:
    def __init__(self, array_1d):
        self.vec = array_1d
        
    def __str__(self):
        ret = ""
        for item in self.vec:
            ret += f"{item} "
            
        return ret
    
    def __len__(self):
        return len(self.vec)
    
    def __getitem__(self, index):
        return self.vec[index]
    
    def __setitem__(self, index, val):
        self.vec[index] = val
    
    def __add__(self, vector):
        vec = self.vec[:]
        for i in range(len(vec)):
            vec[i] += vector[i]
            
        return Vector(vec)
    
    def __sub__(self, vector):
        vec = self.vec[:]
        for i in range(len(vec)):
            vec[i] -= vector[i]
            
        return Vector(vec)
    
    def __mul__(self, scalar):
        vec = self.vec[:]
        for i in range(len(vec)):
            vec[i] *= scalar
            
        return Vector(vec)
    
    def __pow__(self, exp):
        vec = self.vec[:]
        for i in range(len(vec)):
            vec[i] **= exp
            
        return Vector(vec)
    
    def append(self, item):
        vec = self.vec[:]
        vec.append(item)
        
        return Vector(vec)
        
    def concate(self, vector):
        vec = self.vec[:]
        for item in vector:
            vec.append(item)
            
        return Vector(vec)
    
    def inner_product(self, vec):
        ret = 0
        for i in range(len(self.vec)):
            ret += self.vec[i] * vec[i]
            
        return ret
    
    def normalization(self):
        vec = self.vec[:]
        ssum = 0
        for i in vec:
            ssum += i ** 2
        ssum = sqrt(ssum)
            
        for i in range(len(vec)):
            vec[i] /= ssum
            
        return vec
    
    def absolute(self):
        absol = 0
        for item in self.vec:
            absol += item ** 2
            
        return sqrt(absol)