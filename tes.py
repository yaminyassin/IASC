teste1 = [[1, 1, 1, 1],
            [1, 0, 0, 1], #resultado esperado [1, 0]
            [0, 0, 0, 0],
            [0, 0, 0, 0]]

teste2 =[ [0, 0, 0, 0],
            [0, 0, 0, 0], #resultado esperado [0, 1]
            [0, 1, 1, 0], 
            [1, 0, 0, 1]]

teste3 =[ [1, 1, 0, 0],
            [1, 0, 0, 0], #resultado esperado [1, 0]
            [1, 0, 0, 0],
            [1, 1, 0, 0]]

teste4 =[ [1, 0, 0, 0],
            [0, 1, 0, 0],  #resultado esperado [0, 1]
            [0, 0, 1, 0],
            [0, 0, 0, 1]]

print("---------teste 1--------")
for i in range(len(teste1)):    
    print([''.join(str(x)) for x in teste1[i]]) 
print('classificacao esperada [1, 0]')

print("----------teste 2----------")
for i in range(len(teste1)):    
    print([''.join(str(x)) for x in teste2[i]]) 
print('classificacao esperada [0, 1]')

print("----------teste 3----------")
for i in range(len(teste1)):    
    print([''.join(str(x)) for x in teste3[i]]) 
print('classificacao esperada [1, 0]')

print("----------teste 4----------")
for i in range(len(teste1)):    
    print([''.join(str(x)) for x in teste4[i]]) 
print('classificacao esperada [0, 1]')